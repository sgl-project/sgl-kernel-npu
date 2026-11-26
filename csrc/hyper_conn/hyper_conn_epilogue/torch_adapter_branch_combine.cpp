// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "aclnn_runtime.h"
#include "aclnn_sgl_hyper_conn_branch_combine.h"
#include "op_host/hyper_conn_tiling.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using sglang::npu_kernel::hyper_conn::MAX_BRANCHES;
using sglang::npu_kernel::hyper_conn::MAX_WIDTH;

void check_bf16_npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

std::shared_ptr<aclTensor> flat_bf16_descriptor(const at::Tensor &tensor, int64_t span)
{
    int64_t stride = 1;
    auto *descriptor = aclCreateTensor(&span, 1, ACL_BF16, &stride, 0, ACL_FORMAT_ND, &span, 1, tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

bool tensors_overlap(const at::Tensor &a, const at::Tensor &b)
{
    if (!a.defined() || !b.defined() || a.device() != b.device()) {
        return false;
    }
    if (a.unsafeGetTensorImpl() == b.unsafeGetTensorImpl()) {
        return true;
    }
    if (a.numel() == 0 || b.numel() == 0) {
        return false;
    }
    auto byteBounds = [](const at::Tensor &tensor) -> std::pair<uintptr_t, uintptr_t> {
        const auto base = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        const uintptr_t elemSize = tensor.element_size();
        if (tensor.is_contiguous()) {
            return {base, base + static_cast<uintptr_t>(tensor.numel()) * elemSize};
        }
        int64_t minElem = 0;
        int64_t maxElem = 0;
        for (int64_t d = 0; d < tensor.dim(); ++d) {
            const int64_t extent = (tensor.size(d) - 1) * tensor.stride(d);
            if (extent >= 0) {
                maxElem += extent;
            } else {
                minElem += extent;
            }
        }
        const uintptr_t begin = base - static_cast<uintptr_t>(-minElem) * elemSize;
        const uintptr_t end = base + static_cast<uintptr_t>(maxElem + 1) * elemSize;
        return {begin, end};
    };
    const auto [aBegin, aEnd] = byteBounds(a);
    const auto [bBegin, bEnd] = byteBounds(b);
    return aBegin < bEnd && bBegin < aEnd;
}

at::Tensor hyper_conn_branch_combine(const at::Tensor &block, const at::Tensor &residual, const at::Tensor &gate_logits,
                                     int64_t branches, const c10::optional<at::Tensor> &out)
{
    check_bf16_npu(block, "block");
    check_bf16_npu(residual, "residual");
    check_bf16_npu(gate_logits, "gate_logits");
    const c10::DeviceGuard deviceGuard(residual.device());
    TORCH_CHECK(residual.dim() == 2, "residual must be 2-D, got ", residual.dim(), "-D");
    TORCH_CHECK(block.dim() == 2, "block must be 2-D, got ", block.dim(), "-D");
    TORCH_CHECK(gate_logits.dim() == 2, "gate_logits must be 2-D, got ", gate_logits.dim(), "-D");
    TORCH_CHECK(block.is_contiguous() && residual.is_contiguous(), "block and residual must be contiguous");
    TORCH_CHECK(gate_logits.stride(1) == 1, "gate_logits must have a unit column stride");
    TORCH_CHECK(block.device() == residual.device() && gate_logits.device() == residual.device(),
                "all inputs must share one device");
    TORCH_CHECK(branches > 0 && branches <= static_cast<int64_t>(MAX_BRANCHES) && (branches & (branches - 1)) == 0,
                "branches must be a power of two in [1, ", MAX_BRANCHES, "], got ", branches);
    const int64_t rows = residual.size(0);
    TORCH_CHECK(residual.size(1) % branches == 0, "residual's column count ", residual.size(1),
                " must be a multiple of branches ", branches);
    const int64_t width = residual.size(1) / branches;
    TORCH_CHECK(width > 0 && width <= static_cast<int64_t>(MAX_WIDTH) && width % 16 == 0,
                "per-branch width must be a positive multiple of 16 up to ", MAX_WIDTH, ", got ", width);
    TORCH_CHECK(block.size(0) == rows && block.size(1) == width, "block must be [", rows, ", ", width, "]");
    TORCH_CHECK(gate_logits.size(0) == rows && gate_logits.size(1) == branches, "gate_logits must be [", rows, ", ",
                branches, "]");
    TORCH_CHECK(rows == 0 || gate_logits.stride(0) >= branches, "gate_logits' row pitch ", gate_logits.stride(0),
                " must cover its ", branches, " columns");

    at::Tensor result;
    if (out.has_value()) {
        check_bf16_npu(*out, "out");
        TORCH_CHECK(out->device() == residual.device(), "out must be on residual's device");
        TORCH_CHECK(out->sizes() == residual.sizes(), "out must have residual's shape");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
        TORCH_CHECK(!out->requires_grad(), "out must not require gradients");
        TORCH_CHECK(
            !tensors_overlap(*out, block) && !tensors_overlap(*out, residual) && !tensors_overlap(*out, gate_logits),
            "out must not alias or overlap any input tensor");
        result = *out;
    } else {
        result = at::empty_like(residual);
    }
    if (rows == 0) {
        return result;
    }

    const int64_t logitsStride = gate_logits.stride(0);
    const int64_t logitsSpan = (rows - 1) * logitsStride + branches;

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglHyperConnBranchCombineGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnBranchCombineGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglHyperConnBranchCombine)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnBranchCombine"));

    auto blockDesc = sgl_cann::tensor_descriptor(block, ACL_BF16);
    auto residualDesc = sgl_cann::tensor_descriptor(residual, ACL_BF16);
    auto logitsDesc = flat_bf16_descriptor(gate_logits, logitsSpan);
    auto outDesc = sgl_cann::tensor_descriptor(result, ACL_BF16);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = queryWorkspace(blockDesc.get(), residualDesc.get(), logitsDesc.get(), branches, logitsStride,
                                       outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "hyper_conn_branch_combine workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, residual.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{block, residual, gate_logits, result, workspace};
    for (auto &tensor : tensors) {
        if (tensor.numel() > 0) {
            tensor.record_stream(stream);
        }
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{blockDesc, residualDesc, logitsDesc, outDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto ret = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "hyper_conn_branch_combine ACLNN execution failed: ", aclGetRecentErrMsg());
        return ret;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglHyperConnBranchCombine", launch);
    return result;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "hyper_conn_branch_combine(Tensor block, Tensor residual, Tensor gate_logits, "
        "int branches, Tensor? out=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hyper_conn_branch_combine", &hyper_conn_branch_combine);
}
