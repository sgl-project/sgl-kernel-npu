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
#include "aclnn_sgl_hyper_conn_grouped_norm.h"
#include "op_host/hyper_conn_tiling.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using sglang::npu_kernel::hyper_conn::MAX_WIDTH;

void check_bf16_npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on an NPU device");
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

at::Tensor hyper_conn_grouped_norm(const at::Tensor &x, const at::Tensor &weight, c10::optional<int64_t> group_size,
                                   double eps, const c10::optional<at::Tensor> &out)
{
    check_bf16_npu(x, "x");
    check_bf16_npu(weight, "weight");
    const c10::DeviceGuard deviceGuard(x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions, got ", x.dim());
    TORCH_CHECK(weight.dim() == 1, "weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.device() == x.device(), "weight must be on x's device");

    const int64_t lastDim = x.size(-1);
    const int64_t width = group_size.has_value() ? group_size.value() : lastDim;
    TORCH_CHECK(width > 0 && width <= static_cast<int64_t>(MAX_WIDTH) && width % 8 == 0,
                "group size must be a positive multiple of 8 up to ", MAX_WIDTH, ", got ", width);
    TORCH_CHECK(lastDim % width == 0, "x's last dimension ", lastDim, " must be a multiple of the group size ", width);
    const int64_t weightLen = weight.numel();
    TORCH_CHECK(weightLen > 0 && weightLen % width == 0, "weight length ", weightLen,
                " must be a positive multiple of the group size ", width);
    const float epsF32 = static_cast<float>(eps);
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0 && std::isfinite(epsF32) && epsF32 > 0.0f,
                "eps must be positive and finite, got ", eps);

    const int64_t groups = x.numel() / width;
    TORCH_CHECK(groups % (weightLen / width) == 0, "the group count ", groups,
                " must be a whole number of weight periods (", weightLen / width, ")");

    at::Tensor result;
    if (out.has_value()) {
        check_bf16_npu(*out, "out");
        TORCH_CHECK(out->device() == x.device(), "out must be on x's device");
        TORCH_CHECK(out->sizes() == x.sizes(), "out must have x's shape");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
        TORCH_CHECK(!out->requires_grad(), "out must not require gradients");
        TORCH_CHECK(!tensors_overlap(*out, x) && !tensors_overlap(*out, weight),
                    "out must not alias or overlap x or weight");
        result = *out;
    } else {
        result = at::empty_like(x);
    }
    if (groups == 0) {
        return result;
    }

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglHyperConnGroupedNormGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnGroupedNormGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglHyperConnGroupedNorm)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnGroupedNorm"));

    auto srcDesc = sgl_cann::tensor_descriptor(x, ACL_BF16);
    auto weightDesc = sgl_cann::tensor_descriptor(weight, ACL_BF16);
    auto dstDesc = sgl_cann::tensor_descriptor(result, ACL_BF16);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(srcDesc.get(), weightDesc.get(), width, epsF32, dstDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "hyper_conn_grouped_norm workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{x, weight, result, workspace};
    for (auto &tensor : tensors) {
        if (tensor.numel() > 0) {
            tensor.record_stream(stream);
        }
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{srcDesc, weightDesc, dstDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto ret = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "hyper_conn_grouped_norm ACLNN execution failed: ", aclGetRecentErrMsg());
        return ret;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglHyperConnGroupedNorm", launch);
    return result;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hyper_conn_grouped_norm(Tensor x, Tensor weight, int? group_size, float eps, Tensor? out=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hyper_conn_grouped_norm", &hyper_conn_grouped_norm);
}
