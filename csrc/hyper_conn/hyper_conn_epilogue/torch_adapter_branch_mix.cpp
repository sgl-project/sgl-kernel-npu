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
#include "aclnn_sgl_hyper_conn_branch_mix.h"
#include "op_host/hyper_conn_tiling.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

using sglang::npu_kernel::hyper_conn::MAX_BRANCHES;
using sglang::npu_kernel::hyper_conn::MAX_WIDTH;

void check_bf16_npu(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be on an NPU device");
}

at::Tensor hyper_conn_branch_mix(const at::Tensor &normed, const at::Tensor &gate_logits, int64_t branches)
{
    check_bf16_npu(normed, "normed");
    check_bf16_npu(gate_logits, "gate_logits");
    const c10::DeviceGuard deviceGuard(normed.device());
    TORCH_CHECK(normed.dim() == 2, "normed must be 2-D, got ", normed.dim(), "-D");
    TORCH_CHECK(normed.sizes() == gate_logits.sizes(), "gate_logits must have normed's shape");
    TORCH_CHECK(normed.is_contiguous() && gate_logits.is_contiguous(), "normed and gate_logits must be contiguous");
    TORCH_CHECK(gate_logits.device() == normed.device(), "gate_logits must be on normed's device");
    TORCH_CHECK(branches > 0 && branches <= static_cast<int64_t>(MAX_BRANCHES) && (branches & (branches - 1)) == 0,
                "branches must be a power of two in [1, ", MAX_BRANCHES, "], got ", branches);
    const int64_t rows = normed.size(0);
    TORCH_CHECK(normed.size(1) % branches == 0, "normed's column count ", normed.size(1),
                " must be a multiple of branches ", branches);
    const int64_t width = normed.size(1) / branches;
    TORCH_CHECK(width > 0 && width <= static_cast<int64_t>(MAX_WIDTH) && width % 16 == 0,
                "per-branch width must be a positive multiple of 16 up to ", MAX_WIDTH, ", got ", width);

    at::Tensor out = at::empty({rows, width}, normed.options());
    if (rows == 0) {
        return out;
    }

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglHyperConnBranchMixGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnBranchMixGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglHyperConnBranchMix)>(sgl_cann::op_api_symbol("aclnnSglHyperConnBranchMix"));

    auto normedDesc = sgl_cann::tensor_descriptor(normed, ACL_BF16);
    auto logitsDesc = sgl_cann::tensor_descriptor(gate_logits, ACL_BF16);
    auto outDesc = sgl_cann::tensor_descriptor(out, ACL_BF16);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(normedDesc.get(), logitsDesc.get(), branches, outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "hyper_conn_branch_mix workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, normed.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{normed, gate_logits, out, workspace};
    for (auto &tensor : tensors) {
        if (tensor.numel() > 0) {
            tensor.record_stream(stream);
        }
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{normedDesc, logitsDesc, outDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto ret = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "hyper_conn_branch_mix ACLNN execution failed: ", aclGetRecentErrMsg());
        return ret;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglHyperConnBranchMix", launch);
    return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hyper_conn_branch_mix(Tensor normed, Tensor gate_logits, int branches) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hyper_conn_branch_mix", &hyper_conn_branch_mix);
}
