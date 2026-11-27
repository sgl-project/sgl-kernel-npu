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
#include "aclnn_sgl_hyper_conn_scaled_silu.h"
#include "op_host/hyper_conn_tiling.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using sglang::npu_kernel::hyper_conn::MAX_WIDTH;

std::shared_ptr<aclTensor> flat_bf16_descriptor(const at::Tensor &tensor, int64_t span)
{
    int64_t stride = 1;
    auto *descriptor = aclCreateTensor(&span, 1, ACL_BF16, &stride, 0, ACL_FORMAT_ND, &span, 1, tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

at::Tensor hyper_conn_scaled_silu(const at::Tensor &x, double scale)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be on an NPU device");
    const c10::DeviceGuard deviceGuard(x.device());
    TORCH_CHECK(x.dim() == 2, "x must be 2-D, got ", x.dim(), "-D");
    TORCH_CHECK(x.stride(1) == 1, "x must have a unit column stride");
    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    TORCH_CHECK(cols > 0 && cols <= static_cast<int64_t>(MAX_WIDTH) && cols % 16 == 0,
                "x's column count must be a positive multiple of 16 up to ", MAX_WIDTH, ", got ", cols);
    TORCH_CHECK(rows == 0 || x.stride(0) >= cols, "x's row pitch ", x.stride(0), " must cover its ", cols, " columns");
    int exponent = 0;
    const float scaleF32 = static_cast<float>(scale);
    TORCH_CHECK(std::isfinite(scale) && scale > 0.0 && std::frexp(scale, &exponent) == 0.5 && std::isfinite(scaleF32) &&
                    scaleF32 > 0.0f,
                "scale must be a positive power of two, got ", scale);

    at::Tensor out = at::empty({rows, cols}, x.options());
    if (rows == 0) {
        return out;
    }

    const int64_t srcStride = x.stride(0);
    const int64_t srcSpan = (rows - 1) * srcStride + cols;

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglHyperConnScaledSiluGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnScaledSiluGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglHyperConnScaledSilu)>(
        sgl_cann::op_api_symbol("aclnnSglHyperConnScaledSilu"));

    auto srcDesc = flat_bf16_descriptor(x, srcSpan);
    auto dstDesc = sgl_cann::tensor_descriptor(out, ACL_BF16);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        queryWorkspace(srcDesc.get(), rows, cols, srcStride, scaleF32, dstDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "hyper_conn_scaled_silu workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, x.options().dtype(at::kByte));

    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{x, out, workspace};
    for (auto &tensor : tensors) {
        if (tensor.numel() > 0) {
            tensor.record_stream(stream);
        }
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors{srcDesc, dstDesc};
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto ret = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(ret == ACL_SUCCESS, "hyper_conn_scaled_silu ACLNN execution failed: ", aclGetRecentErrMsg());
        return ret;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglHyperConnScaledSilu", launch);
    return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hyper_conn_scaled_silu(Tensor x, float scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hyper_conn_scaled_silu", &hyper_conn_scaled_silu);
}
