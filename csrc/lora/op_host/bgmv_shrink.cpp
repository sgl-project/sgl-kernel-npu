/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/bgmv_shrink.cpp
 */

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_bgmv_shrink_half.h"
#include "aclrtlaunch_bgmv_shrink_bfloat16_t.h"

namespace sglang {
namespace npu_kernel {

extern void bgmv_shrink_impl(at::ScalarType type, void *stream, void *x, void *weight, void *indices,
                             uint32_t indicesSize, void *y, uint32_t batchSize, uint32_t numTokensPerCore,
                             uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    uint32_t blockDim = (batchSize + numTokensPerCore - 1) / numTokensPerCore;
    if (type == at::ScalarType::Float) {
        return;
    } else if (type == at::ScalarType::BFloat16) {
        ACLRT_LAUNCH_KERNEL(bgmv_shrink_bfloat16_t)
        (blockDim, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank,
         scale);
    } else {
        ACLRT_LAUNCH_KERNEL(bgmv_shrink_half)
        (blockDim, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank,
         scale);
    }
}

HOST_API void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices, at::Tensor &y, double scale)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(indices.dim() == 1, "indices should be [batch_size]");
    TORCH_CHECK(x.size(0) == y.size(0) && x.size(0) == indices.size(0),
                "the first dimension of x, y, indices should be same");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void *x_ptr = x.data_ptr();
    void *weight_ptr = weight.data_ptr();
    void *indices_ptr = indices.data_ptr();
    int indices_size = indices.size(0);
    void *y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t lora_rank = y.size(1);
    float scale_f = static_cast<float>(scale);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("bgmv_shrink");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size,
                          input_hidden_token, lora_rank, scale_f]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK(num_tokens_per_core != 0, "num_tokens_per_core should not be 0");
        bgmv_shrink_impl(scalar_type, stream, x_ptr, weight_ptr, indices_ptr, indices_size, y_ptr, batch_size,
                         num_tokens_per_core, input_hidden_token, lora_rank, scale_f);
        return 0;
    });
    cmd.Run();
    return;
}

}  // namespace npu_kernel
}  // namespace sglang
