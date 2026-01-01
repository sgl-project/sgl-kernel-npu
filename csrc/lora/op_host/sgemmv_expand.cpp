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
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_expand.cpp
 */

#include "defines.h"
#include "torch_helper.h"

#include "aclrtlaunch_sgemmv_expand_half.h"
#include "aclrtlaunch_sgemmv_expand_bfloat16_t.h"

namespace sglang {
namespace npu_kernel {

extern void sgemmv_expand_impl(at::ScalarType type, void *stream, void *x, void *weight, void *loraIndices,
                               uint32_t loraIndicesSize, void *seqLen, uint32_t seqLenSize, void *loraRanks,
                               uint32_t loraRanksSize, void *sliceOffsets, uint32_t sliceOffsetsSize, void *yIn,
                               void *yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank,
                               uint32_t outputFullDim)
{
    uint32_t slice_count = sliceOffsetsSize - 1;
    uint32_t blockDim = (batchSize * slice_count + numTokensPerCore - 1) / numTokensPerCore;
    if (type == at::ScalarType::Float) {
        return;
    } else if (type == at::ScalarType::BFloat16) {
        ACLRT_LAUNCH_KERNEL(sgemmv_expand_bfloat16_t)
        (blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize,
         sliceOffsets, sliceOffsetsSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputFullDim);
    } else {
        ACLRT_LAUNCH_KERNEL(sgemmv_expand_half)
        (blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize,
         sliceOffsets, sliceOffsetsSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputFullDim);
    }
}

HOST_API at::Tensor sgemmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                                  at::Tensor &lora_ranks, at::Tensor &slice_offsets, at::Tensor &y)
{
    at::ScalarType scalar_type = y.scalar_type();
    TORCH_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");

    at::Tensor y_out = y;
    void *x_ptr = x.data_ptr();
    void *weight_ptr = weight.data_ptr();
    void *y_ptr = y.data_ptr();
    void *y_out_ptr = y_out.data_ptr();

    void *lora_indices_ptr = lora_indices.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    void *seq_len_ptr = seq_len.data_ptr();
    int seq_len_size = seq_len.size(0);
    void *lora_ranks_ptr = lora_ranks.data_ptr();
    int lora_ranks_size = lora_ranks.size(0);
    void *slice_offsets_ptr = slice_offsets.data_ptr();
    int slice_offsets_size = slice_offsets.size(0);
    int slice_count = slice_offsets_size - 1;
    int batch_size = x.size(0);
    int max_lora_rank = x.size(1) / slice_count;
    int output_full_dim = y.size(1);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("sgemmv_expand");
    cmd.SetCustomHandler([scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr,
                          seq_len_size, lora_ranks_ptr, lora_ranks_size, slice_offsets_ptr, slice_offsets_size, y_ptr,
                          y_out_ptr, batch_size, max_lora_rank, output_full_dim]() -> int {
        int device_id = 0;
        int64_t aiv_num = 0;
        TORCH_CHECK(aclGetDeviceCapability(device_id, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv_num) == ACL_SUCCESS);
        int num_tokens_per_core = (batch_size + aiv_num - 1) / aiv_num;
        TORCH_CHECK(num_tokens_per_core != 0, "num_tokens_per_core should not be 0");
        sgemmv_expand_impl(scalar_type, stream, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr,
                           seq_len_size, lora_ranks_ptr, lora_ranks_size, slice_offsets_ptr, slice_offsets_size, y_ptr,
                           y_out_ptr, batch_size, num_tokens_per_core, max_lora_rank, output_full_dim);
        return 0;
    });
    cmd.Run();
    return y_out;
}

}  // namespace npu_kernel
}  // namespace sglang
