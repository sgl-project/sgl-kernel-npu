/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
 */

#include "defines.h"
#include "tiling/sgemmc_tiling.h"
#include "torch_helper.h"

#include "aclrtlaunch_sgemmc_shrink.h"

namespace sglang {
namespace npu_kernel {

HOST_API void sgemmc_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices, at::Tensor &seq_len,
                            at::Tensor &lora_ranks, at::Tensor &lora_scales, at::Tensor &y)
{
    at::ScalarType scalar_type = x.scalar_type();
    TORCH_CHECK(scalar_type == at::kHalf || scalar_type == at::kBFloat16, "only support half and bf16");
    TORCH_CHECK(x.dim() == 2, "x should be [batch_size, hidden_in]");
    TORCH_CHECK(weight.dim() == 3 || weight.dim() == 4,
                "weight should be [num_loras, hidden_out, hidden_in] or [num_loras, 1, hidden_out, hidden_in]");
    TORCH_CHECK(y.dim() == 2, "y should be [batch_size, hidden_out]");
    TORCH_CHECK(x.size(1) > y.size(1), "hidden in should be greater than hidden out");
    void *x_ptr = x.data_ptr();
    void *weight_ptr = weight.data_ptr();

    void *lora_indices_ptr = lora_indices.data_ptr();
    int lora_indices_size = lora_indices.size(0);
    void *seq_len_ptr = seq_len.data_ptr();
    int seq_len_size = seq_len.size(0);
    void *lora_ranks_ptr = lora_ranks.data_ptr();
    int lora_ranks_size = lora_ranks.size(0);
    void *lora_scales_ptr = lora_scales.data_ptr();
    int lora_scales_size = lora_scales.size(0);

    void *y_ptr = y.data_ptr();
    int batch_size = x.size(0);
    int input_hidden_token = x.size(1);
    uint32_t max_lora_rank = y.size(1);

    uint32_t block_dim;
    uint32_t workspace_size;
    int64_t total_extend_tokens = 0;
    int64_t num_tokens_per_core = 0;

    at::Tensor tiling_tensor = GenerateTiling(block_dim, workspace_size, batch_size, input_hidden_token, max_lora_rank,
                                              TorchNpuHelper::ConvertDataType(scalar_type));

    auto workspace_tensor =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));
    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(sgemmc_shrink, block_dim, x_ptr, weight_ptr, lora_indices_ptr, lora_indices_size, seq_len_ptr,
                    seq_len_size, lora_ranks_ptr, lora_ranks_size, lora_scales_ptr, lora_scales_size, y_ptr, batch_size,
                    num_tokens_per_core, input_hidden_token, max_lora_rank, workspace_tensor, tiling_tensor);
    return;
}

}  // namespace npu_kernel
}  // namespace sglang
