/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "block_sparse_attention.h"

using namespace at;

constexpr std::string_view BLOCK_SPARSE_ATTENTION_NAME = "aclnnBlockSparseAttention";

at::Tensor block_sparse_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                  const at::Tensor &sparse_mask, const at::Tensor &sparse_count_table,
                                  std::string input_layout, int64_t sparse_size, int64_t num_heads,
                                  int64_t num_key_value_heads, double scale_value, bool causal, int64_t inner_precise,
                                  int64_t pre_tokens, int64_t next_tokens, c10::OptionalIntArrayRef actual_seq_lengths,
                                  c10::OptionalIntArrayRef actual_seq_lengths_kv)
{
    TORCH_CHECK(input_layout != "TND", "input_layout currently does not support 'TND'.");
    at::Tensor attention_out =
        at_npu::native::empty_with_format(query.sizes(), query.options(), at_npu::native::get_npu_format(query));

    int64_t sparseMode = 0;
    const char *inputLayoutPtr = input_layout.c_str();

    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    auto actSeqLen = actual_seq_lengths.value_or(at::IntArrayRef{});
    auto actSeqLenKv = actual_seq_lengths_kv.value_or(at::IntArrayRef{});

    EXEC_NPU_CMD<BLOCK_SPARSE_ATTENTION_NAME>(
        query, key, value, nulltensor, nulltensor, actSeqLen, actSeqLenKv, nulltensor, nulltensor, nulltensor,
        nulltensor, nulltensor, sparse_mask, sparse_count_table, num_heads, scale_value, pre_tokens, next_tokens,
        inputLayoutPtr, num_key_value_heads, sparseMode, inner_precise, sparse_size, causal, attention_out);

    return attention_out;
}
