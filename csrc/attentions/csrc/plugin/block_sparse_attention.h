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

#ifndef SPARSE_BLOCK_ATTENTION_MINDIE_SD_IMPL_H
#define SPARSE_BLOCK_ATTENTION_MINDIE_SD_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

at::Tensor block_sparse_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_mask,
    const at::Tensor &sparse_count_table,
    std::string input_layout, int64_t sparse_size, int64_t num_heads,
    int64_t num_key_value_heads, double scale_value, bool causal,
    int64_t inner_precise, int64_t pre_tokens, int64_t next_tokens,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv);

#endif // SPARSE_BLOCK_ATTENTION_MINDIE_SD_IMPL_H