/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * sgl-kernel-npu/svoloch2940194 is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef SPARSE_BLOCK_ESTIMATE_IMPL_H
#define SPARSE_BLOCK_ESTIMATE_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> sparse_block_estimate(
    const at::Tensor &query, const at::Tensor &key,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv,
    std::string input_layout, int64_t stride, int64_t sparse_size, int64_t num_heads,
    int64_t num_key_value_heads, double scale_value, double threshold,
    bool causal, bool keep_sink, bool keep_recent, double row_sparse);

#endif // SPARSE_BLOCK_ESTIMATE_IMPL_H