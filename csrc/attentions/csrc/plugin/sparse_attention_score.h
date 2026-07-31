/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
 */

#ifndef SPARSE_ATTENTION_SCORE_PLUGIN_H
#define SPARSE_ATTENTION_SCORE_PLUGIN_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>

at::Tensor npu_sparse_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &select_idx, const at::Tensor &block_table,
    const c10::optional<at::Tensor> &select_num_idx,
    const c10::optional<at::Tensor> &q_dequant_scale,
    const c10::optional<at::Tensor> &k_dequant_scale,
    const c10::optional<at::Tensor> &v_dequant_scale,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    std::string q_input_layout, std::string kv_input_layout,
    int64_t num_key_value_heads, double scale_value, int64_t block_size,
    int64_t top_k, int64_t inner_precise);

#endif  // SPARSE_ATTENTION_SCORE_PLUGIN_H
