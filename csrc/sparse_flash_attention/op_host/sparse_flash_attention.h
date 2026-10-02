// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_H
#define SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_H

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

#include <tuple>

namespace sglang::npu_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t attention_mode, bool return_softmax_lse);

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t attention_mode, bool return_softmax_lse);

}  // namespace sglang::npu_kernel

#endif  // SGL_KERNEL_NPU_SPARSE_FLASH_ATTENTION_H
