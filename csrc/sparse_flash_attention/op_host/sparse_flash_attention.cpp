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

#include "sparse_flash_attention.h"

#include "op_api_common.h"
#include <torch/library.h>

#include <array>
#include <string>
#include <tuple>

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kTndDimensions = 3;
constexpr int64_t kBsndDimensions = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor> make_outputs(const at::Tensor &query, const at::Tensor &key,
                                                            c10::string_view layout_query, c10::string_view layout_kv,
                                                            bool return_softmax_lse)
{
    const std::string query_layout(layout_query);
    const std::string kv_layout(layout_kv);
    TORCH_CHECK(query_layout == "BSND" || query_layout == "TND", "layout_query must be BSND or TND, but got ",
                query_layout);
    TORCH_CHECK(query.numel() > 0, "query must not be empty");
    TORCH_CHECK(key.numel() > 0, "key must not be empty");

    at::Tensor attention_out;
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    if (query_layout == "TND") {
        TORCH_CHECK(query.dim() == kTndDimensions, "query must be 3D for TND layout, but got ", query.dim(),
                    " dimensions");
        attention_out = at::empty(query.sizes(), query.options());
        if (return_softmax_lse) {
            TORCH_CHECK(key.dim() >= kTndDimensions,
                        "key must have at least 3 dimensions when return_softmax_lse is enabled");
            const int64_t kv_heads = kv_layout == "PA_BSND" ? key.size(2) : key.size(1);
            TORCH_CHECK(kv_heads > 0 && query.size(1) % kv_heads == 0,
                        "query head count must be divisible by KV head count");
            const std::array<int64_t, 3> softmax_shape = {kv_heads, query.size(0), query.size(1) / kv_heads};
            softmax_max = at::empty(softmax_shape, query.options().dtype(at::kFloat));
            softmax_sum = at::empty(softmax_shape, query.options().dtype(at::kFloat));
        }
    } else {
        TORCH_CHECK(query.dim() == kBsndDimensions, "query must be 4D for BSND layout, but got ", query.dim(),
                    " dimensions");
        attention_out = at::empty(query.sizes(), query.options());
        if (return_softmax_lse) {
            TORCH_CHECK(key.dim() >= kBsndDimensions,
                        "key must have at least 4 dimensions when return_softmax_lse is enabled");
            const int64_t kv_heads = key.size(2);
            TORCH_CHECK(kv_heads > 0 && query.size(2) % kv_heads == 0,
                        "query head count must be divisible by KV head count");
            const std::array<int64_t, 4> softmax_shape = {query.size(0), kv_heads, query.size(1),
                                                          query.size(2) / kv_heads};
            softmax_max = at::empty(softmax_shape, query.options().dtype(at::kFloat));
            softmax_sum = at::empty(softmax_shape, query.options().dtype(at::kFloat));
        }
    }

    if (!return_softmax_lse) {
        softmax_max = at::empty({0}, query.options().dtype(at::kFloat));
        softmax_sum = at::empty({0}, query.options().dtype(at::kFloat));
    }
    return {attention_out, softmax_max, softmax_sum};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> make_meta_outputs(const at::Tensor &query, const at::Tensor &key,
                                                                 c10::string_view layout_query,
                                                                 c10::string_view layout_kv, bool return_softmax_lse)
{
    const std::string query_layout(layout_query);
    const std::string kv_layout(layout_kv);
    TORCH_CHECK(query_layout == "BSND" || query_layout == "TND", "layout_query must be BSND or TND, but got ",
                query_layout);

    c10::SymDimVector output_shape;
    c10::SymDimVector softmax_shape;
    if (query_layout == "TND") {
        TORCH_CHECK(query.dim() == kTndDimensions, "query must be 3D for TND layout, but got ", query.dim(),
                    " dimensions");
        output_shape = {query.sym_size(0), query.sym_size(1), query.sym_size(2)};
        if (return_softmax_lse) {
            TORCH_CHECK(key.dim() >= kTndDimensions,
                        "key must have at least 3 dimensions when return_softmax_lse is enabled");
            const c10::SymInt kv_heads = kv_layout == "PA_BSND" ? key.sym_size(2) : key.sym_size(1);
            softmax_shape = {kv_heads, query.sym_size(0), query.sym_size(1) / kv_heads};
        }
    } else {
        TORCH_CHECK(query.dim() == kBsndDimensions, "query must be 4D for BSND layout, but got ", query.dim(),
                    " dimensions");
        output_shape = {query.sym_size(0), query.sym_size(1), query.sym_size(2), query.sym_size(3)};
        if (return_softmax_lse) {
            TORCH_CHECK(key.dim() >= kBsndDimensions,
                        "key must have at least 4 dimensions when return_softmax_lse is enabled");
            const c10::SymInt kv_heads = key.sym_size(2);
            softmax_shape = {query.sym_size(0), kv_heads, query.sym_size(1), query.sym_size(2) / kv_heads};
        }
    }

    if (!return_softmax_lse) {
        softmax_shape = {c10::SymInt(0)};
    }
    at::Tensor attention_out = at::empty_symint(output_shape, query.options());
    at::Tensor softmax_max = at::empty_symint(softmax_shape, query.options().dtype(at::kFloat));
    at::Tensor softmax_sum = at::empty_symint(softmax_shape, query.options().dtype(at::kFloat));
    return {attention_out, softmax_max, softmax_sum};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t attention_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "query must not be empty");
    TORCH_CHECK(key.numel() > 0, "key must not be empty");
    TORCH_CHECK(value.numel() > 0, "value must not be empty");
    TORCH_CHECK(sparse_indices.numel() > 0, "sparse_indices must not be empty");

    const std::string query_layout(layout_query);
    const std::string kv_layout(layout_kv);
    auto [attention_out, softmax_max, softmax_sum] =
        make_outputs(query, key, query_layout, kv_layout, return_softmax_lse);

    // The vLLM ACLNN adapter uses torch_npu's OpCommand and NPU caching
    // allocator, making its workspace valid during graph capture. The output
    // allocation above carries vLLM's PA_BSND DCP shape enhancement.
    char *query_layout_ptr = const_cast<char *>(query_layout.c_str());
    char *kv_layout_ptr = const_cast<char *>(kv_layout.c_str());
    EXEC_NPU_CMD(aclnnSparseFlashAttention, query, key, value, sparse_indices, block_table, actual_seq_lengths_query,
                 actual_seq_lengths_kv, query_rope, key_rope, scale_value, sparse_block_size, query_layout_ptr,
                 kv_layout_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse, attention_out,
                 softmax_max, softmax_sum);
    return {attention_out, softmax_max, softmax_sum};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparse_indices,
    double scale_value, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query, const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv, int64_t sparse_mode, int64_t pre_tokens,
    int64_t next_tokens, int64_t attention_mode, bool return_softmax_lse)
{
    return make_meta_outputs(query, key, layout_query, layout_kv, return_softmax_lse);
}

}  // namespace sglang::npu_kernel
