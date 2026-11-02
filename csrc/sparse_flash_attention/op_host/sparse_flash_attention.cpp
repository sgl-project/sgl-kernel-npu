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

#include <array>
#include <memory>
#include <string>
#include <tuple>

#include "aclrtlaunch_sgl_sparse_flash_attention.h"
#include "ge_helper.h"
#include "sparse_flash_attention_def.h"
#include "tiling/sparse_flash_attention_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kTndDimensions = 3;
constexpr int64_t kBsndDimensions = 4;

using ge_helper::TilingContext;

void CheckTensor(const at::Tensor &tensor, const at::Tensor &query, const char *name)
{
    TORCH_CHECK(tensor.device().type() == query.device().type() && tensor.device().index() == query.device().index(),
                name, " must be on the same device as query");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &query, const char *name)
{
    if (tensor.has_value()) {
        CheckTensor(*tensor, query, name);
    }
}

at::Tensor Placeholder(const at::Tensor &query, at::ScalarType dtype)
{
    return at::empty({1}, query.options().dtype(dtype));
}

c10::optional<at::Tensor> ContiguousOptional(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        return c10::nullopt;
    }
    return tensor->contiguous();
}

void DispatchSparseFlashAttention(uint32_t block_dim, const at::Tensor &query, const at::Tensor &key,
                                  const at::Tensor &value, const at::Tensor &sparse_indices,
                                  const at::Tensor &block_table, const at::Tensor &actual_seq_lengths_query,
                                  const at::Tensor &actual_seq_lengths_kv, const at::Tensor &query_rope,
                                  const at::Tensor &key_rope, const at::Tensor &attention_out,
                                  const at::Tensor &softmax_max, const at::Tensor &softmax_sum,
                                  const at::Tensor &workspace, const at::Tensor &tiling)
{
    EXEC_KERNEL_CMD(sgl_sparse_flash_attention, block_dim, query, key, value, sparse_indices, block_table,
                    actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, key_rope, attention_out, softmax_max,
                    softmax_sum, workspace, tiling);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> make_outputs(const at::Tensor &query, const at::Tensor &key,
                                                            c10::string_view layout_query, c10::string_view layout_kv,
                                                            bool return_softmax_lse)
{
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "sparse_flash_attention: query must be float16 or bfloat16");
    TORCH_CHECK(query.device().type() == DEVICE_TYPE, "sparse_flash_attention: query must be an NPU tensor");

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
    const auto query_input = query.contiguous();
    const auto key_input = key.contiguous();
    const auto value_input = value.contiguous();
    const auto sparse_indices_input = sparse_indices.contiguous();
    const auto block_table_input = ContiguousOptional(block_table);
    const auto actual_seq_lengths_query_input = ContiguousOptional(actual_seq_lengths_query);
    const auto actual_seq_lengths_kv_input = ContiguousOptional(actual_seq_lengths_kv);
    const auto query_rope_input = ContiguousOptional(query_rope);
    const auto key_rope_input = ContiguousOptional(key_rope);

    TORCH_CHECK(query_input.numel() > 0, "query must not be empty");
    TORCH_CHECK(key_input.numel() > 0, "key must not be empty");
    TORCH_CHECK(value_input.numel() > 0, "value must not be empty");
    TORCH_CHECK(sparse_indices_input.numel() > 0, "sparse_indices must not be empty");
    TORCH_CHECK(sparse_indices_input.scalar_type() == at::kInt, "sparse_indices must be int32");
    TORCH_CHECK(
        key_input.scalar_type() == query_input.scalar_type() && value_input.scalar_type() == query_input.scalar_type(),
        "key and value dtypes must match query");
    auto checkInt32 = [](const c10::optional<at::Tensor> &tensor, const char *name) {
        TORCH_CHECK(!tensor.has_value() || tensor->scalar_type() == at::kInt, name, " must be int32");
    };
    checkInt32(block_table_input, "block_table");
    checkInt32(actual_seq_lengths_query_input, "actual_seq_lengths_query");
    checkInt32(actual_seq_lengths_kv_input, "actual_seq_lengths_kv");
    TORCH_CHECK(!query_rope_input.has_value() || query_rope_input->scalar_type() == query_input.scalar_type(),
                "query_rope dtype must match query");
    TORCH_CHECK(!key_rope_input.has_value() || key_rope_input->scalar_type() == key_input.scalar_type(),
                "key_rope dtype must match key");
    CheckTensor(query_input, query_input, "query");
    CheckTensor(key_input, query_input, "key");
    CheckTensor(value_input, query_input, "value");
    CheckTensor(sparse_indices_input, query_input, "sparse_indices");
    CheckOptionalTensor(block_table_input, query_input, "block_table");
    CheckOptionalTensor(actual_seq_lengths_query_input, query_input, "actual_seq_lengths_query");
    CheckOptionalTensor(actual_seq_lengths_kv_input, query_input, "actual_seq_lengths_kv");
    CheckOptionalTensor(query_rope_input, query_input, "query_rope");
    CheckOptionalTensor(key_rope_input, query_input, "key_rope");

    const std::string query_layout(layout_query);
    const std::string kv_layout(layout_kv);
    auto [attention_out, softmax_max, softmax_sum] =
        make_outputs(query_input, key_input, query_layout, kv_layout, return_softmax_lse);

    SFAHost::SparseFlashAttention op("sparse_flash_attention");
    op.SetAttrAny("scale_value", static_cast<float>(scale_value));
    op.SetAttrAny("sparse_block_size", sparse_block_size);
    op.SetAttrStr("layout_query", query_layout);
    op.SetAttrStr("layout_kv", kv_layout);
    op.SetAttrAny("sparse_mode", sparse_mode);
    op.SetAttrAny("pre_tokens", pre_tokens);
    op.SetAttrAny("next_tokens", next_tokens);
    op.SetAttrAny("attention_mode", attention_mode);
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>("sparse_flash_attention");
    auto scalarType = query_input.scalar_type();
    op.SetToContext(context, scalarType);
    context->RegisterTensor(query_input, true);
    context->RegisterTensor(key_input, true);
    context->RegisterTensor(value_input, true);
    context->RegisterTensor(sparse_indices_input, true);
    context->RegisterTensor(block_table_input, true);
    context->RegisterTensor(actual_seq_lengths_query_input, true);
    context->RegisterTensor(actual_seq_lengths_kv_input, true);
    context->RegisterTensor(query_rope_input, true);
    context->RegisterTensor(key_rope_input, true);
    context->RegisterTensor(attention_out, false);
    context->RegisterTensor(softmax_max, false);
    context->RegisterTensor(softmax_sum, false);

    optiling::SFATilingInfo info;
    optiling::SFAInfoParser parser(context.get());
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS, "sparse_flash_attention: parsing tiling inputs failed");
    optiling::SFATilingCheck checker(info);
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS, "sparse_flash_attention: tiling validation failed");
    optiling::SFAMlaTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, "sparse_flash_attention: tiling failed");

    constexpr size_t kTilingDataSize = 144;
    std::array<uint8_t, kTilingDataSize> serializedTilingData{};
    auto &tilingData = tiling.GetTilingData();
    TORCH_CHECK(tilingData.GetDataSize() == serializedTilingData.size(),
                "sparse_flash_attention: unexpected tiling data size");
    tilingData.SaveToBuffer(serializedTilingData.data(), serializedTilingData.size());
    auto tilingTensor = context->GetTilingTensor(serializedTilingData);
    auto workspace =
        at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, query_input.options().dtype(at::kByte));
    auto intPlaceholder = Placeholder(query_input, at::kInt);
    auto queryPlaceholder = Placeholder(query_input, query_input.scalar_type());
    auto blockTableLaunch = block_table_input.value_or(intPlaceholder);
    auto actualSeqLengthsQueryLaunch = actual_seq_lengths_query_input.value_or(intPlaceholder);
    auto actualSeqLengthsKvLaunch = actual_seq_lengths_kv_input.value_or(intPlaceholder);
    auto queryRopeLaunch = query_rope_input.value_or(queryPlaceholder);
    auto keyRopeLaunch = key_rope_input.value_or(queryPlaceholder);

    DispatchSparseFlashAttention(tiling.GetBlockDim(), query_input, key_input, value_input, sparse_indices_input,
                                 blockTableLaunch, actualSeqLengthsQueryLaunch, actualSeqLengthsKvLaunch,
                                 queryRopeLaunch, keyRopeLaunch, attention_out, softmax_max, softmax_sum, workspace,
                                 tilingTensor);
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
