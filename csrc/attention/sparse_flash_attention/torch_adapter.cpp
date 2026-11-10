// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// Framework bridge for the migrated sparse flash attention operator. Validation
// and output shapes mirror the legacy Host wrapper; all computation stays in the
// copied Device kernel.

#include "aclnn_runtime.h"
#include "aclnn_sgl_sparse_flash_attention.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr int64_t kTndDimensions = 3;
constexpr int64_t kBsndDimensions = 4;

aclDataType ToAclType(const at::Tensor &tensor)
{
    switch (tensor.scalar_type()) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        default:
            TORCH_CHECK(false, "sparse_flash_attention does not support dtype ", tensor.scalar_type());
            return ACL_FLOAT;
    }
}

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

c10::optional<at::Tensor> ContiguousOptional(const c10::optional<at::Tensor> &tensor)
{
    if (!tensor.has_value()) {
        return c10::nullopt;
    }
    return tensor->contiguous();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> MakeOutputs(const at::Tensor &query, const at::Tensor &key,
                                                           c10::string_view layout_query, c10::string_view layout_kv,
                                                           bool return_softmax_lse)
{
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "sparse_flash_attention: query must be float16 or bfloat16");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "sparse_flash_attention: query must be an NPU tensor");

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
        // Keep a valid data pointer for ACLNN while exposing the legacy empty output.
        softmax_max = at::empty({1}, query.options().dtype(at::kFloat)).narrow(0, 0, 0);
        softmax_sum = at::empty({1}, query.options().dtype(at::kFloat)).narrow(0, 0, 0);
    }
    return {attention_out, softmax_max, softmax_sum};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sgl_sparse_flash_attention(
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
    const c10::DeviceGuard deviceGuard(query_input.device());
    auto [attention_out, softmax_max, softmax_sum] =
        MakeOutputs(query_input, key_input, query_layout, kv_layout, return_softmax_lse);

    auto descriptor = [](const at::Tensor &tensor) { return sgl_cann::tensor_descriptor(tensor, ToAclType(tensor)); };
    auto optionalDescriptor = [&descriptor](const c10::optional<at::Tensor> &tensor) -> std::shared_ptr<aclTensor> {
        if (!tensor.has_value()) {
            return {};
        }
        return descriptor(*tensor);
    };

    auto queryDesc = descriptor(query_input);
    auto keyDesc = descriptor(key_input);
    auto valueDesc = descriptor(value_input);
    auto sparseIndicesDesc = descriptor(sparse_indices_input);
    auto blockTableDesc = optionalDescriptor(block_table_input);
    auto actualSeqLengthsQueryDesc = optionalDescriptor(actual_seq_lengths_query_input);
    auto actualSeqLengthsKvDesc = optionalDescriptor(actual_seq_lengths_kv_input);
    auto queryRopeDesc = optionalDescriptor(query_rope_input);
    auto keyRopeDesc = optionalDescriptor(key_rope_input);
    auto attentionOutDesc = descriptor(attention_out);
    auto softmaxMaxDesc = descriptor(softmax_max);
    auto softmaxSumDesc = descriptor(softmax_sum);

    static auto workspaceQuery = reinterpret_cast<decltype(&aclnnSglSparseFlashAttentionGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseFlashAttentionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglSparseFlashAttention)>(
        sgl_cann::op_api_symbol("aclnnSglSparseFlashAttention"));
    // The generated ACLNN interface exposes string attributes as mutable
    // char*, so keep writable copies of the exact layout names.
    std::string layoutQueryString(query_layout);
    std::string layoutKvString(kv_layout);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status =
        workspaceQuery(queryDesc.get(), keyDesc.get(), valueDesc.get(), sparseIndicesDesc.get(), blockTableDesc.get(),
                       actualSeqLengthsQueryDesc.get(), actualSeqLengthsKvDesc.get(), queryRopeDesc.get(),
                       keyRopeDesc.get(), static_cast<float>(scale_value), sparse_block_size, layoutQueryString.data(),
                       layoutKvString.data(), sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse,
                       attentionOutDesc.get(), softmaxMaxDesc.get(), softmaxSumDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "sparse_flash_attention workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query_input.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{query_input,   key_input,   value_input, sparse_indices_input,
                                    attention_out, softmax_max, softmax_sum, workspace};
    for (const auto *tensor : {&block_table_input, &actual_seq_lengths_query_input, &actual_seq_lengths_kv_input,
                               &query_rope_input, &key_rope_input}) {
        if (tensor->has_value()) {
            tensors.push_back(**tensor);
        }
    }
    for (const auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    auto aclStream = stream.stream(false);

    std::vector<std::shared_ptr<aclTensor>> descriptors{queryDesc,
                                                        keyDesc,
                                                        valueDesc,
                                                        sparseIndicesDesc,
                                                        blockTableDesc,
                                                        actualSeqLengthsQueryDesc,
                                                        actualSeqLengthsKvDesc,
                                                        queryRopeDesc,
                                                        keyRopeDesc,
                                                        attentionOutDesc,
                                                        softmaxMaxDesc,
                                                        softmaxSumDesc};
    auto launch = [tensors, descriptors, layoutQueryString, layoutKvString, workspace, workspaceSize, executor,
                   aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "sparse_flash_attention ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSparseFlashAttention", launch);
    return {attention_out, softmax_max, softmax_sum};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, "
        "float scale_value, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, "
        "int sparse_block_size=1, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
        "int attention_mode=2, bool return_softmax_lse=False) "
        "-> (Tensor attention_out, Tensor softmax_max, Tensor softmax_sum)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_flash_attention", &sgl_sparse_flash_attention);
}
