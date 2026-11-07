// Framework bridge for the independent AdaBlockSparseAttention CANN operator.
// Argument order and validation follow the legacy attentions plugin; the
// computation remains in the separately copied native implementation.

#include "aclnn_runtime.h"
#include "aclnn_sgl_ada_block_sparse_attention.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

aclDataType TorchToAclDtype(const at::Tensor &tensor)
{
    switch (tensor.scalar_type()) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kChar:
            return ACL_INT8;
        case at::kByte:
            return ACL_UINT8;
        case at::kBool:
            return ACL_BOOL;
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        case at::kFloat:
            return ACL_FLOAT;
        default:
            TORCH_CHECK(false, "ada block sparse attention does not support dtype ", tensor.scalar_type());
    }
    return ACL_FLOAT;
}

std::shared_ptr<aclIntArray> MakeIntArray(const c10::OptionalIntArrayRef &values)
{
    if (!values.has_value() || values->empty()) {
        return {};
    }
    return {aclCreateIntArray(values->data(), static_cast<uint64_t>(values->size())), aclDestroyIntArray};
}

at::Tensor ada_block_sparse_attention(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                      const at::Tensor &sparse_mask, const at::Tensor &sparse_count_table,
                                      std::string input_layout, int64_t sparse_size, int64_t num_heads,
                                      int64_t num_key_value_heads, double scale_value, bool causal,
                                      int64_t inner_precise, int64_t pre_tokens, int64_t next_tokens,
                                      c10::OptionalIntArrayRef actual_seq_lengths,
                                      c10::OptionalIntArrayRef actual_seq_lengths_kv)
{
    TORCH_CHECK(input_layout != "TND", "input_layout currently does not support 'TND'.");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on an NPU");
    for (const auto &tensor : {key, value, sparse_mask, sparse_count_table}) {
        TORCH_CHECK(tensor.device() == query.device(), "all inputs must be on the same NPU");
    }
    const c10::DeviceGuard deviceGuard(query.device());

    auto q = query.contiguous(), k = key.contiguous(), v = value.contiguous();
    auto mask = sparse_mask.contiguous(), countTable = sparse_count_table.contiguous();

    // The legacy plugin allocated the output with query.options().
    auto attentionOut = at::empty(q.sizes(), query.options());
    auto seqLengths = MakeIntArray(actual_seq_lengths);
    auto seqLengthsKv = MakeIntArray(actual_seq_lengths_kv);

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglAdaBlockSparseAttentionGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglAdaBlockSparseAttentionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglAdaBlockSparseAttention)>(
        sgl_cann::op_api_symbol("aclnnSglAdaBlockSparseAttention"));
    auto qDesc = sgl_cann::tensor_descriptor(q, TorchToAclDtype(q));
    auto kDesc = sgl_cann::tensor_descriptor(k, TorchToAclDtype(k));
    auto vDesc = sgl_cann::tensor_descriptor(v, TorchToAclDtype(v));
    auto maskDesc = sgl_cann::tensor_descriptor(mask, TorchToAclDtype(mask));
    auto countDesc = sgl_cann::tensor_descriptor(countTable, TorchToAclDtype(countTable));
    auto outDesc = sgl_cann::tensor_descriptor(attentionOut, TorchToAclDtype(attentionOut));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // Unused optional ports (pse_shift, atten_mask, quantization scales) stay nullptr.
    const auto status = queryWorkspace(qDesc.get(), kDesc.get(), vDesc.get(), nullptr, nullptr, seqLengths.get(),
                                       seqLengthsKv.get(), nullptr, nullptr, nullptr, nullptr, nullptr, maskDesc.get(),
                                       countDesc.get(), num_heads, static_cast<float>(scale_value), pre_tokens,
                                       next_tokens, input_layout.data(), num_key_value_heads, static_cast<int64_t>(0),
                                       inner_precise, sparse_size, causal, outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "ada block sparse attention workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));

    std::vector<at::Tensor> tensors{q, k, v, mask, countTable, attentionOut, workspace};
    std::vector<std::shared_ptr<aclTensor>> descriptors{qDesc, kDesc, vDesc, maskDesc, countDesc, outDesc};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, seqLengths, seqLengthsKv, input_layout, workspace, workspaceSize, executor,
                   aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "ada block sparse attention ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglAdaBlockSparseAttention", launch);
    return attentionOut;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "ada_block_sparse_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_mask, "
        "Tensor sparse_count_table, str input_layout='BNSD', int sparse_size=128, int num_heads=1, "
        "int num_key_value_heads=1, float scale_value=1, bool causal=True, int inner_precise=1, "
        "int pre_tokens=214748647, int next_tokens=0, int[]? actual_seq_lengths=None, "
        "int[]? actual_seq_lengths_kv=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("ada_block_sparse_attention", &ada_block_sparse_attention);
}
