#include "aclnn_runtime.h"
#include "aclnn_sgl_mlp_lightning_indexer.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <string>
#include <tuple>
#include <vector>

namespace {

aclDataType acl_dtype(at::ScalarType dtype)
{
    // Describe the actual storage dtype; CANN's OpDef validates the input contract.
    switch (dtype) {
        case at::kByte:
            return ACL_UINT8;
        case at::kChar:
            return ACL_INT8;
        case at::kShort:
            return ACL_INT16;
        case at::kInt:
            return ACL_INT32;
        case at::kLong:
            return ACL_INT64;
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kDouble:
            return ACL_DOUBLE;
        case at::kBool:
            return ACL_BOOL;
        case at::kBFloat16:
            return ACL_BF16;
        default:
            return ACL_DT_UNDEFINED;
    }
}

std::tuple<at::Tensor, at::Tensor> mlp_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query, const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &init_tensor,
    const c10::optional<at::Tensor> &local_tensor, c10::string_view layout_query, c10::string_view layout_key,
    int64_t sparse_count, int64_t kv_block_len, int64_t q_block_len, int64_t init_num, int64_t local_num,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    const c10::DeviceGuard deviceGuard(query.device());
    // These checks are present in the official op-plugin adapter.
    TORCH_CHECK(query.is_contiguous(), "query should be contiguous.");
    TORCH_CHECK(key.is_contiguous(), "key should be contiguous.");
    TORCH_CHECK(weights.is_contiguous(), "weights should be contiguous.");

    auto queryLayout = std::make_shared<std::string>(layout_query);
    auto keyLayout = std::make_shared<std::string>(layout_key);
    at::SmallVector<int64_t, 4> outputSize;
    if (*queryLayout == "BSND") {
        outputSize = {query.size(0), query.size(1), key.size(2), sparse_count};
    } else {
        int64_t nDimIndex = *keyLayout == "PA_BSND" ? 2 : 1;
        outputSize = {query.size(0), key.size(nDimIndex), sparse_count};
    }
    auto sparseIndices = at::empty(outputSize, query.options().dtype(at::kInt));
    // The second Tensor keeps its full shape even when return_value is false.
    auto sparseValues = at::empty(outputSize, query.options());

    // The paired SGLang caller supplies int64 lengths. Do not add a conversion.
    std::vector<at::Tensor> tensors = {
        query,
        key,
        weights,
        cur_seq_lengths_query.value_or(at::Tensor()),
        cur_seq_lengths_key.value_or(at::Tensor()),
        block_table.value_or(at::Tensor()),
        init_tensor.value_or(at::Tensor()),
        local_tensor.value_or(at::Tensor()),
        sparseIndices,
        sparseValues,
    };
    std::vector<std::shared_ptr<aclTensor>> descriptors;
    descriptors.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        descriptors.push_back(tensor.defined() ? sgl_cann::tensor_descriptor(tensor, acl_dtype(tensor.scalar_type()))
                                               : nullptr);
    }

    static auto getWorkspaceSize = reinterpret_cast<decltype(&aclnnSglMlpLightningIndexerGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglMlpLightningIndexerGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglMlpLightningIndexer)>(
        sgl_cann::op_api_symbol("aclnnSglMlpLightningIndexer"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status =
        getWorkspaceSize(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                         descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), descriptors[7].get(),
                         const_cast<char *>(queryLayout->c_str()), const_cast<char *>(keyLayout->c_str()), sparse_count,
                         kv_block_len, q_block_len, init_num, local_num, sparse_mode, pre_tokens, next_tokens,
                         return_value, descriptors[8].get(), descriptors[9].get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "MlpLightningIndexer workspace query failed: ", aclGetRecentErrMsg());

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors) {
        if (tensor.defined() && tensor.device().type() == c10::DeviceType::PrivateUse1) {
            tensor.record_stream(stream);
        }
    }
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // Retain descriptors, attributes and storage through the asynchronous host queue.
    auto launch = [tensors, descriptors, queryLayout, keyLayout, workspace, workspaceSize, executor,
                   aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "MlpLightningIndexer ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglMlpLightningIndexer", launch);
    return {sparseIndices, sparseValues};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "mlp_lightning_indexer(Tensor query, Tensor key, Tensor weights, "
        "Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, "
        "Tensor? block_table=None, Tensor? init_tensor=None, Tensor? local_tensor=None, "
        "str layout_query='BSND', str layout_key='PA_BSND', int sparse_count=2048, int kv_block_len=1, "
        "int q_block_len=1, int init_num=0, int local_num=0, int sparse_mode=3, "
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, bool return_value=False) "
        "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mlp_lightning_indexer", &mlp_lightning_indexer);
}
