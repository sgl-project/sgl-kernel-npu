#include <tuple>

#include "aclrtlaunch_mlp_lightning_indexer.h"
#include "defines.h"
#include "ge_helper.h"
#include "mlp_lightning_indexer_def.h"
#include "mlp_lightning_indexer_tiling.h"
#include "torch_helper.h"

namespace sglang::MlpLIHost {
namespace {

constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;

std::tuple<at::Tensor, at::Tensor> ConstructOutputs(const at::Tensor &query, const at::Tensor &key,
                                                    int64_t sparse_count, const std::string &layout_query,
                                                    const std::string &layout_key)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    if (layout_query == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int64_t nDimIndex = layout_key == "PA_BSND" ? DIM_2 : DIM_1;
        outputSize = {query.size(DIM_0), key.size(nDimIndex), sparse_count};
    }
    auto sparseIndices = at::empty(outputSize, query.options().dtype(at::kInt));
    auto sparseValues = at::empty(outputSize, query.options());
    return std::make_tuple(sparseIndices, sparseValues);
}

at::Tensor BuildTilingTensor(const LITilingData &tilingData)
{
    auto tilingCpu = at::empty({static_cast<int64_t>(sizeof(LITilingData))},
                               at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tilingCpu.data_ptr<uint8_t>(), &tilingData, sizeof(LITilingData));
    return TorchNpuHelper::CopyTensorHostToDevice(tilingCpu).view({static_cast<int64_t>(sizeof(LITilingData))});
}

}  // namespace
}  // namespace sglang::MlpLIHost

namespace sglang::npu_kernel {

HOST_API std::tuple<at::Tensor, at::Tensor> mlp_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &cur_seq_lengths_query,
    const c10::optional<at::Tensor> &cur_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &init_tensor,
    const c10::optional<at::Tensor> &local_tensor,
    c10::string_view layout_query, c10::string_view layout_key,
    int64_t sparse_count, int64_t kv_block_len, int64_t q_block_len,
    int64_t init_num, int64_t local_num, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    using namespace MlpLIHost;
    MlpLightningIndexer opDef("mlp_lightning_indexer");
    auto context = std::make_shared<ge_helper::TilingContext>("mlp_lightning_indexer");

    opDef.SetAttrStr("layout_query", std::string(layout_query));
    opDef.SetAttrStr("layout_key", std::string(layout_key));
    opDef.SetAttrAny("sparse_count", static_cast<int32_t>(sparse_count));
    opDef.SetAttrAny("kv_block_len", static_cast<int32_t>(kv_block_len));
    opDef.SetAttrAny("q_block_len", static_cast<int32_t>(q_block_len));
    opDef.SetAttrAny("init_num", static_cast<int32_t>(init_num));
    opDef.SetAttrAny("local_num", static_cast<int32_t>(local_num));
    opDef.SetAttrAny("sparse_mode", static_cast<int32_t>(sparse_mode));
    opDef.SetAttrAny("pre_tokens", pre_tokens);
    opDef.SetAttrAny("next_tokens", next_tokens);
    opDef.SetAttrAny("return_value", return_value);

    auto [sparse_indices, sparse_values] =
        ConstructOutputs(query, key, sparse_count, std::string(layout_query), std::string(layout_key));

    opDef.SetToContext(context, query.scalar_type());
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(cur_seq_lengths_query, true);
    context->RegisterTensor(cur_seq_lengths_key, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(init_tensor, true);
    context->RegisterTensor(local_tensor, true);
    context->RegisterTensor(sparse_indices, false);
    context->RegisterTensor(sparse_values, false);

    LITilingInfo liInfo;
    LIInfoParser parser(context.get());
    TORCH_CHECK(parser.ParseAndCheck(liInfo) == ge::GRAPH_SUCCESS, "mlp_lightning_indexer ParseAndCheck failed");

    LightningIndexerTiling tiling(context.get());
    TORCH_CHECK(tiling.DoTiling(&liInfo) == ge::GRAPH_SUCCESS, "mlp_lightning_indexer DoTiling failed");
    const auto &tilingData = tiling.GetTilingData();
    auto tilingTensor = BuildTilingTensor(tilingData);

    auto workspace =
        at::empty({static_cast<int64_t>(context->GetWorkspaceSize())},
                  at::TensorOptions().dtype(at::kByte).device(query.device()));

    EXEC_KERNEL_CMD(mlp_lightning_indexer, tilingData.usedCoreNum, query, key, weights, cur_seq_lengths_query,
                    cur_seq_lengths_key, block_table, init_tensor, local_tensor, sparse_indices, sparse_values,
                    workspace, tilingTensor);
    return {sparse_indices, sparse_values};
}

}  // namespace sglang::npu_kernel
