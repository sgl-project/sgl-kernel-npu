#include <cstring>
#include <tuple>
#include <unordered_map>

#include "aclrtlaunch_mlp_lightning_indexer.h"
#include "common.h"
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
constexpr uint32_t MAX_CAPTURE_NUM = 1024U;

uint32_t actualCaptureNum = 0U;
static std::unordered_map<uint64_t, uint32_t> captureMap;
static at::Tensor globalTilingBuffer;
static c10::DeviceIndex cachedDeviceIndex = -1;

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

at::Tensor GetOrCreateCachedTilingTensor(const LITilingData &tilingData, const at::Tensor &device_anchor)
{
    const int64_t tiling_size = static_cast<int64_t>(sizeof(LITilingData));
    auto hash_key = std::make_tuple(
        tilingData.bSize, tilingData.n2Size, tilingData.gSize, tilingData.s1Size, tilingData.s2Size,
        tilingData.sparseCount, tilingData.blockLen, tilingData.qBlockLen, tilingData.initNum, tilingData.localNum,
        tilingData.usedCoreNum, tilingData.blockSize, tilingData.maxBlockNumPerBatch, tilingData.sparseMode,
        tilingData.preTokens, tilingData.nextTokens, tilingData.returnValue, tilingData.tilingKey);
    uint64_t hash_value = host_utils::TupleHasher::Hash(hash_key);
    c10::DeviceIndex device_index = device_anchor.get_device();

    if (!globalTilingBuffer.defined() || cachedDeviceIndex != device_index) {
        globalTilingBuffer =
            at::empty({tiling_size * MAX_CAPTURE_NUM},
                      at::TensorOptions().dtype(at::kByte).device(device_anchor.device()));
        captureMap.clear();
        actualCaptureNum = 0U;
        cachedDeviceIndex = device_index;
    }

    auto it = captureMap.find(hash_value);
    if (it != captureMap.end()) {
        return at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + tiling_size * it->second, tiling_size, at::kByte);
    }

    auto tiling_cpu = at::zeros({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &tilingData, sizeof(LITilingData));
    auto device_tiling =
        sglang::npu_kernel::TorchNpuHelper::CopyTensorHostToDevice(tiling_cpu).view({tiling_size});

    if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        return device_tiling;
    }

    uint32_t slot = actualCaptureNum++;
    captureMap.emplace(hash_value, slot);
    globalTilingBuffer.slice(0, static_cast<int64_t>(slot) * tiling_size, static_cast<int64_t>(slot + 1) * tiling_size)
        .copy_(device_tiling);
    return at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + tiling_size * slot, tiling_size, at::kByte);
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

    auto queryScalarType = query.scalar_type();
    opDef.SetToContext(context, queryScalarType);
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
    auto tilingTensor = GetOrCreateCachedTilingTensor(tilingData, query);

    auto workspace =
        at::empty({static_cast<int64_t>(context->GetWorkspaceSize())},
                  at::TensorOptions().dtype(at::kByte).device(query.device()));

    uint32_t blockDim = tilingData.usedCoreNum;
    EXEC_KERNEL_CMD(mlp_lightning_indexer, blockDim, query, key, weights, cur_seq_lengths_query,
                    cur_seq_lengths_key, block_table, init_tensor, local_tensor, sparse_indices, sparse_values,
                    workspace, tilingTensor);
    return {sparse_indices, sparse_values};
}

}  // namespace sglang::npu_kernel
