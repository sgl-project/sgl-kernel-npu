#include <cstdio>
#include <string>
#include "acl/acl.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/lightning_indexer_tiling.h"
#include "defines.h"
#include "torch_helper.h"
#include "ge_helper.h"
#include "common_tiling.h"
#include "lightning_indexer_def.h"
#include "common.h"
#include "aclrtlaunch_lightning_indexer.h"

namespace sglang::LIHost {

using namespace ge_helper;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
constexpr uint32_t MAX_DECODE_BS = 512;
// npu tensor max size
constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;

// namespace scope global parameters
uint32_t actualCaptureNum = 0;
static std::unordered_map<uint64_t, uint32_t> captureMap;
// at::Tensor workspace;

inline at::Tensor ConstructLightningIndexerOutputTensor(const at::Tensor &query, const at::Tensor &key,
                                                        const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                                        int64_t sparse_count, std::string query_layout_str,
                                                        std::string key_layout_str)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);

    if (query_layout_str == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor output = at::empty(outputSize, query.options().dtype(at::kInt));

    return output;
}
}  // namespace sglang::LIHost

namespace sglang {
namespace npu_kernel {
HOST_API at::Tensor lightning_indexer(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                                      const c10::optional<at::Tensor> &actual_seq_lengths_query,
                                      const c10::optional<at::Tensor> &actual_seq_lengths_key,
                                      const c10::optional<at::Tensor> &block_table,
                                      c10::optional<c10::string_view> layout_query,
                                      c10::optional<c10::string_view> layout_key, c10::optional<int64_t> sparse_count,
                                      c10::optional<int64_t> sparse_mode)
{
    using namespace LIHost;
    LightningIndexer indexer("lightning_indexer");
    auto context = std::make_shared<TilingContext>("lightning_indexer");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    std::string layoutQuery(indexer.GetAttr(ATTR_QUERY_LAYOUT_INDEX).GetString());
    std::string layoutKey(indexer.GetAttr(ATTR_KEY_LAYOUT_INDEX).GetString());
    int64_t sparseCount = std::any_cast<int32_t>(indexer.GetAttr(ATTR_SPARSE_COUNT_INDEX).GetValue());

    if (layout_query.has_value()) {
        layoutQuery = std::string(layout_query.value());
        indexer.SetAttrStr("layout_query", layoutQuery);
    }
    if (layout_key.has_value()) {
        layoutKey = std::string(layout_key.value());
        indexer.SetAttrStr("layout_key", layoutKey);
    }
    if (sparse_count.has_value()) {
        sparseCount = sparse_count.value();
        indexer.SetAttrAny("sparse_count", static_cast<int32_t>(sparseCount));
    }
    if (sparse_mode.has_value()) {
        indexer.SetAttrAny("sparse_mode", static_cast<int32_t>(sparse_mode.value()));
    }

    at::Tensor sparse_indices = ConstructLightningIndexerOutputTensor(query, key, actual_seq_lengths_query, sparseCount,
                                                                      layoutQuery, layoutKey);

    auto qScalarType = query.scalar_type();

    at::Tensor actualSeqLengthsQuery =
        actual_seq_lengths_query.has_value()
            ? actual_seq_lengths_query.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    at::Tensor actualSeqLengthsKey =
        actual_seq_lengths_key.has_value()
            ? actual_seq_lengths_key.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    at::Tensor blockTable =
        block_table.has_value()
            ? block_table.value()
            : at::empty({1}, at::TensorOptions().dtype(qScalarType).device(query.options().device()));

    indexer.SetToContext(context, qScalarType);
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_key, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(sparse_indices, false);

    LITilingInfo liInfo;
    LIInfoParser LIInfoParser(context.get());
    TORCH_CHECK(LIInfoParser.ParseAndCheck(liInfo) == ge::GRAPH_SUCCESS, "lightning_indexer ParseAndCheck failed")

    LightningIndexerTiling liTiling(context.get());
    liTiling.DoTiling(&liInfo);
    const auto &tilingData = liTiling.GetTilingData();

    uint32_t tilingSize = sizeof(LITilingData);
    auto blockDim = tilingData.usedCoreNum;
    auto bs = tilingData.bSize;
    at::Tensor tilingTensor;

    auto tup =
        std::make_tuple(tilingData.bSize, tilingData.n2Size, tilingData.gSize, tilingData.s1Size, tilingData.s2Size,
                        tilingData.blockSize, tilingData.maxBlockNumPerBatch, tilingData.tilingKey);
    auto hashValue = host_utils::TupleHasher::Hash(tup);

    static auto globalTilingBuffer = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                               at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    if (captureMap.find(hashValue) != captureMap.end()) {
        // For decode replay phase and part of prefill phase, get cached tiling data from globalTilingBuffer
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    } else if (actualCaptureNum >= MAX_CAPTURE_NUM) {
        // For tiling hash that not exist in capture map and exceeds MAX_CAPTURE_NUM, reload its' tiling data to NPU
        static auto tilingBuffer =
            at::empty({tilingSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
        aclrtMemcpy(tilingBuffer.data_ptr<uint8_t>(), tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        tilingTensor = at::from_blob(tilingBuffer.data_ptr<uint8_t>(), tilingSize, at::kByte);
    } else {
        // Captured tiling cached here
        captureMap[hashValue] = actualCaptureNum;
        aclrtMemcpy(globalTilingBuffer.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, tilingSize, &tilingData,
                    tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
        tilingTensor = at::from_blob(globalTilingBuffer.data_ptr<uint8_t>() + (tilingSize * captureMap[hashValue]),
                                     tilingSize, at::kByte);
    }

    size_t workspaceSize = context->GetWorkspaceSize();
    auto workspace = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    EXEC_KERNEL_CMD(lightning_indexer, blockDim, query, key, weights, actualSeqLengthsQuery, actualSeqLengthsKey,
                    blockTable, sparse_indices, workspace, tilingTensor);
    return sparse_indices;
}
}  // namespace npu_kernel
}  // namespace sglang
