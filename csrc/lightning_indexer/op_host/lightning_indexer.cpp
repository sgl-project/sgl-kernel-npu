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
#include "aclrtlaunch_lightning_indexer.h"

namespace sglang::LIHost {

using namespace ge_helper;
constexpr uint32_t MAX_CAPTURE_NUM = 1024;
// npu tensor max size
constexpr int SIZE = 8;
constexpr int DIM_0 = 0;
constexpr int DIM_1 = 1;
constexpr int DIM_2 = 2;
constexpr int DIM_3 = 3;

// namespace scope global parameters
uint32_t actualCaptureNum = 0;
std::unordered_map<uint64_t, uint32_t> captureMap;
at::Tensor workspace;

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
                                      const at::Tensor &actual_seq_lengths_query,
                                      const at::Tensor &actual_seq_lengths_key, const at::Tensor &block_table,
                                      c10::optional<c10::string_view> layout_query, c10::optional<c10::string_view> layout_key,
                                      c10::optional<int64_t> sparse_count, c10::optional<int64_t> sparse_mode)
{
    using namespace LIHost;
    std::cout << "0" << std::endl;
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

    at::Tensor sparse_indices = ConstructLightningIndexerOutputTensor(query, key, actual_seq_lengths_query,
                                                                      sparseCount, layoutQuery, layoutKey);

    auto qScalarType = query.scalar_type();
    std::cout << "1" << std::endl;
    indexer.SetToContext(context, qScalarType);
    std::cout << "2" << std::endl;
    context->RegisterTensor(query, true);
    context->RegisterTensor(key, true);
    context->RegisterTensor(weights, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_key, true);
    context->RegisterTensor(block_table, true);
    context->RegisterTensor(sparse_indices, false);
    std::cout << "3" << std::endl;

    LITilingInfo liInfo;
    LIInfoParser LIInfoParser(context.get());
    std::cout << "4" << std::endl;
    TORCH_CHECK(LIInfoParser.ParseAndCheck(liInfo) == ge::GRAPH_SUCCESS, "lightning_indexer ParseAndCheck failed")

    LightningIndexerTiling liTiling(context.get());
    liTiling.DoTiling(&liInfo);
    std::cout << "5" << std::endl;
    const auto &tilingData = liTiling.GetTilingData();

    uint32_t tilingSize = sizeof(LITilingData);
    auto blockDim = tilingData.usedCoreNum;
    auto bs = query.sizes()[0];
    uint64_t mapKey = tilingData.tilingKey;
    mapKey = (mapKey << 32) | bs;
    std::cout << "mapKey is " << mapKey << std::endl;

    static auto globalTilingData = at::empty({tilingSize * MAX_CAPTURE_NUM},
                                             at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    if (captureMap.find(mapKey) == captureMap.end()) {
        std::cout << "step in" << std::endl;
        TORCH_CHECK(actualCaptureNum < MAX_CAPTURE_NUM, "lightning_indexer captureNum overflow")
        captureMap[mapKey] = actualCaptureNum;
        aclrtMemcpy(globalTilingData.data_ptr<uint8_t>() + actualCaptureNum * tilingSize, tilingSize, &tilingData,
                    tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
        actualCaptureNum++;
    }
    at::Tensor tilingTensor =
        at::from_blob(globalTilingData.data_ptr<uint8_t>() + (tilingSize * captureMap[mapKey]), tilingSize, at::kByte);

    size_t userWorkspaceSize = *context->GetWorkspaceSizes(1);
    workspace = at::empty({userWorkspaceSize}, at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    std::cout << "6" << std::endl;
    EXEC_KERNEL_CMD(lightning_indexer, blockDim, query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key,
                    block_table, sparse_indices, workspace, tilingTensor);
    return sparse_indices;
}
}  // namespace npu_kernel
}  // namespace sglang
