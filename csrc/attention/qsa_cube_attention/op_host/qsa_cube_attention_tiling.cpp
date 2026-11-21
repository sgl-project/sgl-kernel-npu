#include "qsa_cube_attention_tiling.h"
#include <algorithm>
#include <cmath>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"

namespace optiling {
namespace {
constexpr size_t IDX_Q = 0;
constexpr size_t IDX_K = 1;
constexpr size_t IDX_V = 2;
constexpr size_t IDX_INDICES = 3;
constexpr size_t IDX_TABLE = 4;
constexpr size_t IDX_T2R = 5;
constexpr size_t IDX_OUT = 0;

constexpr int64_t HEAD_DIM = 256;
constexpr int64_t TOPK = 2051;
constexpr int64_t COMPACT_WIDTH = 514;  // 512 group IDs, logical position, sequence length
constexpr int64_t PAIRED_MAX_CACHE_TOKENS = 65536;
constexpr int64_t PAIRED_TP4_MAX_CACHE_TOKENS = 33792;
constexpr int64_t SLOT_TILE = 64;
constexpr int64_t NUM_TILES = (TOPK + SLOT_TILE - 1) / SLOT_TILE;  // 33

ge::graphStatus CheckInput(const gert::TilingContext *context, size_t idx, ge::DataType dtype,
                           size_t rank, const char *name)
{
    const auto *desc = context->GetInputDesc(idx);
    if (desc == nullptr || desc->GetDataType() != dtype) {
        OP_LOGE(context->GetNodeName(), "%s desc/dtype mismatch.", name);
        return ge::GRAPH_FAILED;
    }
    const auto *shape = context->GetInputShape(idx);
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != rank) {
        OP_LOGE(context->GetNodeName(), "%s must have rank %zu.", name, rank);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus Tiling4SglQsaCubeAttention(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(IDX_Q));
    if (CheckInput(context, IDX_Q, ge::DT_BF16, 3, "q") != ge::GRAPH_SUCCESS ||
        CheckInput(context, IDX_K, ge::DT_BF16, 4, "k") != ge::GRAPH_SUCCESS ||
        CheckInput(context, IDX_V, ge::DT_BF16, 4, "v") != ge::GRAPH_SUCCESS ||
        CheckInput(context, IDX_INDICES, ge::DT_INT32, 2, "logical_indices") != ge::GRAPH_SUCCESS ||
        CheckInput(context, IDX_TABLE, ge::DT_INT32, 2, "block_table") != ge::GRAPH_SUCCESS ||
        CheckInput(context, IDX_T2R, ge::DT_INT32, 1, "token_to_req") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto *outDesc = context->GetOutputDesc(IDX_OUT);
    const auto *outShape = context->GetOutputShape(IDX_OUT);
    if (outDesc == nullptr || outShape == nullptr || outDesc->GetDataType() != ge::DT_BF16 ||
        outShape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "out must be bf16 rank3.");
        return ge::GRAPH_FAILED;
    }
    const auto qShape = context->GetInputShape(IDX_Q)->GetStorageShape();
    const auto kShape = context->GetInputShape(IDX_K)->GetStorageShape();
    const auto vShape = context->GetInputShape(IDX_V)->GetStorageShape();
    const auto idxShape = context->GetInputShape(IDX_INDICES)->GetStorageShape();
    const auto tableShape = context->GetInputShape(IDX_TABLE)->GetStorageShape();
    const auto ttrShape = context->GetInputShape(IDX_T2R)->GetStorageShape();
    const auto outStorage = outShape->GetStorageShape();
    const int64_t rows = qShape.GetDim(0);
    const int64_t numQueryHeads = qShape.GetDim(1);
    const int64_t numRequests = tableShape.GetDim(0);
    const int64_t numCacheBlocks = kShape.GetDim(0);
    const int64_t pageTokens = kShape.GetDim(1);
    const int64_t pageTableWidth = tableShape.GetDim(1);
    bool shapeOk = rows > 1 && (numQueryHeads == 3 || numQueryHeads == 6 || numQueryHeads == 12) && qShape.GetDim(2) == HEAD_DIM &&
                   numCacheBlocks > 0 && (numQueryHeads == 12 ? (pageTokens == 768 || pageTokens == 1536) :
                                                                            (pageTokens == numQueryHeads * HEAD_DIM || (numQueryHeads == 3 && pageTokens == 128))) &&
                   kShape.GetDim(2) == 1 && kShape.GetDim(3) == HEAD_DIM &&
                   vShape.GetDim(0) == numCacheBlocks && vShape.GetDim(1) == pageTokens &&
                   vShape.GetDim(2) == 1 && vShape.GetDim(3) == HEAD_DIM &&
                   idxShape.GetDim(0) == rows && (idxShape.GetDim(1) == TOPK || idxShape.GetDim(1) == COMPACT_WIDTH) &&
                   tableShape.GetDim(0) > 0 && tableShape.GetDim(1) > 0 && pageTableWidth <= ((numQueryHeads == 3 && pageTokens == 128) ? 512 : 176) &&
                   ttrShape.GetDim(0) == rows &&
                   outStorage.GetDim(0) == rows && outStorage.GetDim(1) == numQueryHeads &&
                   outStorage.GetDim(2) == HEAD_DIM;
    if (!shapeOk) {
        OP_LOGE(context->GetNodeName(), "shapes do not match the long-row QSA contract.");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t cores = platform.GetCoreNumAic();
    if (cores == 0) {
        cores = platform.GetCoreNum();
    }
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }
    // The caller narrows block_table with its CPU-side max_seq_len. Keep the
    // full-cache QK scan within the measured profitable range for each head count.
    const int64_t sharedQueryRows = 120 / numQueryHeads;
    // A shared group must occupy every AIC; undersized H3 launches lose parallelism.
    // H12 combines replicated queries, not KV pages. Python selects the measured
    // per-replica capacity; this host bound supports both physical page layouts.
    const int64_t sharedPageLimit = numQueryHeads == 12 ?
        (pageTokens == 1536 ? PAIRED_TP4_MAX_CACHE_TOKENS : PAIRED_MAX_CACHE_TOKENS) / pageTokens :
                                                        (numQueryHeads == 3 ? 19 * 768 / pageTokens : 11);
    const bool sharedQk = rows >= static_cast<int64_t>(cores) * sharedQueryRows &&
                          pageTableWidth <= sharedPageLimit;
    // Sparse paired queries retain N64 attention beyond the dense scan range.
    const int64_t queryRowsPerGroup = sharedQk ? sharedQueryRows : (numQueryHeads == 12 ? 2 : 12 / numQueryHeads);
    const uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(
        (rows + queryRowsPerGroup - 1) / queryRowsPerGroup, static_cast<int64_t>(cores)));

    SglQsaCubeAttentionTilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(rows));
    tiling.set_numQueryHeads(static_cast<uint32_t>(numQueryHeads));
    tiling.set_numRequests(static_cast<uint32_t>(numRequests));
    tiling.set_numCacheBlocks(static_cast<uint32_t>(numCacheBlocks));
    tiling.set_pageTokens(static_cast<uint32_t>(pageTokens));
    tiling.set_pageTableWidth(static_cast<uint32_t>(pageTableWidth));
    // Logical attention width stays 2051; topk carries the input row stride.
    tiling.set_topk(static_cast<uint32_t>(idxShape.GetDim(1)));
    tiling.set_numTiles(static_cast<uint32_t>(NUM_TILES));
    tiling.set_usedCoreNum(blockDim);
    context->SetBlockDim(blockDim);
    // Compile each head/path pair separately instead of sharing one entry.
    context->SetTilingKey(2 * numQueryHeads + (sharedQk ? 1 : 0));
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = platform.GetLibApiWorkSpaceSize() + static_cast<size_t>(blockDim) * (1153024 + 256);
    if (sharedQk) {
        workspace[0] += static_cast<size_t>(blockDim) *
            (256 + static_cast<size_t>(128) * pageTableWidth * pageTokens * sizeof(float));
    }
    auto raw = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4SglQsaCubeAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SglQsaCubeAttention)
    .Tiling(Tiling4SglQsaCubeAttention)
    .TilingParse<SglQsaCubeAttentionCompileInfo>(TilingPrepare4SglQsaCubeAttention);
}  // namespace optiling
