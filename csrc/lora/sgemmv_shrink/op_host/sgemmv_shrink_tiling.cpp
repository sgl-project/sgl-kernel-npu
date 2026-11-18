// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
//
// Tiling for sgemmv_shrink: the legacy Host computed the per-core token count
// and block count from the AIV core number, then launched the Kernel directly.
// The same formulas now produce a tiling blob consumed by the framework.

#include "../op_kernel/sgemmv_shrink_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <limits>

namespace optiling {

static ge::graphStatus TilingSglSgemmvShrink(gert::TilingContext *context)
{
    // Input order: x, weight, lora_indices, seq_len, lora_ranks, lora_scales, y.
    const auto *xShapePtr = context->GetInputShape(0);
    const auto *weightShapePtr = context->GetInputShape(1);
    const auto *loraIndicesShapePtr = context->GetInputShape(2);
    const auto *seqLenShapePtr = context->GetInputShape(3);
    const auto *loraRanksShapePtr = context->GetInputShape(4);
    const auto *loraScalesShapePtr = context->GetInputShape(5);
    const auto *yShapePtr = context->GetInputShape(6);
    const auto *yDesc = context->GetInputDesc(6);
    if (!xShapePtr || !weightShapePtr || !loraIndicesShapePtr || !seqLenShapePtr || !loraRanksShapePtr ||
        !loraScalesShapePtr || !yShapePtr || !yDesc || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto &xShape = xShapePtr->GetStorageShape();
    const auto &weightShape = weightShapePtr->GetStorageShape();
    const auto &loraIndicesShape = loraIndicesShapePtr->GetStorageShape();
    const auto &seqLenShape = seqLenShapePtr->GetStorageShape();
    const auto &loraRanksShape = loraRanksShapePtr->GetStorageShape();
    const auto &loraScalesShape = loraScalesShapePtr->GetStorageShape();
    const auto &yShape = yShapePtr->GetStorageShape();

    // Mirror the legacy HOST_API checks that are observable from shapes.
    if (xShape.GetDimNum() != 2 || (weightShape.GetDimNum() != 3 && weightShape.GetDimNum() != 4) ||
        yShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    const auto *sliceCountPtr = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const int64_t sliceCount = sliceCountPtr ? *sliceCountPtr : 1;

    const int64_t loraIndicesSize = loraIndicesShape.GetDim(0);
    const int64_t seqLenSize = seqLenShape.GetDim(0);
    const int64_t loraRanksSize = loraRanksShape.GetDim(0);
    const int64_t loraScalesSize = loraScalesShape.GetDim(0);
    const int64_t batchSize = xShape.GetDim(0);
    const int64_t inputHiddenDim = xShape.GetDim(1);
    const int64_t outputDim = yShape.GetDim(1);
    // The legacy Host required hidden_in > hidden_out (the shrink direction).
    if (sliceCount < 1 || loraIndicesSize < 0 || seqLenSize < 0 || loraRanksSize < 0 || loraScalesSize < 0 ||
        batchSize <= 0 || inputHiddenDim <= outputDim || outputDim < sliceCount) {
        return ge::GRAPH_FAILED;
    }
    const int64_t maxLoRARank = outputDim / sliceCount;

    // The Kernel addresses every dimension with uint32, so reject values that
    // would be truncated instead of silently wrapping.
    constexpr uint64_t maxU32 = std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(loraIndicesSize) > maxU32 || static_cast<uint64_t>(seqLenSize) > maxU32 ||
        static_cast<uint64_t>(loraRanksSize) > maxU32 || static_cast<uint64_t>(loraScalesSize) > maxU32 ||
        static_cast<uint64_t>(batchSize) > maxU32 || static_cast<uint64_t>(inputHiddenDim) > maxU32 ||
        static_cast<uint64_t>(maxLoRARank) > maxU32 || static_cast<uint64_t>(sliceCount) > maxU32) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t aivNum = platform.GetCoreNumAiv();
    if (aivNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t numTokensPerCore =
        (static_cast<uint64_t>(batchSize) + static_cast<uint64_t>(aivNum) - 1) / static_cast<uint64_t>(aivNum);
    // The legacy handler rejected batch == 0 through this same check.
    if (numTokensPerCore == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalBlocks = static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(sliceCount);
    const uint64_t blockDim = (totalBlocks + numTokensPerCore - 1) / numTokensPerCore;
    if (blockDim == 0 || blockDim > maxU32) {
        return ge::GRAPH_FAILED;
    }

    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(SglSgemmvShrinkTilingData) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<SglSgemmvShrinkTilingData>();
    if (!tiling) {
        return ge::GRAPH_FAILED;
    }
    tiling->loraIndicesSize = static_cast<uint32_t>(loraIndicesSize);
    tiling->seqLenSize = static_cast<uint32_t>(seqLenSize);
    tiling->loraRanksSize = static_cast<uint32_t>(loraRanksSize);
    tiling->loraScalesSize = static_cast<uint32_t>(loraScalesSize);
    tiling->batchSize = static_cast<uint32_t>(batchSize);
    tiling->numTokensPerCore = static_cast<uint32_t>(numTokensPerCore);
    tiling->inputHiddenDim = static_cast<uint32_t>(inputHiddenDim);
    tiling->maxLoRARank = static_cast<uint32_t>(maxLoRARank);
    tiling->slicesCount = static_cast<uint32_t>(sliceCount);
    raw->SetDataSize(sizeof(SglSgemmvShrinkTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    // The legacy Kernel needed no scratch memory beyond the system workspace.
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// Keep the same empty prepare hook as the reference migrated operator.
struct SglSgemmvShrinkCompileInfo {};
static ge::graphStatus TilingPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SglSgemmvShrink).Tiling(TilingSglSgemmvShrink).TilingParse<SglSgemmvShrinkCompileInfo>(TilingPrepare);

}  // namespace optiling
