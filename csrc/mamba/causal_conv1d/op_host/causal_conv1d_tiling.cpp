/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */
// Provenance: adapted from sgl-kernel-npu csrc/causal_conv1d/op_host/causal_conv1d.cpp.
// Original upstream path/import revision is unknown. Tiling formulas and layout are unchanged.

#include "../op_kernel/causal_conv1d_tiling_data.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace optiling {
namespace {
constexpr int64_t DIM_ALIGN = 16;
constexpr int64_t MAX_DIM_TILE = 4096;
constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MIN_WIDTH = 2;
constexpr int64_t ASCENDC_RESERVED_WORKSPACE = 16 * 1024 * 1024;

constexpr uint32_t CAUSAL_CONV1D_TPL_RUN_MODE_FN = 0;
constexpr uint32_t CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_2 = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_3 = 2;
constexpr uint32_t CAUSAL_CONV1D_TPL_WIDTH_4 = 3;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_INVALID = 0;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS = 1;
constexpr uint32_t CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD = 2;

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignUp(int64_t value, int64_t align)
{
    return CeilDiv(value, align) * align;
}

struct UpdateDimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

// Mirror of the GE update-mode tiling policy (ChooseCanonicalUpdateBaseDimChoice):
// pick a baseDim from a fixed candidate set that ideally divides dim and makes the
// batch*baseDimCnt grid land as close as possible to (>=) the AIV core count, so the
// batch-parallel update kernel keeps all cores busy. Falls back to ceil-division when
// no candidate divides dim exactly.
UpdateDimTileChoice ChooseUpdateBaseDimChoice(int64_t batch, int64_t dim, int32_t numCores)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};
    const int64_t coreNum = (numCores > 0) ? static_cast<int64_t>(numCores) : 1;

    auto chooseOnce = [&](bool requireExactDiv) -> UpdateDimTileChoice {
        UpdateDimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        UpdateDimTileChoice bestUnder;

        for (int64_t candBaseDim : candidates) {
            if (candBaseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % candBaseDim != 0)) {
                continue;
            }
            const int64_t baseDimCnt = requireExactDiv ? (dim / candBaseDim) : CeilDiv(dim, candBaseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }
            if (gridSize >= coreNum) {
                const int64_t gap = gridSize - coreNum;
                if (gap < bestOverGap) {
                    bestOver = {candBaseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && candBaseDim < bestUnder.baseDim)) {
                bestUnder = {candBaseDim, baseDimCnt, gridSize};
            }
        }
        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    UpdateDimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    return result;
}

void ComputeTilingData(int64_t dim, int64_t cuSeqlen, int64_t seqLen, int64_t batch, int64_t inputMode, int64_t width,
                       int64_t stateLen, int64_t numCacheLines, int64_t activationMode, int64_t padSlotId, bool hasBias,
                       bool hasCacheIndices, bool hasInitialState, bool hasNumAccept, bool isBf16, int32_t numCores,
                       int64_t runMode, CausalConv1dTilingData &td)
{
    (void)padSlotId;
    std::memset(&td, 0, sizeof(td));

    td.dim = dim;
    td.cuSeqlen = cuSeqlen;
    td.seqLen = seqLen;
    td.inputMode = inputMode;
    td.width = width;
    td.stateLen = stateLen;
    td.numCacheLines = numCacheLines;
    td.batch = batch;
    td.activationMode = activationMode;
    td.padSlotId = padSlotId;
    td.hasBias = hasBias ? 1 : 0;
    td.hasCacheIndices = hasCacheIndices ? 1 : 0;
    td.hasInitialStateMode = hasInitialState ? 1 : 0;
    td.hasInitStateWorkspace = hasInitialState ? 1 : 0;
    td.hasNumAcceptedTokens = hasNumAccept ? 1 : 0;

    td.dtypeKey = isBf16 ? 0 : 1;
    td.runModeKey = static_cast<uint32_t>(runMode);
    td.widthKey = (width == 2)   ? CAUSAL_CONV1D_TPL_WIDTH_2
                  : (width == 3) ? CAUSAL_CONV1D_TPL_WIDTH_3
                                 : CAUSAL_CONV1D_TPL_WIDTH_4;

    if (runMode == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE) {
        UpdateDimTileChoice choice = ChooseUpdateBaseDimChoice(batch, dim, numCores);
        if (choice.baseDim <= 0 || choice.baseDimCnt <= 0) {
            choice.baseDim = (dim > 0 && dim <= MAX_DIM_TILE) ? dim : MAX_DIM_TILE;
            choice.baseDimCnt = (choice.baseDim > 0) ? CeilDiv(dim, choice.baseDim) : 1;
            if (choice.baseDimCnt <= 0) {
                choice.baseDimCnt = 1;
            }
        }
        td.baseDim = choice.baseDim;
        td.baseDimCnt = choice.baseDimCnt;
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
        td.tokenBlockSize = 0;
        td.tokenBlockCnt = 0;
    } else if (dim <= MAX_DIM_TILE && numCores > 0) {
        td.baseDim = dim;
        td.baseDimCnt = 1;
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        int64_t tokenCoreBudget = numCores;
        int64_t idealBlockSize = CeilDiv(cuSeqlen, tokenCoreBudget);
        if (idealBlockSize <= 0) {
            idealBlockSize = 1;
        }
        td.tokenBlockSize = idealBlockSize;
        td.tokenBlockCnt = CeilDiv(cuSeqlen, td.tokenBlockSize);
    } else {
        td.baseDim = MAX_DIM_TILE;
        td.baseDimCnt = CeilDiv(dim, td.baseDim);
        td.fnPlanKey = CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        int64_t tokenCoreBudget = (numCores > 0) ? (numCores / td.baseDimCnt) : 1;
        if (tokenCoreBudget <= 0) {
            tokenCoreBudget = 1;
        }
        int64_t idealBlockSize = CeilDiv(cuSeqlen, tokenCoreBudget);
        if (idealBlockSize <= 0) {
            idealBlockSize = 1;
        }
        td.tokenBlockSize = idealBlockSize;
        td.tokenBlockCnt = CeilDiv(cuSeqlen, td.tokenBlockSize);
    }

    td.hasExplicitTokenSeqRanges = 0;
    td.explicitTokenSeqRangeCount = 0;
}

int64_t ComputeWorkspaceSize(int32_t blockDim, int64_t batch, int64_t width, int64_t dim, bool hasInitialState)
{
    if (!hasInitialState) {
        return 0;
    }
    constexpr int64_t kDtypeSize = 2;
    constexpr int64_t kSyncBytesPerBlock = 32;
    int64_t historyCount = (width - 1 > 0) ? width - 1 : 0;
    return ASCENDC_RESERVED_WORKSPACE + static_cast<int64_t>(blockDim) * kSyncBytesPerBlock +
           batch * historyCount * dim * kDtypeSize;
}

// Optional empty tensors have the same absence semantics as the Torch bridge.
bool HasElements(gert::TilingContext *context, size_t index)
{
    const auto *shape = context->GetOptionalInputShape(index);
    return shape && context->GetOptionalInputDesc(index) && shape->GetStorageShape().GetShapeSize() > 0;
}

ge::graphStatus TilingCausalConv1d(gert::TilingContext *context)
{
    if (!context || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    const auto *xShape = context->GetInputShape(0);
    const auto *weightShape = context->GetInputShape(1);
    const auto *stateShape = context->GetInputShape(2);
    const auto *xDesc = context->GetInputDesc(0);
    const auto *weightDesc = context->GetInputDesc(1);
    const auto *stateDesc = context->GetInputDesc(2);
    if (!xShape || !weightShape || !stateShape || !xDesc || !weightDesc || !stateDesc) return ge::GRAPH_FAILED;
    const auto &x = xShape->GetStorageShape();
    const auto &weight = weightShape->GetStorageShape();
    const auto &state = stateShape->GetStorageShape();
    const auto dtype = xDesc->GetDataType();
    if ((x.GetDimNum() != 2 && x.GetDimNum() != 3) || weight.GetDimNum() != 2 || state.GetDimNum() != 3 ||
        (dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16) || weightDesc->GetDataType() != dtype ||
        stateDesc->GetDataType() != dtype)
        return ge::GRAPH_FAILED;

    const int64_t dim = x.GetDimNum() == 2 ? x.GetDim(1) : x.GetDim(2);
    const int64_t width = weight.GetDim(0);
    if (weight.GetDim(1) != dim || state.GetDim(2) != dim || width < MIN_WIDTH || width > MAX_WIDTH)
        return ge::GRAPH_FAILED;
    const int64_t inputMode = x.GetDimNum() == 2 ? 0 : 1;
    const int64_t seqLen = inputMode == 1 ? x.GetDim(1) : 0;
    int64_t batch = inputMode == 1 ? x.GetDim(0) : 0;
    const int64_t cuSeqlen = inputMode == 0 ? x.GetDim(0) : batch * seqLen;
    if (inputMode == 0) {
        const auto *queryShape = context->GetOptionalInputShape(4);
        if (!queryShape || queryShape->GetStorageShape().GetDimNum() == 0 ||
            queryShape->GetStorageShape().GetDim(0) < 2)
            return ge::GRAPH_FAILED;
        batch = queryShape->GetStorageShape().GetDim(0) - 1;
    }

    const auto *attrs = context->GetAttrs();
    const auto *activation = attrs ? attrs->GetAttrPointer<int64_t>(0) : nullptr;
    const auto *pad = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const auto *mode = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    const int64_t activationMode = activation ? *activation : 0;
    const int64_t padSlotId = pad ? *pad : -1;
    const int64_t runMode = mode ? *mode : 0;
    const bool hasBias = HasElements(context, 3);
    const bool hasCacheIndices = HasElements(context, 5);
    const bool hasInitialState = HasElements(context, 6);
    const bool hasNumAccept = HasElements(context, 7);

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int32_t maxAivCore = static_cast<int32_t>(platform.GetCoreNumAiv());
    CausalConv1dTilingData tilingData;
    ComputeTilingData(dim, cuSeqlen, seqLen, batch, inputMode, width, state.GetDim(1), state.GetDim(0), activationMode,
                      padSlotId, hasBias, hasCacheIndices, hasInitialState, hasNumAccept, dtype == ge::DT_BF16,
                      maxAivCore, runMode, tilingData);
    const int64_t totalBlocks = runMode == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE
                                    ? tilingData.batch * tilingData.baseDimCnt
                                    : tilingData.tokenBlockCnt * tilingData.baseDimCnt;
    int32_t blockDim = std::min(maxAivCore, static_cast<int32_t>(totalBlocks));
    if (blockDim <= 0) blockDim = 1;

    // Retain the legacy system prefix, per-block sync area and initial-state snapshot.
    const int32_t libApiWorkspaceSize = static_cast<int32_t>(platform.GetLibApiWorkSpaceSize());
    const int64_t ws = ComputeWorkspaceSize(blockDim, batch, width, dim, hasInitialState);
    int64_t totalWorkspace = std::max(static_cast<int64_t>(libApiWorkspaceSize), ws);
    if (totalWorkspace <= 0) totalWorkspace = libApiWorkspaceSize;
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(tilingData) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &tilingData, sizeof(tilingData));
    raw->SetDataSize(sizeof(tilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0);
    workspace[0] = totalWorkspace;
    return ge::GRAPH_SUCCESS;
}

struct SglCausalConv1dCompileInfo {};
ge::graphStatus TilingPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(SglCausalConv1d).Tiling(TilingCausalConv1d).TilingParse<SglCausalConv1dCompileInfo>(TilingPrepare);
}  // namespace optiling
