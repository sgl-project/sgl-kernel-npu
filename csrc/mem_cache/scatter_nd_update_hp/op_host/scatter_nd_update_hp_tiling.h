/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Compile information and host tiling implementation for arch22.
#pragma once

#include "../op_kernel/scatter_nd_update_hp_tiling_arch22.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
struct ScatterNdUpdateSkArch22CompileInfo {
    uint32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

class ScatterNdUpdateSkArch22Tiling
{
public:
    explicit ScatterNdUpdateSkArch22Tiling(gert::TilingContext *context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline bool IsSort(uint64_t totalLength) const;
    inline bool IsLinearIndex(uint64_t totalLength) const;
    inline size_t CalcWorkSpaceSize();
    inline void SetTilingKeyMode();
    inline void GetDtypeSize();
    inline void Tiling4Scatter(uint64_t totalLength);
    inline void Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim);
    inline void Tiling4Hp(uint64_t indexRow);
    inline uint64_t Tiling4HpScatterShape();
    inline void Tiling4HpIndexTile(uint64_t updateUbBytes);
    inline void Tiling4HpCorePartition(uint64_t indexRow);
    inline ge::graphStatus HandleViewStride();

    ScatterNdUpdateSkArch22TilingData tilingData_;
    gert::TilingContext *tilingContext_ = nullptr;

    uint64_t coreNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t ubSize_ = 0;
    bool isSort_ = false;
    uint64_t dataTypeSize_ = 0;
    bool isInt64Indices_ = false;
    bool needLargeIndexKernel_ = false;
    uint64_t isViewStride0_ = 0;
    uint64_t varStride0Elements_ = 0;
    uint64_t firstDimStrideRows_ = 1;

private:
    // LinearIndex
    uint64_t indexDim_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t blockRemainLength_ = 0;
    uint64_t tailBlockNum_ = 0;
    uint64_t frontBlockNum_ = 0;
    uint64_t frontCoreNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t indicesMask_[MAX_DIM_NUM] = {0};

    // Scatter
    uint64_t scatterLength_ = 1;
    uint64_t tailRow_ = 0;
    uint64_t frontRow_ = 0;
    uint64_t frontNum_ = 0;
    uint64_t tailNum_ = 0;
    uint64_t ubLengthForUpdates_ = 0;
    uint64_t scatterAlignLength_ = 0;
    uint64_t formDim_ = 0;
    uint64_t copyRow_ = 0;
    uint64_t scatterTileNum_ = 1;
    uint64_t scatterTileLength_ = 0;
    uint64_t scatterTileTail_ = 0;
    uint64_t scatterTileAlignLength_ = 0;

    // HighPerformance
    uint64_t hpCoreNum_ = 0;
    uint64_t hpFrontIndexNum_ = 0;
    uint64_t hpTailIndexNum_ = 0;
    uint64_t hpFrontCoreNum_ = 0;
    uint64_t hpTailCoreNum_ = 0;
    uint64_t hpIndexTileLength_ = 0;
    uint64_t hpScatterTileLength_ = 0;
    uint64_t hpScatterTileNum_ = 1;
    uint64_t hpScatterTileTail_ = 0;
    uint64_t hpRowBytesAligned_ = 0;
    uint64_t hpRowsPerBatch_ = 1;
};
}  // namespace optiling
