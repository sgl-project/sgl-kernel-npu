/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Serialized tiling data shared through the generated kernel tiling header.
#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint64_t MAX_DIM_NUM = 8;

BEGIN_TILING_DATA_DEF(ScatterNdUpdateSkScatterTiling)
TILING_DATA_FIELD_DEF(uint64_t, scatterLength)
TILING_DATA_FIELD_DEF(uint64_t, tailRow)
TILING_DATA_FIELD_DEF(uint64_t, frontRow)
TILING_DATA_FIELD_DEF(uint64_t, frontNum)
TILING_DATA_FIELD_DEF(uint64_t, tailNum)
TILING_DATA_FIELD_DEF(uint64_t, ubLengthForUpdates)
TILING_DATA_FIELD_DEF(uint64_t, scatterAlignLength)
TILING_DATA_FIELD_DEF(uint64_t, formDim)
TILING_DATA_FIELD_DEF(uint64_t, copyRow)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileNum)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileLength)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileTail)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileAlignLength)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateSkScatterTilingOp, ScatterNdUpdateSkScatterTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateSkLinearIndexTiling)
TILING_DATA_FIELD_DEF(uint64_t, coreNum)
TILING_DATA_FIELD_DEF(uint64_t, ubSize)
TILING_DATA_FIELD_DEF(uint64_t, indexDim)
TILING_DATA_FIELD_DEF(uint64_t, blockLength)
TILING_DATA_FIELD_DEF(uint64_t, blockNum)
TILING_DATA_FIELD_DEF(uint64_t, blockRemainLength)
TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, sortWorkspace)
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, indicesMask)
TILING_DATA_FIELD_DEF(uint64_t, isInt64Indices)
TILING_DATA_FIELD_DEF(uint64_t, needLargeIndexKernel)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateSkLinearIndexTilingOp, ScatterNdUpdateSkLinearIndexTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateSkViewTiling)
TILING_DATA_FIELD_DEF(uint64_t, isViewStride0)
TILING_DATA_FIELD_DEF(uint64_t, varStride0Elements)
TILING_DATA_FIELD_DEF(uint64_t, firstDimStrideRows)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateSkViewTilingOp, ScatterNdUpdateSkViewTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateSkHpTiling)
TILING_DATA_FIELD_DEF(uint64_t, hpCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpFrontIndexNum)
TILING_DATA_FIELD_DEF(uint64_t, hpTailIndexNum)
TILING_DATA_FIELD_DEF(uint64_t, hpFrontCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpTailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpIndexTileLength)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileLength)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileNum)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileTail)
TILING_DATA_FIELD_DEF(uint64_t, hpRowBytesAligned)
TILING_DATA_FIELD_DEF(uint64_t, hpRowsPerBatch)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateSkHpTilingOp, ScatterNdUpdateSkHpTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateSkArch22TilingData)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateSkScatterTiling, scatterTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateSkLinearIndexTiling, linearIndexTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateSkHpTiling, hpTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateSkViewTiling, viewTiling)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SglScatterNdUpdateHp, ScatterNdUpdateSkArch22TilingData)
REGISTER_TILING_DATA_CLASS(ScatterNdUpdateSkTilingDataOp, ScatterNdUpdateSkArch22TilingData)

}  // namespace optiling
