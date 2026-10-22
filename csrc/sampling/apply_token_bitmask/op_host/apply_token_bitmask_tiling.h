#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SglApplyTokenBitMaskTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numRows);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, logitsStride);
    TILING_DATA_FIELD_DEF(uint32_t, bitmaskStride);
    TILING_DATA_FIELD_DEF(uint32_t, baseRows);
    TILING_DATA_FIELD_DEF(uint32_t, extraCores);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SglApplyTokenBitMask, SglApplyTokenBitMaskTilingData)

}  // namespace optiling
