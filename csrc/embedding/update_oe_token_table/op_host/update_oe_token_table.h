
#ifndef __UPDATE_OE_TOKEN_TABLE_H
#define __UPDATE_OE_TOKEN_TABLE_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UpdateOeTokenTableTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockFactor);
TILING_DATA_FIELD_DEF(uint32_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, maxContextLen);
TILING_DATA_FIELD_DEF(uint32_t, ignoreTokenNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SglUpdateOeTokenTable, UpdateOeTokenTableTilingData)
}  // namespace optiling

#endif  // __UPDATE_OE_TOKEN_TABLE_H
