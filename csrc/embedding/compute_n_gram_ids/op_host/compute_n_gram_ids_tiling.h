
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ComputeNGramIdsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, totalTask);
TILING_DATA_FIELD_DEF(int32_t, oeN);  // codespell:ignore oen
TILING_DATA_FIELD_DEF(int32_t, oeK);
TILING_DATA_FIELD_DEF(int32_t, maxContextLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SglComputeNGramIds, ComputeNGramIdsTilingData)
}  // namespace optiling
