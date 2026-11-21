/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Tiling data for the A3 short-prefill QSA kernel.
 */
#ifndef QSA_CUBE_ATTENTION_TILING_H
#define QSA_CUBE_ATTENTION_TILING_H
#include <cstdint>
#include <register/tilingdata_base.h>
namespace optiling {
BEGIN_TILING_DATA_DEF(SglQsaCubeAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, numQueryHeads);
TILING_DATA_FIELD_DEF(uint32_t, numRequests);
TILING_DATA_FIELD_DEF(uint32_t, numCacheBlocks);
TILING_DATA_FIELD_DEF(uint32_t, pageTokens);
TILING_DATA_FIELD_DEF(uint32_t, pageTableWidth);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, numTiles);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SglQsaCubeAttention, SglQsaCubeAttentionTilingData)
struct SglQsaCubeAttentionCompileInfo {};
}  // namespace optiling
#endif
