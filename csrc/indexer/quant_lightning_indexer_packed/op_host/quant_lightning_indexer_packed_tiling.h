// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#pragma once

#include "../op_kernel/quant_lightning_indexer_tiling.h"
#include <type_traits>

namespace QLIPacked {
static_assert(std::is_trivial<QLITilingData>::value && std::is_standard_layout<QLITilingData>::value,
              "QLITilingData must remain a POD for direct kernel launch");
static_assert(sizeof(QLITilingData) == 13 * sizeof(uint32_t), "Unexpected QLITilingData layout");

// Host-only, shape-based tiling for TND query and a packed PA_BSND key cache.
inline QLITilingData MakeTilingData(uint32_t batch, uint32_t tokens, uint32_t groups, uint32_t pageSize,
                                    uint32_t maxPages, uint32_t sparseCount, uint32_t sparseMode,
                                    uint32_t usedCores)
{
    QLITilingData tiling{};
    tiling.bSize = batch;
    tiling.tSize = tokens;
    tiling.n2Size = 1; // Packed index keys have one head.
    tiling.gSize = groups;
    tiling.s1Size = 0; // TND uses cumulative query lengths instead.
    tiling.s2Size = maxPages * pageSize;
    tiling.sparseCount = sparseCount;
    tiling.keyStride0 = pageSize * TOKEN_BYTES;
    tiling.keyDequantScaleStride0 = pageSize * TOKEN_BYTES / SCALE_BYTES;
    tiling.usedCoreNum = usedCores;
    tiling.blockSize = pageSize;
    tiling.maxBlockNumPerBatch = maxPages;
    tiling.sparseMode = sparseMode;
    return tiling;
}
} // namespace QLIPacked
