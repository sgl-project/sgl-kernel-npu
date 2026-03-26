/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling.h
 * \brief host-side tiling helpers for causal_conv1d
 */

#ifndef CAUSAL_CONV1D_TILING_HOST_H_
#define CAUSAL_CONV1D_TILING_HOST_H_

#include <array>
#include <cstdint>
#include <limits>

#include "causal_conv1d_tiling_data.h"

namespace SGLang {
namespace CausalConv1d {

struct DimTileChoice {
    int64_t dimTileSize = 0;
    int64_t blocksPerSeq = 0;
    int64_t gridSize = 0;
};

inline DimTileChoice ChooseDimTileSize(int64_t batch, int64_t dim, int32_t core_num)
{
    constexpr std::array<int64_t, 4> kCandidates = {4096, 2048, 1024, 512};
    DimTileChoice best_over;
    int64_t best_over_gap = std::numeric_limits<int64_t>::max();
    DimTileChoice best_under;

    for (const int64_t dim_tile_size : kCandidates) {
        if (dim % dim_tile_size != 0) {
            continue;
        }
        const int64_t blocks_per_seq = dim / dim_tile_size;
        const int64_t grid_size = batch * blocks_per_seq;
        if (grid_size <= 0) {
            continue;
        }

        if (grid_size >= static_cast<int64_t>(core_num)) {
            const int64_t gap = grid_size - static_cast<int64_t>(core_num);
            if (gap < best_over_gap) {
                best_over = {dim_tile_size, blocks_per_seq, grid_size};
                best_over_gap = gap;
            }
        } else if (grid_size > best_under.gridSize ||
                   (grid_size == best_under.gridSize && dim_tile_size < best_under.dimTileSize)) {
            best_under = {dim_tile_size, blocks_per_seq, grid_size};
        }
    }

    return best_over.dimTileSize != 0 ? best_over : best_under;
}

inline void ComputeTilingData(
    int64_t batch,
    int64_t cu_seqlen,
    int64_t seq_len,
    int64_t input_mode,
    int64_t dim,
    int64_t width,
    int64_t state_len,
    int64_t num_cache_lines,
    bool has_bias,
    bool activation_mode,
    int64_t pad_slot_id,
    int32_t core_num,
    sglang::npu_kernel::CausalConv1dTilingData& tiling_data)
{
    tiling_data.dim = dim;
    tiling_data.cuSeqlen = cu_seqlen;
    tiling_data.seqLen = seq_len;
    tiling_data.inputMode = input_mode;
    tiling_data.width = width;
    tiling_data.stateLen = state_len;
    tiling_data.numCacheLines = num_cache_lines;
    tiling_data.batch = batch;
    tiling_data.activationMode = activation_mode ? 1 : 0;
    tiling_data.padSlotId = pad_slot_id;
    tiling_data.hasBias = has_bias ? 1 : 0;

    const DimTileChoice choice = ChooseDimTileSize(batch, dim, core_num);
    tiling_data.dimTileSize = choice.dimTileSize;
    tiling_data.blocksPerSeq = choice.blocksPerSeq;
}

}  // namespace CausalConv1d
}  // namespace SGLang

#endif  // CAUSAL_CONV1D_TILING_HOST_H_
