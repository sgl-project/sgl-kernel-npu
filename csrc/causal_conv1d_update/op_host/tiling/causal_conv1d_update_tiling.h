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
 * \file causal_conv1d_update_tiling.h
 * \brief tiling data struct for host side
 */

#ifndef CAUSAL_CONV1D_UPDATE_TILING_HOST_H_
#define CAUSAL_CONV1D_UPDATE_TILING_HOST_H_

#include <cstdint>

struct CausalConv1dUpdateTilingData {
    // used core num
    int64_t numCore;

    // batch per core
    int64_t blockFactor;
    int64_t blockTailFactor;

    // x [batch, seqLen, dim]
    // weight [width, dim]
    int64_t batch;
    int64_t seqLen;
    int64_t dim;
    int64_t width;
    int64_t stateLen;

    int64_t hasIndices;
    int64_t hasBias;
    int64_t hasNumAccept;
    int64_t hasQueryLoc;
    int64_t activationMode;
    int64_t padSlotId;
};

namespace SGLang {
namespace CausalConv1dUpdate {

// Helper function to compute tiling data
inline void ComputeTilingData(
    const int64_t batch,
    const int64_t seq_len,
    const int64_t dim,
    const int64_t width,
    const int64_t state_len,
    const bool has_indices,
    const bool has_bias,
    const bool has_num_accept,
    const bool has_query_loc,
    const bool activation_mode,
    const int64_t pad_slot_id,
    const int32_t num_cores,
    CausalConv1dUpdateTilingData& tiling_data
) {
    tiling_data.batch = batch;
    tiling_data.seqLen = seq_len;
    tiling_data.dim = dim;
    tiling_data.width = width;
    tiling_data.stateLen = state_len;
    tiling_data.hasIndices = has_indices ? 1 : 0;
    tiling_data.hasBias = has_bias ? 1 : 0;
    tiling_data.hasNumAccept = has_num_accept ? 1 : 0;
    tiling_data.hasQueryLoc = has_query_loc ? 1 : 0;
    tiling_data.activationMode = activation_mode ? 1 : 0;
    tiling_data.padSlotId = pad_slot_id;

    // Compute block factor
    tiling_data.numCore = num_cores;
    tiling_data.blockFactor = (batch + num_cores - 1) / num_cores;
    tiling_data.blockTailFactor = batch - tiling_data.blockFactor * (num_cores - 1);
    if (tiling_data.blockTailFactor <= 0) {
        tiling_data.blockTailFactor = tiling_data.blockFactor;
    }
}

} // namespace CausalConv1dUpdate
} // namespace SGLang

#endif // CAUSAL_CONV1D_UPDATE_TILING_HOST_H_
