/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_apt.cpp
 * \brief direct-launch entry of chunk_gated_delta_rule (arch35 / Ascend950)
 *
 * Mirrors op_kernel/chunk_gated_delta_rule_apt.cpp from the aclnn source, but
 * replaces GET_TILING_DATA (unavailable under direct-launch compilation) with a
 * GM->stack tiling copy identical to the arch22 direct-launch entry, and
 * replaces TILING_KEY_IS dispatch (no TilingContext under direct-launch) with
 * dispatch on the tiling struct fields stateIsFp32 / hasGamma, which encode the
 * same four variants defined in chunk_gated_delta_rule_tiling_key.h.
 */

#include "arch35/chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_tiling_key.h"

using namespace AscendC;
using namespace matmul;
using namespace ChunkGatedDeltaRule;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                                             GM_ADDR initialState, GM_ADDR seqlens, GM_ADDR gOptional,
                                                             GM_ADDR out, GM_ADDR finalState, GM_ADDR chunkState,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    constexpr size_t TILING_WORDS = (sizeof(ChunkGatedDeltaRuleTilingData) + 7) / 8;
    alignas(8) uint64_t tilingBuf[TILING_WORDS];
    const __gm__ uint64_t *gmTilingWords = reinterpret_cast<const __gm__ uint64_t *>(tilingGM);
    for (size_t i = 0; i < TILING_WORDS; ++i) {
        tilingBuf[i] = gmTilingWords[i];
    }
    ChunkGatedDeltaRuleTilingData *tilingData = reinterpret_cast<ChunkGatedDeltaRuleTilingData *>(tilingBuf);

    CGDRInitParams initParams{query, key, value, beta, initialState, seqlens, gOptional, out, finalState, chunkState};

    if (tilingData->stateIsFp32 != 0 && tilingData->hasGamma != 0) {
        CGDR<bfloat16_t, float, float, true> op(&pipe, tilingData);
        op.Init(initParams, user);
        op.Process();
    } else if (tilingData->stateIsFp32 != 0) {
        CGDR<bfloat16_t, float, float, false> op(&pipe, tilingData);
        op.Init(initParams, user);
        op.Process();
    } else if (tilingData->hasGamma != 0) {
        CGDR<bfloat16_t, float, bfloat16_t, true> op(&pipe, tilingData);
        op.Init(initParams, user);
        op.Process();
    } else {
        CGDR<bfloat16_t, float, bfloat16_t, false> op(&pipe, tilingData);
        op.Init(initParams, user);
        op.Process();
    }
}
