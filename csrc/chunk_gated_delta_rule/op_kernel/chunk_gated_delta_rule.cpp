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
 * \file chunk_gated_delta_rule.cpp
 * \brief direct-launch entry of chunk_gated_delta_rule (arch22)
 */

#include "arch22/chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule_tiling_data.h"

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

    CGDR<bfloat16_t, float> op(&pipe, tilingData);
    CGDRInitParams initParams{query, key, value, beta, initialState, seqlens, gOptional, out, finalState, chunkState};
    op.Init(initParams, user);
    op.Process();
}
