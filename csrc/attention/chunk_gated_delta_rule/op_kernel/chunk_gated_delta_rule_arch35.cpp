/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License in the software repository for the full text of the License.
 */
// Provenance: cann/ops-transformer attention/chunk_gated_delta_rule/ -> sgl-kernel-npu
// csrc/chunk_gated_delta_rule/op_kernel/chunk_gated_delta_rule_apt.cpp -> attention CANN copy. Original import revision
// is unknown; this copy preserves the local computation.

// Native CANN entry (arch35); tiling fields retain the legacy dispatch.

#include "arch35/chunk_gated_delta_rule.h"
#include "chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_tiling_key.h"

using namespace AscendC;
using namespace matmul;
using namespace ChunkGatedDeltaRule;

extern "C" __global__ __aicore__ void sgl_chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                 GM_ADDR beta, GM_ADDR initialState, GM_ADDR seqlens,
                                                                 GM_ADDR gOptional, GM_ADDR chunkState, GM_ADDR out,
                                                                 GM_ADDR finalState, GM_ADDR chunkStateRef,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    GET_TILING_DATA(data, tilingGM);
    ChunkGatedDeltaRuleTilingData *tilingData = &data;

    CGDRInitParams initParams{query,   key,       value, beta,       initialState,
                              seqlens, gOptional, out,   finalState, chunkStateRef};

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
