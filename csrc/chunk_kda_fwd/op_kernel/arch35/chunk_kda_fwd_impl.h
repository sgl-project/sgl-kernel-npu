/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../chunk_kda_fwd_common.h"
#include "chunk_kda_fwd_fwd_h.h"

namespace KdaForward::arch35 {

template <bool SAFE_GATE, typename T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void Run(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling, AscendC::TPipe &pipe)
{
    const auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, h, userWorkspace, tiling);
    RunFrontEnd<SAFE_GATE, T, float, TilingData,
                COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, aqk, akk, addresses, userWorkspace, tiling, pipe);
    if (tiling.useDenseFwdH) {
        ChunkKdaFwdFwdH<T, float, TilingData> fwdH;
        fwdH.Init(
            addresses.gk, initialState, attnOut, addresses.finalState,
            aqk, akk, addresses.w, addresses.u, addresses.qgScaled,
            addresses.kg, addresses.vNew, addresses.h, tiling);
        fwdH.Process();
        return;
    }

    const int64_t fwdHTaskCount =
        (tiling.isVarLen ? tiling.seqNum : tiling.batch) * tiling.vHeadNum;
    const bool isolateGenericBackEnd =
        (!tiling.isVarLen && tiling.seqlen % tiling.chunkSize == 0) ||
        fwdHTaskCount > tiling.prepareUsedCoreNum;
    if (isolateGenericBackEnd) {
        pipe.Destroy();
        RunGenericBackEnd<T, TilingData>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, tiling);
    } else {
        RunGenericBackEnd<T, TilingData>(
            q, k, v, beta, initialState, cuSeqlens, chunkIndices, aqk,
            attnOut, addresses, userWorkspace, tiling, pipe);
    }
}

} // namespace KdaForward::arch35
