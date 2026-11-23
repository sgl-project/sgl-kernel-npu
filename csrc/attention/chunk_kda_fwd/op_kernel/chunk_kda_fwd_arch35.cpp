/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Key-based build follows ops-transformer attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_common.h"
#include "chunk_kda_fwd_tiling_data.h"

// Keep the fast implementation out of the generic key's compiler input.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && (!defined(TILING_KEY_VAR) || TILING_KEY_VAR == 2UL)
#define KDA_COMPILE_ARCH35_FAST_PATH
#include "arch35/chunk_kda_fwd_impl.h"
#endif

namespace KdaForward {

#if !defined(TILING_KEY_VAR) || TILING_KEY_VAR == 1UL
template <typename T, typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog,
                                               GM_ADDR dtBias, GM_ADDR initialState, GM_ADDR cuSeqlens,
                                               GM_ADDR chunkIndices, GM_ADDR attnOut, GM_ADDR finalState, GM_ADDR gk,
                                               GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
                                               GM_ADDR vNew, GM_ADDR h, GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        RunGeneric<true, T, TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens, chunkIndices, attnOut, finalState, gk, aqk, akk, w,
            u, qg, kg, vNew, h, userWorkspace, tiling);
    } else {
        RunGeneric<false, T, TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens, chunkIndices, attnOut, finalState, gk, aqk, akk, w,
            u, qg, kg, vNew, h, userWorkspace, tiling);
    }
}
#endif

#ifdef KDA_COMPILE_ARCH35_FAST_PATH
template <typename T, typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta, GM_ADDR aLog,
                                              GM_ADDR dtBias, GM_ADDR initialState, GM_ADDR cuSeqlens,
                                              GM_ADDR chunkIndices, GM_ADDR attnOut, GM_ADDR finalState, GM_ADDR gk,
                                              GM_ADDR aqk, GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg,
                                              GM_ADDR vNew, GM_ADDR h, GM_ADDR userWorkspace, const TilingData &tiling)
{
    AscendC::TPipe pipe;
    if (tiling.safeGate) {
        arch35::Run<true, T, TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens, chunkIndices, attnOut, finalState, gk, aqk, akk, w,
            u, qg, kg, vNew, h, userWorkspace, tiling, pipe);
    } else {
        arch35::Run<false, T, TilingData, COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens, chunkIndices, attnOut, finalState, gk, aqk, akk, w,
            u, qg, kg, vNew, h, userWorkspace, tiling, pipe);
    }
}
#endif

}  // namespace KdaForward

using optiling::ChunkKdaFwdTilingData;

extern "C" __global__ __aicore__ void sgl_chunk_kda_fwd(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
                                                        GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
                                                        GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
                                                        GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
                                                        GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new,
                                                        GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkKdaFwdTilingData);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    constexpr size_t TILING_WORDS = (sizeof(ChunkKdaFwdTilingData) + 7) / 8;
    alignas(8) uint64_t tilingBuf[TILING_WORDS];
    const __gm__ uint64_t *gmTilingWords = reinterpret_cast<const __gm__ uint64_t *>(tiling);
    for (size_t i = 0; i < TILING_WORDS; ++i) {
        tilingBuf[i] = gmTilingWords[i];
    }
    ChunkKdaFwdTilingData *tilingData = reinterpret_cast<ChunkKdaFwdTilingData *>(tilingBuf);

    if (TILING_KEY_IS(1)) {
#if !defined(TILING_KEY_VAR) || TILING_KEY_VAR == 1UL
        KdaForward::DispatchGenericSafeGate<bfloat16_t, ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens, chunk_indices, attn_out, final_state, gk, aqk,
            akk, w, u, qg, kg, v_new, h, userWorkspace, *tilingData);
#endif
    } else if (TILING_KEY_IS(2)) {
#ifdef KDA_COMPILE_ARCH35_FAST_PATH
        KdaForward::DispatchArch35SafeGate<bfloat16_t, ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens, chunk_indices, attn_out, final_state, gk, aqk,
            akk, w, u, qg, kg, v_new, h, userWorkspace, *tilingData);
#endif
    }
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
