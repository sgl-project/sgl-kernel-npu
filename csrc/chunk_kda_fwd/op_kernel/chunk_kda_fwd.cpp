/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_common.h"
#include "chunk_kda_fwd_tiling_data.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define KDA_COMPILE_ARCH35_FAST_PATH 1
#include "arch35/chunk_kda_fwd_impl.h"
#else
#define KDA_COMPILE_ARCH35_FAST_PATH 0
#endif

namespace KdaForward {

template <bool SAFE_GATE, typename T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    RunGeneric<SAFE_GATE, T, TilingData,
               COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, h, userWorkspace, tiling);
}

#if KDA_COMPILE_ARCH35_FAST_PATH
template <typename T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    AscendC::TPipe pipe;
    if (tiling.safeGate) {
        arch35::Run<true, T, TilingData,
                    COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    } else {
        arch35::Run<false, T, TilingData,
                    COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    }
}
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR,
    const TilingData &)
{
}
#endif

template <typename T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchGeneric<true, T, TilingData,
                        COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchGeneric<false, T, TilingData,
                        COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}

} // namespace KdaForward

using ChunkKdaFwd::ChunkKdaFwdTilingData;

// FwdH uses SyncAll<false>() and cross-core flags.  The original framework
// operator requests batch scheduling with TilingContext::SetScheduleMode(1).
// Kernel-launch operators must express the same requirement on the entrypoint
// so all requested cores are available together instead of being scheduled in
// partial waves.
//
// ascendc_library's generated device translation unit defines the guard below,
// temporarily rewrites __global__ to inline, and includes this source as the
// implementation body.  Schedule-mode attributes are invalid on that inline
// copy, so attach the attribute only to the real kernel declaration processed
// outside the generated inline-body pass.
#if defined(__CHUNK_KDA_FWD__KERNEL_FUN_H__)
#define KDA_FWD_SCHEDULE_MODE
#else
#define KDA_FWD_SCHEDULE_MODE __schedmode__(1)
#endif
extern "C" KDA_FWD_SCHEDULE_MODE __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(ChunkKdaFwdTilingData);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);

    // Direct-launch: copy the tiling blob from GM to the stack. The aclnn
    // GET_TILING_DATA_WITH_STRUCT macro is unavailable under the plain AscendC
    // direct-launch compilation, so the host-serialized struct is rehydrated
    // here exactly like chunk_gated_delta_rule does.
    constexpr size_t TILING_WORDS = (sizeof(ChunkKdaFwdTilingData) + 7) / 8;
    alignas(8) uint64_t tilingBuf[TILING_WORDS];
    const __gm__ uint64_t *gmTilingWords = reinterpret_cast<const __gm__ uint64_t *>(tiling);
    for (size_t i = 0; i < TILING_WORDS; ++i) {
        tilingBuf[i] = gmTilingWords[i];
    }
    ChunkKdaFwdTilingData *tilingData = reinterpret_cast<ChunkKdaFwdTilingData *>(tilingBuf);

    // Tiling key 2 is selected when chunk_size == 64 and K == V == 128; the
    // host encodes it implicitly through these shape fields, so dispatch on
    // them instead of the aclnn TILING_KEY_IS compile-time key.
    const bool useFastPath = tilingData->chunkSize == 64 && tilingData->kHeadDim == 128 &&
                             tilingData->vHeadDim == 128;
    if (useFastPath) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaForward::DispatchArch35SafeGate<bfloat16_t, ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, *tilingData);
#else
        KdaForward::DispatchGenericSafeGate<bfloat16_t, ChunkKdaFwdTilingData, 64, 128, 128>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, *tilingData);
#endif
    } else {
        KdaForward::DispatchGenericSafeGate<bfloat16_t, ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, *tilingData);
    }
}

#undef KDA_FWD_SCHEDULE_MODE

#undef KDA_COMPILE_ARCH35_FAST_PATH
