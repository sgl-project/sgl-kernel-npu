/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_attention_score_tilingkey.h"
#include "sparse_attention_score_kernel_interface.cpp"
#include "kernel_common.hpp"  // SparseAttn::SparseAttentionScoreTilingData (host/kernel tiling blob contract)

// NOTE: use GM_ADDR here, not the equivalent `__gm__ uint8_t *`: CANN extract_host_stub.py
// mis-parses the explicit form, generating a broken indirect wrapper. Host passes raw GM addresses.
extern "C" __global__ __aicore__ void sparse_attention_score(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR selectIdx, GM_ADDR blockTable, GM_ADDR selectNumIdx,
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
    GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    // No GE SetTilingKey in native dispatch; read tilingKey from the tiling blob
    // instead (TILING_KEY_VAR unavailable). Arch kernel bodies are unchanged.
    auto tilingData = reinterpret_cast<__gm__ SparseAttn::SparseAttentionScoreTilingData *>(tiling);
    auto tilingKey = tilingData->tilingKey;
    if (tilingKey >= SASA_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        if (tilingKey == SASA_FP8_D128_TILING) {
            SasaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, SasaKernelArch35::Format::TND>(
                query, key, value, selectIdx, blockTable, selectNumIdx, actualSeqLengths, actualSeqLengthsKv,
                qDequantScale, kDequantScale, vDequantScale, attentionOut, softmaxLse, user, tiling);
        } else if (tilingKey == SASA_FP16_D128_TILING) {
            SasaInferIntfRegular<half, half, float, SasaKernelArch35::Format::TND>(
                query, key, value, selectIdx, blockTable, selectNumIdx, actualSeqLengths, actualSeqLengthsKv,
                attentionOut, softmaxLse, user, tiling);
        } else if (tilingKey == SASA_BF16_D128_TILING) {
            SasaInferIntfRegular<bfloat16_t, bfloat16_t, float, SasaKernelArch35::Format::TND>(
                query, key, value, selectIdx, blockTable, selectNumIdx, actualSeqLengths, actualSeqLengthsKv,
                attentionOut, softmaxLse, user, tiling);
        }
#elif (__CCE_AICORE__ == 220)
        if (tilingKey == SASA_FP16_D128_ARCH22_TILING) {
            SasaInferIntfRegularArch22<half, float>(query, key, value, selectIdx, blockTable, selectNumIdx,
                                                    actualSeqLengths, actualSeqLengthsKv, attentionOut, softmaxLse,
                                                    user, tiling);
        } else if (tilingKey == SASA_BF16_D128_ARCH22_TILING) {
            SasaInferIntfRegularArch22<bfloat16_t, float>(query, key, value, selectIdx, blockTable, selectNumIdx,
                                                          actualSeqLengths, actualSeqLengthsKv, attentionOut,
                                                          softmaxLse, user, tiling);
        }
#endif
    }
}
