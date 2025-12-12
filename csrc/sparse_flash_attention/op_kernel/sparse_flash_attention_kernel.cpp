/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file sparse_flash_attention.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_flash_attention_kernel_mla.h"

using namespace AscendC;
using namespace sglang::SFAHost;

extern "C" __global__ __aicore__ void sparse_flash_attention(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR sparseIndices, GM_ADDR blocktable,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKV,
    GM_ADDR queryRope, GM_ADDR keyRope,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // 1. 獲取 Tiling Data 並讀取 Key
    auto tilingData = reinterpret_cast<__gm__ sglang::SFAHost::SparseFlashAttentionTilingDataMla *>(tiling);
    auto tilingKey = tilingData->tilingKey;

    // 2. 定義不同配置的算子實例
    // 注意：SFAType 的模板參數順序需與頭文件 sparse_flash_attention_kernel_mla.h 中定義的 class SparseFlashAttentionMla 一致
    // 假設順序為: <typename Q_T, typename KV_T, typename OUT_T, bool FLASH_DECODE, SFA_LAYOUT LAYOUT, SFA_LAYOUT KV_LAYOUT, int TEMPLATE_MODE>
    
    SparseFlashAttentionMla<SFAType<half, half, half, false, SFA_LAYOUT::BSND, SFA_LAYOUT::BSND, 0>> op_fp16;
    SparseFlashAttentionMla<SFAType<bfloat16_t, bfloat16_t, bfloat16_t, false, SFA_LAYOUT::BSND, SFA_LAYOUT::BSND, 0>> op_bf16;

    // 3. 根據 Tiling Key 進行運行時分發
    switch (tilingKey) {
        case 1: 
            op_fp16.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,
                         blocktable, queryRope, keyRope, attentionOut, user, tilingData, tiling, &tPipe);
            op_fp16.Process();
            break;
            
        case 2: 
            op_bf16.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,
                         blocktable, queryRope, keyRope, attentionOut, user, tilingData, tiling, &tPipe);
            op_bf16.Process();
            break;
        default:
            break;
    }
}