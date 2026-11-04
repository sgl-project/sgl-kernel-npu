/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v3_apt.cpp
 * \brief
 */
#include "moe_v3_mrgsort_out.h"
#include "moe_v3_mrgsort.h"
#include "moe_v3_sort_one_core.h"
#include "moe_v3_sort_multi_core.h"
#include "moe_v3_expert_tokens_count.h"
#include "moe_v3_expert_tokens_count_v2.h"
#include "moe_v3_row_idx_gather.h"
#include "moe_v3_gather_out.h"
#include "moe_v3_gather_dynamic_quant.h"
#include "moe_v3_gather_mxfp8_quant.h"

/*
 * 非量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_GATHER 1000000     // 单核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER 1001000    // 单核排序、非量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER 1100000   // 多核排序、非量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER 1101000  // 多核排序、非量化、SCATTER索引

/*
 * 动态量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER 1020000     // 单核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER 1021000    // 单核排序、动态量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER 1120000   // 多核排序、动态量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER 1121000  // 多核排序、动态量化、SCATTER索引

/*
 * MXFP8量化
 */
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER 1030000     // 单核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER 1031000    // 单核排序、MXFP8量化、SCATTER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER 1130000   // 多核排序、MXFP8量化、GATHER索引
#define MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER 1131000  // 多核排序、MXFP8量化、SCATTER索引

using namespace AscendC;
using namespace MoeInitRoutingV3;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ArgsortExpandedRowIdxSimt(
    int64_t elements, int64_t indexBase, __gm__ int32_t *expandedRowIdxGmAddr,
    __gm__ int32_t *expandedRowIdxGatherGmAddr)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx()); index < static_cast<int32_t>(elements);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int64_t outIndices = expandedRowIdxGmAddr[indexBase + index];
        expandedRowIdxGatherGmAddr[outIndices] = indexBase + index;
    }
}

__aicore__ inline void moe_init_routing_v3(GM_ADDR expertIdx, GM_ADDR expandedRowIdx, GM_ADDR expertTokensCountOrCumsum,
                                           GM_ADDR workspace, const MoeInitRoutingV3Arch35TilingData *tilingData,
                                           uint64_t tilingKey, uint32_t rankId, uint32_t expertPerRank)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    if (workspace == nullptr) {
        return;
    }

    // 1.排序阶段，计算SortedExpertIdx和SortedRowIdx。若rowIdxType=1(Scatter)，则SortedRowIdx直接写到输出expandedRowIdx。
    TPipe sortPipe;
    if (tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_GATHER || tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_SCATTER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_GATHER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_DYNAMICQUANT_SCATTER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_GATHER ||
        tilingKey == MOE_INIT_ROUTING_V3_SORTONECORE_MXFP8QUANT_SCATTER) {
        // 单核排序
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, workspace, tilingData, &sortPipe);
        op.Process();
    } else if (tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_SCATTER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_DYNAMICQUANT_SCATTER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_GATHER ||
               tilingKey == MOE_INIT_ROUTING_V3_SORTMULTICORE_MXFP8QUANT_SCATTER) {
        // 多核排序
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, workspace, tilingData, &sortPipe);
        op.Process();
    }

    sortPipe.Destroy();
    // 2.TokensCount阶段，计算输出expertTokensCountOrCumsum
    // 使用localTilingData限制expertEnd为本卡需要的范围，避免计算远端卡数据
    TPipe histogramPipe;
    ExpertTokensCountV2 countOp;
    countOp.Init(expertIdx, expertTokensCountOrCumsum, tilingData, &histogramPipe);
    countOp.Process();
    histogramPipe.Destroy();
}
