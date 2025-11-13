/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_neg_one_tiling.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_NEG_ONE_TILING_H
#define MOE_DISTRIBUTE_COMBINE_NEG_ONE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// a3
struct MoeDistributeCombineNegOneInfo {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t zeroExpertNum;
    uint32_t copyExpertNum;
    uint32_t constExpertNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    bool isTokenMask;       // input active mask 1dims or not
    bool isExpertMask;      // input active mask 2dims or not
    bool hasSharedExpertX;  // input shared expert x or not
    bool hasElasticInfo;    // has elasticinfo or not
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    float armAvgFactor;
    float epsilon;
};
struct MoeDistributeCombineNegOneTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeCombineNegOneInfo moeDistributeCombineNegOneInfo;
};

#endif  //__MOE_DISTRIBUTE_COMBINE_NEG_ONE_TILING_H__
