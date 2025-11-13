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
 * \file moe_distribute_dispatch_neg_one_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_NEG_ONE_TILING_H
#define MOE_DISTRIBUTE_DISPATCH_NEG_ONE_TILING_H

struct MoeDistributeDispatchNegOneInfo {
    uint32_t epWorldSize;          // epWorldSize
    uint32_t tpWorldSize;          // tpWorldSize
    uint32_t epRankId;             // epRankId
    uint32_t tpRankId;             // tpRankId
    uint32_t expertShardType;      // expert type
    uint32_t sharedExpertNum;      // shared expert number
    uint32_t sharedExpertRankNum;  // shared expert rank number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t quantMode;            // quant mode
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    bool isTokenMask;              // input active mask 1dims or not
    bool isExpertMask;             // input active mask 2dims or not
    bool hasElasticInfo;           // has elasticinfo or not
    bool reserved3;                // reserved
    uint64_t totalUbSize;          // epWorldSize
    uint64_t totalWinSize;
    uint32_t expertTokenNumsType;  // expert token nums type, support 0: cumsum mode, 1: count mode
    int32_t zeroComputeExpertNum;  // sum of zero、copy and const expert nums
    uint32_t cumSumUBMinValue;     // Minimum value for CumSum remainder（in UB）
};

struct MoeDistributeDispatchNegOneTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    MoeDistributeDispatchNegOneInfo moeDistributeDispatchNegOneInfo;
};

#endif
