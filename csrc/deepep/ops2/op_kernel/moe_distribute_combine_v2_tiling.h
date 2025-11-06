/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_combine_tiling.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_CMOBINE_A2_TILING_H
#define MOE_DISTRIBUTE_CMOBINE_A2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeCombineV2Info {
    uint32_t epWorldSize;          // epWorldSize
    uint32_t tpWorldSize;          // tpWorldSize
    uint32_t epRankId;             // epRankId
    uint32_t tpRankId;             // tpRankId
    uint32_t expertSharedType;     // expert type
    uint32_t sharedExpertRankNum;  // shared expert number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t zeroExpertNum;        // zero expert number
    uint32_t copyExpertNum;        // copy expert number
    uint32_t constExpertNum;       // const expert number
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;          // epWorldSize
    bool isTokenMask;              // input active mask 1dims or not
    bool isExpertMask;             // input active mask 2dims or not
    int8_t reserved[7];            // Pad 7 int8 for memory alignment
};

struct MoeDistributeCombineV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineV2Info moeDistributeCombineV2Info;
};

#endif  //__MOE_DISTRIBUTE_CMOBINE_A2_TILING_H__
