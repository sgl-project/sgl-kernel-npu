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
    uint32_t sharedExpertNum;      // shared expert number
    uint32_t sharedExpertRankNum;  // shared expert rank number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t moeExpertPerRankNum;  // moe expert per rank
    uint32_t zeroExpertNum;        // zero expert number
    uint32_t copyExpertNum;        // copy expert number
    uint32_t constExpertNum;       // const expert number
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;          // totalUbSize
    uint64_t totalWinSize;         // totalWinSize
    bool isTokenMask;              // input active mask 1dims or not
    bool isExpertMask;             // input active mask 2dims or not
    bool hasSharedExpertX;         // input shared expert x or not
    int8_t reserved[7];            // Pad 7 int8 for memory alignment
};

struct MoeDistributeCombineV2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineV2Info moeDistributeCombineV2Info;
};

#endif  //__MOE_DISTRIBUTE_CMOBINE_A2_TILING_H__
