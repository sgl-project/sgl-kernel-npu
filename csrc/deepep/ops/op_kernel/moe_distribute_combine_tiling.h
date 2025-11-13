#ifndef MOE_DISTRIBUTE_CMOBINE_A2_TILING_H
#define MOE_DISTRIBUTE_CMOBINE_A2_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeCombineA2Info {
    uint32_t epWorldSize;          // epWorldSize
    uint32_t tpWorldSize;          // tpWorldSize
    uint32_t epRankId;             // epRankId
    uint32_t tpRankId;             // tpRankId
    uint32_t expertSharedType;     // expert type
    uint32_t sharedExpertRankNum;  // shared expert number
    uint32_t moeExpertNum;         // moe expert number
    uint32_t globalBs;             // globalBs = BS * worldSize
    uint32_t bs;                   // bs
    uint32_t k;                    // k
    uint32_t h;                    // h
    uint32_t aivNum;               // aivNum
    uint64_t totalUbSize;          // epWorldSize
    uint32_t hcclBufferSize;       // HCCL windows, unit:B
    uint32_t rsd;
};

struct MoeDistributeCombineA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineA2Info moeDistributeCombineInfo;
};

#endif  //__MOE_DISTRIBUTE_CMOBINE_A2_TILING_H__
