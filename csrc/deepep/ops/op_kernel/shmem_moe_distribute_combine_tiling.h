/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#ifndef ASCENDC_SHMEM_COMM_MOE_DISTRIBUTE_CMOBINE_TILING_H
#define ASCENDC_SHMEM_COMM_MOE_DISTRIBUTE_CMOBINE_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// a3
struct ShmemMoeDistributeCombineInfo {
    uint32_t epWorldSize;               // epWorldSize
    uint32_t tpWorldSize;               // tpWorldSize
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    uint64_t magic;
    uint64_t shmemptr;                  // shmem ptr
};
struct ShmemMoeDistributeCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    ShmemMoeDistributeCombineInfo moeDistributeCombineInfo;
};

#endif //__ASCENDC_SHMEM_COMM_MOE_DISTRIBUTE_CMOBINE_TILING_H__
