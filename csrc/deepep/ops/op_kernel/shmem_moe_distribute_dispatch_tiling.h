/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef ASCENDC_SHMEM_COMM_MOE_DISTRIBUTE_DISPATCH_TILING_H
#define ASCENDC_SHMEM_COMM_MOE_DISTRIBUTE_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ShmemMoeDistributeDispatchInfo {
    uint32_t epWorldSize;                // epWorldSize
    uint32_t tpWorldSize;                // tpWorldSize
    uint32_t epRankId;                   // epRankId
    uint32_t tpRankId;                   // tpRankId
    uint32_t expertSharedType;           // expert type
    uint32_t sharedExpertRankNum;        // shared expert number
    uint32_t moeExpertNum;               // moe expert number
    uint32_t quantMode;                  // quant mode
    uint32_t globalBs;                   // globalBs = BS * worldSize
    uint32_t bs;                         // bs
    uint32_t k;                          // k
    uint32_t h;                          // h
    uint32_t aivNum;                     // aivNum
    bool isQuant;                        // whether quant or not
    bool reserved1;                      // reserved
    bool reserved2;                      // reserved
    bool reserved3;                      // reserved
    uint64_t totalUbSize;                // epWorldSize
    uint64_t totalWinSize;
    uint32_t expertTokenNumsType;        // expert token nums type, support 0: cumsum mode, 1: count mode
    uint64_t shmemptr;                   // shmem ptr
};

struct ShmemMoeDistributeDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    Mc2CcTiling mc2CcTiling2;
    ShmemMoeDistributeDispatchInfo moeDistributeDispatchInfo;
};

#endif
