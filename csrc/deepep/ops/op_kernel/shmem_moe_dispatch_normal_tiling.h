#ifndef SHMEM_MOE_DISPATCH_NORMAL_TILING_H
#define SHMEM_MOE_DISPATCH_NORMAL_TILING_H

struct ShmemMoeDispatchNormalInfo {
    uint32_t epWorldSize;   // epWorldSize
    uint32_t tpWorldSize;   // tpWorldSize
    uint32_t epRankId;      // epRankId
    uint32_t tpRankId;      // tpRankId
    uint32_t moeExpertNum;  // moe expert number
    uint32_t quantMode;     // quant mode
    uint32_t globalBs;      // globalBs = BS * worldSize
    uint32_t bs;            // bs
    uint32_t k;             // k
    uint32_t h;             // h
    uint32_t aivNum;        // aivNum
    bool isQuant;           // whether quant or not
    bool isEnableDiagnose;  // whether enable diagnose or not
    bool reserved2;         // reserved
    bool reserved3;         // reserved
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct ShmemMoeDispatchNormalTilingData {
    ShmemMoeDispatchNormalInfo moeDispatchNormalInfo;
    uint64_t shmemPtr;  // shmem symmetric point
};

#endif
