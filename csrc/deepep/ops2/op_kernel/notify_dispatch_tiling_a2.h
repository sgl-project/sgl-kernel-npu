#ifndef NOTIFY_DISPATCH_TILING_A2_H
#define NOTIFY_DISPATCH_TILING_A2_H

#include "kernel_tiling/kernel_tiling.h"

struct NotifyDispatchInfoA2 {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankSize;
    uint32_t localRankId;
    uint32_t sendCount;
    uint32_t numTokens;
    uint32_t topkNum;
    uint32_t numExperts;
    uint32_t aivNum;
    uint64_t totalUbSize;
};

struct NotifyDispatchA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    NotifyDispatchInfoA2 notifyDispatchInfoA2;
};

#endif
