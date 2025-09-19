#ifndef DISPATCH_LAYOUT_A2_TILING_H
#define DISPATCH_LAYOUT_A2_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct DispatchLayoutA2Info {
    uint32_t numTokens;
    uint32_t numRanks;
    uint32_t numExperts;
    uint32_t numTopk;
    uint32_t localRankSize;
    uint64_t totalUbSize;
};

struct DispatchLayoutA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    DispatchLayoutA2Info dispatchLayoutA2Info;
};

#endif
