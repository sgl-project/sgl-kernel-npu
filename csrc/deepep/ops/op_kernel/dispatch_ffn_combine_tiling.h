/*!
 * \file dispatch_ffn_combine_tiling.h
 * \brief
 */

#include "dispatch_ffn_combine_kernel/moe_init_routing_quant_v2/moe_init_routing_v2_tiling.h"
#include "dispatch_ffn_combine_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

#ifndef ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
struct DispatchFFNCombineInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t isTransposeB;
    uint32_t isWeightNz;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t listLen;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t swizzleDirect = -1;
    int32_t swizzleOffset = -1;
    int32_t ubMoveNum = -1;
    int32_t pValue = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
    uint64_t initRoutingQuantTilingKey;
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
};

struct DispatchFFNCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    DispatchFFNCombineInfo dispatchFFNCombineInfo;
    CoCTiling cocTiling;
};
#endif
