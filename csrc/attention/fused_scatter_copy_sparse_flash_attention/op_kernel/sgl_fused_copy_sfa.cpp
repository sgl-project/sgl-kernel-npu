// Copyright (c) 2026 Huawei Technologies Co., Ltd.
#include "cann_copy_sfa_tiling_data.h"
#include "fused_copy_sfa_impl.h"

extern "C" __global__ __aicore__ void sgl_fused_copy_sfa(
    GM_ADDR query, GM_ADDR queryRope, GM_ADDR hbmCkv, GM_ADDR hbmRope, GM_ADDR queryEnds, GM_ADDR kvLens,
    GM_ADDR cacheTokens, GM_ADDR topkSrc, GM_ADDR topkDst, GM_ADDR topkMiss, GM_ADDR missSrc, GM_ADDR missDst,
    GM_ADDR misses, GM_ADDR hbmTable, GM_ADDR dramTable, GM_ADDR current, GM_ADDR positions, GM_ADDR valid,
    GM_ADDR requestSlots, GM_ADDR reqToToken, GM_ADDR residentEnd, GM_ADDR attentionOut, GM_ADDR hbmCkvOut,
    GM_ADDR hbmRopeOut, GM_ADDR residentEndOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(CannCopySfaTiling);
    const auto *data = reinterpret_cast<const __gm__ CannCopySfaTiling *>(tiling);
    auto dramCkv = reinterpret_cast<GM_ADDR>(data->dramCkvAddress);
    auto dramRope = reinterpret_cast<GM_ADDR>(data->dramRopeAddress);
    if (data->combinedHbm) {
        RunCopySfa<true>(query, hbmCkvOut, hbmCkvOut, topkDst, cacheTokens, hbmTable, queryEnds, kvLens, queryRope,
                         hbmCkvOut + 512 * 2, dramRope, dramCkv, dramTable, topkSrc, topkMiss, missSrc, missDst, misses,
                         attentionOut, current, positions, valid, requestSlots, reqToToken, residentEndOut, workspace,
                         tiling);
    } else {
        RunCopySfa<false>(query, hbmCkvOut, hbmCkvOut, topkDst, cacheTokens, hbmTable, queryEnds, kvLens, queryRope,
                          hbmRopeOut, dramRope, dramCkv, dramTable, topkSrc, topkMiss, missSrc, missDst, misses,
                          attentionOut, current, positions, valid, requestSlots, reqToToken, residentEndOut, workspace,
                          tiling);
    }
}
