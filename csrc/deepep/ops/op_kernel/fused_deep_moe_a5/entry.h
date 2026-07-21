/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "fused_deep_moe_a5.h"
#include <kernel_operator.h>
#include "lib/matmul_intf.h"
#include "fused_deep_moe_a5_tiling.h"

using namespace Cam;

#define CALL_FUSED_DEEP_MOE                                                                                  \
    constexpr bool WEIGHT_NZ = (FORMAT_GMM1_WEIGHT == FORMAT_FRACTAL_NZ);                                    \
    FusedDeepMoe<DTYPE_X, DTYPE_GMM1_WEIGHT, WEIGHT_NZ, int32_t, false, TILING_KEY_VAR> op;                  \
    op.Init(x, expert_ids, gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale, expert_scales,    \
            share_gmm1_weight, share_gmm1_weight_scale, share_gmm2_weight, share_gmm2_weight_scale,          \
            expert_smooth_scales, share_smooth_scales, x_active_mask, output, share_output, expertTokenNums, \
            workspace, nullptr, &tiling_data);                                                               \
    op.Process()

#define CALL_FUSED_DEEP_MOE_IF_TILINGKEY(tilingKey) \
    if constexpr (TILING_KEY_IS(tilingKey)) {       \
        CALL_FUSED_DEEP_MOE;                        \
    }

#define CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(tilingKey) \
    else if constexpr (TILING_KEY_IS(tilingKey))      \
    {                                                 \
        CALL_FUSED_DEEP_MOE;                          \
    }

extern "C" __global__ __aicore__ void fused_deep_moe(GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_weight,
                                                     GM_ADDR gmm1_weight_scale, GM_ADDR gmm2_weight,
                                                     GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales,
                                                     GM_ADDR share_gmm1_weight, GM_ADDR share_gmm1_weight_scale,
                                                     GM_ADDR share_gmm2_weight, GM_ADDR share_gmm2_weight_scale,
                                                     GM_ADDR expert_smooth_scales, GM_ADDR share_smooth_scales,
                                                     GM_ADDR x_active_mask, GM_ADDR output, GM_ADDR share_output,
                                                     GM_ADDR expertTokenNums, GM_ADDR workspace, GM_ADDR tiling)
{
    icache_preload(8);
    REGISTER_TILING_DEFAULT(FusedDeepMoeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    CALL_FUSED_DEEP_MOE_IF_TILINGKEY(0)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(1)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(2)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(3)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(4)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(5)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(6)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(7)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(8)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(9)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(10)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(11)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(12)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(13)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(14)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(15)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(16)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(17)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(18)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(19)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(20)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(21)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(22)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(23)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(24)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(25)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(26)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(27)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(28)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(29)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(30)
    CALL_FUSED_DEEP_MOE_ELIF_TILINGKEY(31)
}
