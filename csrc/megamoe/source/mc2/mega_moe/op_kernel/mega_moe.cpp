/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "mega_moe.h"
#include "mega_moe_tiling.h"
#include "mega_moe_tiling_key.h"

using namespace AscendC;
using namespace MegaMoeImpl;

template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t PipelineMode>
__global__ __aicore__ void mega_moe(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                    GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR xActiveMask,
                                    GM_ADDR timerAddr, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
                                    GM_ADDR tilingGM)
{
    static_assert(PipelineMode <= MEGA_MOE_PIPELINE_RAGGED_N_STRIPES, "unsupported MegaMoe pipeline mode");
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);

    if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
        // Enumerate the core-wave modes explicitly: a mode outside this list
        // (sparse-decode today, any future addition) must not inherit the
        // core-wave epilogue/readiness protocol by falling through a
        // "not generic" test.
        constexpr bool UseCoreWavePipeline =
            PipelineMode == MEGA_MOE_PIPELINE_CORE_WAVE || PipelineMode == MEGA_MOE_PIPELINE_OWNER_CHAIN ||
            PipelineMode == MEGA_MOE_PIPELINE_OWNER_CHAIN_K16 || PipelineMode == MEGA_MOE_PIPELINE_RAGGED_N_STRIPES;
        MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DispatchQuantOutType, UseCoreWavePipeline,
                PipelineMode == MEGA_MOE_PIPELINE_OWNER_CHAIN || PipelineMode == MEGA_MOE_PIPELINE_OWNER_CHAIN_K16 ||
                    PipelineMode == MEGA_MOE_PIPELINE_RAGGED_N_STRIPES,
                PipelineMode == MEGA_MOE_PIPELINE_OWNER_CHAIN_K16 || PipelineMode == MEGA_MOE_PIPELINE_RAGGED_N_STRIPES,
                PipelineMode == MEGA_MOE_PIPELINE_SPARSE_DECODE, PipelineMode == MEGA_MOE_PIPELINE_RAGGED_N_STRIPES,
                false>
            op;
        op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                timerAddr, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
        op.Process();
    } else if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_NO_QUANT) {
        static_assert(PipelineMode == MEGA_MOE_PIPELINE_GENERIC,
                      "BF16 no-quant requires the complete-stage generic pipeline");
        MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DispatchQuantOutType, false, false, false, false, false, true> op;
        op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                timerAddr, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
        op.Process();
    }
}
