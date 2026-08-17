/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_template_tiling_key.h"
#include "lightning_indexer_kernel.h"
#include "../op_host/tiling/mlp_lightning_tiling_data.h"

using namespace LIKernel;
using namespace LICommon;
using sglang::MlpLIHost::LITilingData;

extern "C" __global__ __aicore__ void mlp_lightning_indexer(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR curSeqLengthsQ, GM_ADDR curSeqLengths,
    GM_ADDR blocktable, GM_ADDR initNumTensor, GM_ADDR localNumTensor, GM_ADDR sparseIndices,
    GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
    return;
#else
    AscendC::TPipe tPipe;
    __gm__ uint8_t *userWorkspace = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    auto tilingData = reinterpret_cast<__gm__ LITilingData *>(tiling);
    auto tilingKey = tilingData->tilingKey;

    LIPreload<LIType<half, half, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> half_pa_bs_op;
    LIPreload<LIType<half, half, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>> half_pa_tnd_op;
    LIPreload<LIType<half, half, int32_t, false, LI_LAYOUT::BSND, LI_LAYOUT::BSND>> half_bs_op;
    LIPreload<LIType<half, half, int32_t, false, LI_LAYOUT::TND, LI_LAYOUT::TND>> half_tnd_op;
    LIPreload<LIType<bfloat16_t, bfloat16_t, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> bf16_pa_bs_op;
    LIPreload<LIType<bfloat16_t, bfloat16_t, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>> bf16_pa_tnd_op;
    LIPreload<LIType<bfloat16_t, bfloat16_t, int32_t, false, LI_LAYOUT::BSND, LI_LAYOUT::BSND>> bf16_bs_op;
    LIPreload<LIType<bfloat16_t, bfloat16_t, int32_t, false, LI_LAYOUT::TND, LI_LAYOUT::TND>> bf16_tnd_op;

    switch (tilingKey) {
        case GET_TPL_TILING_KEY(LI_TPL_FP16, LI_TPL_FP16, LI_TPL_INT32, 1, LI_LAYOUT_BSND, LI_LAYOUT_PA_BSND):
            half_pa_bs_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                               localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            half_pa_bs_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_FP16, LI_TPL_FP16, LI_TPL_INT32, 1, LI_LAYOUT_TND, LI_LAYOUT_PA_BSND):
            half_pa_tnd_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                                localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            half_pa_tnd_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_FP16, LI_TPL_FP16, LI_TPL_INT32, 0, LI_LAYOUT_BSND, LI_LAYOUT_BSND):
            half_bs_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                            localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            half_bs_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_FP16, LI_TPL_FP16, LI_TPL_INT32, 0, LI_LAYOUT_TND, LI_LAYOUT_TND):
            half_tnd_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                             localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            half_tnd_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_BF16, LI_TPL_BF16, LI_TPL_INT32, 1, LI_LAYOUT_BSND, LI_LAYOUT_PA_BSND):
            bf16_pa_bs_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                               localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            bf16_pa_bs_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_BF16, LI_TPL_BF16, LI_TPL_INT32, 1, LI_LAYOUT_TND, LI_LAYOUT_PA_BSND):
            bf16_pa_tnd_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                                localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            bf16_pa_tnd_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_BF16, LI_TPL_BF16, LI_TPL_INT32, 0, LI_LAYOUT_BSND, LI_LAYOUT_BSND):
            bf16_bs_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                            localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            bf16_bs_op.Process();
            break;
        case GET_TPL_TILING_KEY(LI_TPL_BF16, LI_TPL_BF16, LI_TPL_INT32, 0, LI_LAYOUT_TND, LI_LAYOUT_TND):
            bf16_tnd_op.Init(query, key, weights, curSeqLengthsQ, curSeqLengths, blocktable, initNumTensor,
                             localNumTensor, sparseIndices, sparseValues, userWorkspace, tilingData, &tPipe);
            bf16_tnd_op.Process();
            break;
        default:
            break;
    }
#endif
}
