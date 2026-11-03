/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "quant_lightning_indexer_tiling.h"
#include "arch35/quant_lightning_indexer_kernel_arch35.h"

using namespace QLIKernel;

extern "C" __global__ __aicore__ void sgl_quant_lightning_indexer_packed(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
    __gm__ uint8_t* queryScale, __gm__ uint8_t* actualSeqLengthsQ,
    __gm__ uint8_t* actualSeqLengthsK, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* sparseIndices, __gm__ uint8_t* workspace,
    uint32_t bSize, uint32_t tSize, uint32_t n2Size, uint32_t gSize,
    uint32_t s1Size, uint32_t s2Size, uint32_t sparseCount,
    uint32_t keyStride0, uint32_t keyDequantScaleStride0, uint32_t usedCoreNum,
    uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // Reconstruct the local record from the host-computed scalar launch arguments.
    QLITilingData tiling{};
    tiling.bSize = bSize;
    tiling.tSize = tSize;
    tiling.n2Size = n2Size;
    tiling.gSize = gSize;
    tiling.s1Size = s1Size;
    tiling.s2Size = s2Size;
    tiling.sparseCount = sparseCount;
    tiling.keyStride0 = keyStride0;
    tiling.keyDequantScaleStride0 = keyDequantScaleStride0;
    tiling.usedCoreNum = usedCoreNum;
    tiling.blockSize = blockSize;
    tiling.maxBlockNumPerBatch = maxBlockNumPerBatch;
    tiling.sparseMode = sparseMode;
    TPipe pipe;
    // One physical page contains FP8 keys followed by its FP32 scales.
    auto* keyScale = key + static_cast<uint64_t>(tiling.blockSize) * QLIPacked::HEAD_DIM;
    using PackedType = QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
                               LI_LAYOUT::TND, LI_LAYOUT::PA_BSND,
                               bfloat16_t, float32_t, float32_t, uint16_t>;
    SglQuantLightningIndexerPackedKernel<PackedType> op;
    op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ,
            actualSeqLengthsK, blocktable, sparseIndices, workspace, &tiling, &pipe);
    op.Process();
}
