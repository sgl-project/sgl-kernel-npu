/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_a5.cpp
 * \brief Ascend 950 direct-launch kernel entry for the Compressor op.
 */

#ifndef EVENT_ID0
#define EVENT_ID0 0
#endif
#ifndef EVENT_ID1
#define EVENT_ID1 1
#endif
#ifndef EVENT_ID2
#define EVENT_ID2 2
#endif
#ifndef EVENT_ID3
#define EVENT_ID3 3
#endif
#ifndef EVENT_ID4
#define EVENT_ID4 4
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 5
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 6
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 7
#endif

#include "kernel_operator.h"

#if defined(__NPU_ARCH__)
#include "arch35/compressor_kernel.h"
#include "arch35/compressor_kernel_full_load.h"
#include "arch35/compressor_template_tiling_key.h"

using namespace Compressor;

#define INVOKE_A5_COMPRESSOR(templateClass, ...)                                                                   \
    do {                                                                                                           \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData);                                                \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed, \
                startPos, cmpKvOut, workspace);                                                                    \
        op.Process();                                                                                              \
    } while (0)

#define A5_COMPRESSOR_CASE(templateId, templateClass, layout, dtype, coff, cache)                                   \
    case GET_TPL_TILING_KEY(layout, dtype, coff, 2, cache, templateId):                                             \
        INVOKE_A5_COMPRESSOR(templateClass, static_cast<X_LAYOUT>(layout), static_cast<X_DTYPE>(dtype),             \
                             static_cast<COFF>(coff), static_cast<ROTARY_MODE>(2), static_cast<CACHE_MODE>(cache)); \
        break;
__global__ __aicore__ void compressor(
    GM_ADDR x, GM_ADDR wKv, GM_ADDR wGate, GM_ADDR stateCache, GM_ADDR ape, GM_ADDR normWeight, GM_ADDR ropeSin,
    GM_ADDR ropeCos, GM_ADDR stateBlockTable, GM_ADDR cuSeqlens, GM_ADDR seqUsed, GM_ADDR startPos, GM_ADDR cmpKvOut,
    GM_ADDR stateCacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    auto tilingDataGm = reinterpret_cast<__gm__ optiling::CompressorTilingData *>(tiling);
    optiling::CompressorTilingData tilingDataIn;
    tilingDataIn.baseParams.batchSize = tilingDataGm->baseParams.batchSize;
    tilingDataIn.baseParams.seqSize = tilingDataGm->baseParams.seqSize;
    tilingDataIn.baseParams.hiddenSize = tilingDataGm->baseParams.hiddenSize;
    tilingDataIn.baseParams.tokenSize = tilingDataGm->baseParams.tokenSize;
    tilingDataIn.baseParams.headDim = tilingDataGm->baseParams.headDim;
    tilingDataIn.baseParams.ropeHeadDim = tilingDataGm->baseParams.ropeHeadDim;
    tilingDataIn.baseParams.csSize = tilingDataGm->baseParams.csSize;
    tilingDataIn.baseParams.cmpRatio = tilingDataGm->baseParams.cmpRatio;
    tilingDataIn.baseParams.cgSize = tilingDataGm->baseParams.cgSize;
    tilingDataIn.baseParams.normEps = tilingDataGm->baseParams.normEps;
    tilingDataIn.baseParams.reciprocalD = tilingDataGm->baseParams.reciprocalD;
    tilingDataIn.baseParams.usedCoreNum = tilingDataGm->baseParams.usedCoreNum;
    tilingDataIn.baseParams.nSize = tilingDataGm->baseParams.nSize;
    tilingDataIn.baseParams.stateCacheStrideDim0 = tilingDataGm->baseParams.stateCacheStrideDim0;
    tilingDataIn.baseParams.kBaseNum = tilingDataGm->baseParams.kBaseNum;
    tilingDataIn.baseParams.kBaseSize = tilingDataGm->baseParams.kBaseSize;
    tilingDataIn.baseParams.coreGroupNum = tilingDataGm->baseParams.coreGroupNum;
    tilingDataIn.baseParams.mLoopNum = tilingDataGm->baseParams.mLoopNum;
    for (uint32_t i = 0; i < CMP_MAX_AIC_CORE_NUM; ++i) {
        tilingDataIn.baseParams.splitCoreParam[i].mStart = tilingDataGm->baseParams.splitCoreParam[i].mStart;
        tilingDataIn.baseParams.splitCoreParam[i].mEnd = tilingDataGm->baseParams.splitCoreParam[i].mEnd;
        tilingDataIn.baseParams.splitCoreParam[i].nStart = tilingDataGm->baseParams.splitCoreParam[i].nStart;
        tilingDataIn.baseParams.splitCoreParam[i].nEnd = tilingDataGm->baseParams.splitCoreParam[i].nEnd;
        tilingDataIn.baseParams.splitCoreParam[i].kStart = tilingDataGm->baseParams.splitCoreParam[i].kStart;
        tilingDataIn.baseParams.splitCoreParam[i].kEnd = tilingDataGm->baseParams.splitCoreParam[i].kEnd;
    }
    tilingDataIn.pageAttentionParams.blockNum = tilingDataGm->pageAttentionParams.blockNum;
    tilingDataIn.pageAttentionParams.blockSize = tilingDataGm->pageAttentionParams.blockSize;
    tilingDataIn.pageAttentionParams.maxBlockNumPerBatch = tilingDataGm->pageAttentionParams.maxBlockNumPerBatch;
    tilingDataIn.innerSplitParams.mBaseSize = tilingDataGm->innerSplitParams.mBaseSize;
    tilingDataIn.innerSplitParams.dBaseSize = tilingDataGm->innerSplitParams.dBaseSize;
    tilingDataIn.workspaceParams.mm1KvResSize = tilingDataGm->workspaceParams.mm1KvResSize;
    tilingDataIn.workspaceParams.mm1ScoreResSize = tilingDataGm->workspaceParams.mm1ScoreResSize;
    tilingDataIn.workspaceParams.vec1ResSize = tilingDataGm->workspaceParams.vec1ResSize;
    tilingDataIn.workspaceParams.vec1TailCacheSize = tilingDataGm->workspaceParams.vec1TailCacheSize;
    tilingDataIn.workspaceParams.dbWorkspaceRatio = tilingDataGm->workspaceParams.dbWorkspaceRatio;
    tilingDataIn.tilingKey = tilingDataGm->tilingKey;
    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    uint64_t key = tilingData->tilingKey;
#if defined(COMPRESSOR_A5_DEBUG)
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::printf("[compressor][device-debug] gm_key=%llu local_key=%llu block=%u template=%llu\n",
                        static_cast<unsigned long long>(tilingDataGm->tilingKey),
                        static_cast<unsigned long long>(key), AscendC::GetBlockIdx(),
                        static_cast<unsigned long long>((key >> 11) & 0x3));
    }
#endif
    uint64_t templateId = (key >> 11) & 0x3;
    if (templateId == static_cast<uint64_t>(TEMPLATE_ID::EMPTY_X)) {
        return;
    }

    switch (key) {
        // NORMAL: BSH layout.
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 0, 1, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 0, 1, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 0, 2, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 0, 2, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 1, 1, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 1, 1, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 1, 2, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 0, 1, 2, 2)
        // NORMAL: TH layout.
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 0, 1, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 0, 1, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 0, 2, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 0, 2, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 1, 1, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 1, 1, 2)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 1, 2, 1)
        A5_COMPRESSOR_CASE(0, CompressorKernel, 1, 1, 2, 2)
        // FULL_LOAD: BSH layout only.
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 0, 1, 1)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 0, 1, 2)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 0, 2, 1)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 0, 2, 2)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 1, 1, 1)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 1, 1, 2)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 1, 2, 1)
        A5_COMPRESSOR_CASE(2, CompressorKernelFullLoad, 0, 1, 2, 2)
        default:
            break;
    }
}

#undef A5_COMPRESSOR_CASE
#undef INVOKE_A5_COMPRESSOR
#endif
