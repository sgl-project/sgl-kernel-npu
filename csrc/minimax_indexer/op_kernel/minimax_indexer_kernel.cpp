/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

// /*!
//  * \file minimax_indexer_kernel.cpp
//  * \brief
//  */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "minimax_indexer_common.h"
#include "minimax_indexer_service_vector.h"
#include "minimax_indexer_service_cube.h"
#include "../op_host/tiling/minimax_tiling_data.h"

namespace sglang::npu_kernel::MIKernel {
using namespace MICommon;
using namespace MIServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// TempLoopInfo caches B/N/S1 info before the S2 loop starts (RunInfo not yet set); also avoids redundant computation
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1 loop end index
    uint32_t s2LoopEnd = 0U;    // S2 loop end index
    uint32_t actS1Size = 1ULL;  // actual S1 size for the current batch iteration
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // whether to clean output when actual S1 < shape S1
    uint32_t actMBaseSize = 0U;            // actual M-axis (gS1) size
    uint32_t mBasicSizeTail = 0U;          // gS1 tail base-block size
    uint32_t s2BasicSizeTail = 0U;         // S2 tail base-block size
};

template <typename MIT>
class MIPreload
{
public:
    __aicore__ inline MIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *reqToToken, __gm__ uint8_t *reqPoolIdx,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const __gm__ MIHost::MITilingData *tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // ================================= Type Definitions =================================
    using Q_T = typename MIT::queryType;
    using K_T = typename MIT::keyType;
    using OUT_T = typename MIT::outputType;
    static constexpr bool PAGE_ATTENTION = MIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = MIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = MIT::keyLayout;

    using MM1_OUT_T = float;

    MIMatmul<MIT> matmulService;
    MIVector<MIT> vectorService;

    // ================================= Constants =================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    // KV blocks per Cube<->Vector tile (s2BaseSize = S2_TILE_BLOCKS * block_size).
    static constexpr uint32_t S2_TILE_BLOCKS = 2;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================ Global Buffers =================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<K_T> weightsGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // Fused direct page lookup: logical->physical block ids are gathered from
    // req_to_token[req_pool_indices[b]][blk*blockSize] // blockSize in-kernel.
    GlobalTensor<int32_t> reqToTokenGm;
    GlobalTensor<int32_t> reqPoolIdxGm;

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;   // stores Q@K^T scores
    GlobalTensor<float> vec1ResGm;      // per-head topk16 values partials [aic,B,G,topk]
    GlobalTensor<int32_t> vec1ParamGm;  // per-head topk16 indices partials [aic,B,G,topk]

    // ================================ Class Members ====================================
    // AIC/AIV core info
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    MICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    MICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const __gm__ MIHost::MITilingData *tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, MICommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, MICommon::RunInfo &runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, MICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::InitTilingData(const __gm__ MIHost::MITilingData *tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    // Dedicated cross-engine flag IDs (disjoint from sparse_attention_score's 1/2/3).
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C1 = SYNC_V1_C1_FLAG;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.initBlocks = tilingData->initBlocks;
    constInfo.localBlocks = tilingData->localBlocks;
    constInfo.smScaleLog2e = tilingData->smScaleLog2e;
    constInfo.directMode = tilingData->directMode;
    constInfo.maxTokenSlots = tilingData->maxTokenSlots;
    constInfo.appendLocal = tilingData->appendLocal;
    constInfo.packedMode = tilingData->packedMode;
    constInfo.outputLayout = LAYOUT_T;  // output layout matches input
    if (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = tilingData->n2Size;  // support multi kv-head (was hardcoded 1)
    constInfo.headDim = HEAD_DIM;

    constInfo.mBaseSize = M_BASE_SIZE;
    // 2-block tile amortizes per-tile Cube overhead.
    constInfo.s2BaseSize = S2_TILE_BLOCKS * constInfo.kCacheBlockSize;
    constInfo.s1BaseSize = (constInfo.mBaseSize + constInfo.gSize - 1) / constInfo.gSize;
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                        __gm__ uint8_t *actualSeqLengths)
{
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        // packed mode carries [B*gqa] per-row lengths; reserve the full span so the
        // per-(b, g) scalar reads never step past the buffer.
        // packed mode: per-row [B*gqa] causal lengths.
        constInfo.actualLenDims = constInfo.packedMode ? constInfo.batchSize * constInfo.gSize : constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename MIT>
__aicore__ inline uint32_t MIPreload<MIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                           GlobalTensor<uint32_t> &actualSeqLengthsGm,
                                                           uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    if (constInfo.packedMode) {
        // Packed verify: use per-batch row-max as shared block space.
        uint32_t maxLen = 0;
        for (uint32_t g = 0; g < constInfo.gSize; g++) {
            uint32_t l = actualSeqLengthsGm.GetValue(bIdx * constInfo.gSize + g);
            maxLen = (l > maxLen) ? l : maxLen;
        }
        actS2Size = maxLen;
    } else {
        actS2Size = GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm,
                                    constInfo.kSeqSize);
    }
}

template <typename MIT>
__aicore__ inline uint32_t MIPreload<MIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                   uint32_t actS2Size)
{
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    uint32_t validS2LenBase = actS2Size - actS1Size;
    uint32_t validS2Len = s1Offset + validS2LenBase + constInfo.s1BaseSize;
    validS2Len = Min(validS2Len, actS2Size);
    validS2Len = Max(validS2Len, 0);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename MIT>
__aicore__ inline uint32_t MIPreload<MIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// Per-batch partition: each core owns whole batches, no cross-core WS merge.
template <typename MIT>
__aicore__ void inline MIPreload<MIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, MICommon::SplitCoreInfo &info)
{
    const uint32_t n = constInfo.batchSize * constInfo.kHeadNum;
    uint32_t base = (coreNum == 0) ? 0 : n / coreNum;
    uint32_t rem = (coreNum == 0) ? 0 : n % coreNum;
    uint32_t start = curCoreIdx * base + (curCoreIdx < rem ? curCoreIdx : rem);
    uint32_t cnt = base + (curCoreIdx < rem ? 1 : 0);
    info.bN2Start = start;
    info.bN2End = (cnt == 0) ? (start - 1) : (start + cnt - 1);
    info.gS1Start = 0;
    info.gS1End = 0;
    info.s2Start = 0;
    info.s2End = 0;
    info.isLD = true;
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize - 1);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount +  // T-axis, S1-axis offset
                    n2Idx * constInfo.sparseCount;                                  // N2-axis offset
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            // Fused output layout [QH, B, outW] (decode S1=1): clear every head row.
            const uint64_t outW = constInfo.sparseCount + constInfo.appendLocal;
            for (uint32_t g = 0; g < constInfo.gSize; g++) {
                uint64_t indiceOutOffset = g * constInfo.batchSize * outW + bIdx * outW;
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                            __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *reqToToken,
                                            __gm__ uint8_t *reqPoolIdx, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *workspace, const __gm__ MIHost::MITilingData *tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // compute core partitioning
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // WS layout: mm1Res[aic,2,mBase,s2Base] | vec1ResGm[aic,B,G,topk] | vec1ParamGm[aic,B,G,topk]
    uint64_t offset = 0;
    // usedCoreNum (not GetBlockNum) sizes per-core WS (AIV returns 2x).
    uint64_t singleCoreMm1ResSize = WS_DOBULE * constInfo.mBaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += usedCoreNum * singleCoreMm1ResSize;

    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += usedCoreNum * constInfo.batchSize * constInfo.gSize * constInfo.sparseCount * sizeof(float);
    vec1ParamGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset += usedCoreNum * constInfo.batchSize * constInfo.gSize * constInfo.sparseCount * sizeof(int32_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ K_T *)weights);
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, indiceOutGm,
                                           actualSeqLengthsGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
            reqToTokenGm.SetGlobalBuffer((__gm__ int32_t *)reqToToken);
            reqPoolIdxGm.SetGlobalBuffer((__gm__ int32_t *)reqPoolIdx);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, reqToTokenGm, reqPoolIdxGm, keyGm, queryGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    (void)((bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End));
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    // Per-batch: each core processes ALL blocks of its batches.
    tempLoopInfo.s2LoopEnd = s2BlockNum - 1;
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, MICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // compute actual base-block size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        MICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, MICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx - 1);
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset =
        keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::Process()
{
    if (usedCoreNum == 0) {
        // no compute tasks, clean output directly
        ProcessInvalid();
        return;
    }
    ProcessMain();
    ProcessDecode();
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            MICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize > totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // idle core returns immediately
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    // SyncAll<false>: sync BOTH AIC+AIV (default only syncs AIV).
    AscendC::SyncAll<false>();
    // Seed after SyncAll: setting it before SyncAll caused intermittent failures.
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<MICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<MICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }

    MICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        // Drain 2 seed credits.
        AscendC::CrossCoreWaitFlag<0, PIPE_S>(constInfo.syncV1C1);
        AscendC::CrossCoreWaitFlag<0, PIPE_S>(constInfo.syncV1C1);
    }
}
template <typename MIT>
__aicore__ inline void MIPreload<MIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, MICommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec(runInfo);
    }
}

template <typename MIT>
__aicore__ inline void MIPreload<MIT>::ProcessDecode()
{
    AscendC::SyncAll<false>();
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        vectorService.ProcessLD(splitCoreInfo.bN2Start, splitCoreInfo.bN2End);
    }
}
}  // namespace sglang::npu_kernel::MIKernel

__global__ __aicore__ void minimax_indexer(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actualSeqLengthsQ,
                                           GM_ADDR actualSeqLengths, GM_ADDR blocktable, GM_ADDR reqToToken,
                                           GM_ADDR reqPoolIdx, GM_ADDR sparseIndices, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    using namespace sglang::npu_kernel::MICommon;
    using namespace sglang::npu_kernel::MIKernel;

    __gm__ uint8_t *userWorkspace = workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    auto tilingData = reinterpret_cast<__gm__ sglang::MIHost::MITilingData *>(tiling);

    MIPreload<MIType<half, half, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>> half_pa_tnd_pabsnd_op;
    MIPreload<MIType<bfloat16_t, bfloat16_t, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>> bf16_pa_tnd_pabsnd_op;
    MIPreload<MIType<half, half, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>> half_pa_bsnd_pabsnd_op;
    MIPreload<MIType<bfloat16_t, bfloat16_t, int32_t, true, LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND>>
        bf16_pa_bsnd_pabsnd_op;

    auto tilingKey = tilingData->tilingKey;
    switch (tilingKey) {
        case 0x01013112:
            half_pa_tnd_pabsnd_op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, reqToToken,
                                       reqPoolIdx, sparseIndices, userWorkspace, tilingData, &tPipe);
            half_pa_tnd_pabsnd_op.Process();
            break;
        case 0x0c0c3112:
            bf16_pa_tnd_pabsnd_op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, reqToToken,
                                       reqPoolIdx, sparseIndices, userWorkspace, tilingData, &tPipe);
            bf16_pa_tnd_pabsnd_op.Process();
            break;
        case 0x01013102:
            half_pa_bsnd_pabsnd_op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable,
                                        reqToToken, reqPoolIdx, sparseIndices, userWorkspace, tilingData, &tPipe);
            half_pa_bsnd_pabsnd_op.Process();
            break;
        case 0x0c0c3102:
            bf16_pa_bsnd_pabsnd_op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable,
                                        reqToToken, reqPoolIdx, sparseIndices, userWorkspace, tilingData, &tPipe);
            bf16_pa_bsnd_pabsnd_op.Process();
            break;
    }
}
