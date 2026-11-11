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
 * \file compressor_kernel_perf.h
 * \brief
 */

#ifndef COMPRESSOR_KERNEL_PERF_H
#define COMPRESSOR_KERNEL_PERF_H

#include "compressor_comm.h"
#include "compressor_template_tiling_key.h"
#include "compressor_tiling_data.h"
#include "compressor_tools.h"
#include "compressor_block_cube_perf.h"
#include "compressor_block_vec_perf.h"

using namespace AscendC;

namespace Compressor {

struct CmpBlockInfo {
    __aicore__ inline CmpBlockInfo(){};
    __aicore__ inline CmpBlockInfo(uint32_t bIdx, uint32_t sIdx, bool needReset = false)
        : bIdx(bIdx), sIdx(sIdx), needReset(needReset){};

    uint32_t bIdx = 0U;
    uint32_t sIdx = 0U;
    uint32_t bSeqUsed = 0U;
    uint32_t bStartPos = 0U;
    bool needReset = false;
    bool isFirst = true;

    uint32_t headSeqCnt = 0U;
    uint32_t validSeqCnt = 0U;
    uint32_t tailSeqCnt = 0U;
    bool isCompress = 0U;
};

struct BasicBlockInfo {
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
    uint32_t compressedTcNum = 0;
    uint32_t dealSeqCnt = 0;
    uint32_t dealTcNum = 0;
};

struct BatchInfo {
    uint32_t tcNum = 0;
    uint32_t compressedTcNum = 0;
    uint32_t remSeqCnt = 0;
    uint32_t seqCnt = 0;
    uint32_t seqUsedCnt = 0;
    uint32_t headHolderSeq = 0;
    uint32_t bStartPos = 0;
    uint32_t bIdx = 0;
    uint32_t sIdx = 0;
};

template <typename COMP>
class CompressorKernelPerf
{
public:
    __aicore__ inline CompressorKernelPerf(TPipe *pipe,
                                           const __gm__ optiling::CompressorTilingData *__restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                __gm__ uint8_t *stateCache, __gm__ uint8_t *ape, __gm__ uint8_t *normWeight,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                __gm__ uint8_t *cmpKvOut, __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

private:
    // ================================Init functions==================================
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    // ================================Process functions================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void SetBaseSize();
    // get the number of base blocks
    __aicore__ inline uint32_t GetLoopTimes();
    __aicore__ inline void SkipInvalidBatch(BatchInfo &batchInfo);
    __aicore__ inline void UpdateCurGroup(BasicBlockInfo &basicBlockInfo, BatchInfo batchInfo, uint32_t &curGroupQuota,
                                          uint32_t curDealSeq);
    __aicore__ inline BasicBlockInfo SkipOneLoop(BatchInfo &batchInfo);
    // compute basic core-split information
    __aicore__ inline void CalcSplitCoreInfo();

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info, bool isNeedExcute);
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline void ComputeVec2(const Vec2RunInfo &info);

    __aicore__ inline bool IsNeedExcuteC1(RunInfo info);
    __aicore__ inline bool IsNeedSyncAll(uint32_t curBasicBlockIdx);
    __aicore__ inline void CalcC1V1Params(RunInfo &info, Vec1RunInfo &vec1Info, BatchInfo &batchInfo, uint32_t loopIdx);
    __aicore__ inline void UpdateVec2Info(Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx, const Vec1RunInfo &info);
    __aicore__ inline bool IsNeedExcuteV2(Vec2RunInfo &vec2Info);

    using X_T = typename AscendC::Conditional<COMP::xDtype == X_DTYPE::BF16, bfloat16_t, half>::type;
    using T = float;
    using MM1_OUT_T = T;
    using VEC1_OUT_T = T;

    // constants
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint32_t SYNC_C1_FLAG = 3;
    static constexpr uint32_t SYNC_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_FLAG2 = 5;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 6;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 8;

    // ==============================TilingData&TPipe==============================
    TPipe *pipe_;
    const __gm__ optiling::CompressorTilingData *__restrict tilingData_;
    // ===========================Workspace Global Tensor===========================
    GlobalTensor<MM1_OUT_T> mm1KvResGm;
    GlobalTensor<MM1_OUT_T> mm1ScoreResGm;
    GlobalTensor<MM1_OUT_T> vec1KvCacheGm;
    GlobalTensor<MM1_OUT_T> vec1ScoreCacheGm;
    GlobalTensor<MM1_OUT_T> Vec1InputKvGm;
    GlobalTensor<MM1_OUT_T> Vec1InputScoreGm;
    GlobalTensor<VEC1_OUT_T> vec1ResGm;
    GlobalTensor<VEC1_OUT_T> vec2InputGm;
    // ================================Task Info====================================
    CompressorTools<COMP> tools_;
    ConstInfo constInfo{};
    uint32_t aiCoreIdx = 0;

    // ==============================Service Define==============================
    CompressorBlockCubePerf<COMP> blockCube_;
    CompressorBlockVectorPerf<COMP> blockVec_;

    uint32_t allCompressedTcNum_ = 0;
    uint32_t curCompressedTcNum_ = 0;
    uint32_t accDealSize = 0;
    uint32_t loopTimes = 0;
    uint32_t cubeLoop = 0;
    uint32_t vec1Loop = 0;
    uint32_t vec2Loop = 0;
    bool isFirstUpdateCurGroup = true;
};

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                                        __gm__ uint8_t *stateCache, __gm__ uint8_t *ape,
                                                        __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin,
                                                        __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                                        __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed,
                                                        __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut,
                                                        __gm__ uint8_t *workspace)
{
    if ASCEND_IS_AIV {
        constInfo.aiCoreIdx = GetBlockIdx() / 2;
    } else {
        constInfo.aiCoreIdx = GetBlockIdx();
    }
    InitTilingData();
    // init tools
    tools_.toolParams_.seqSize = tilingData_->baseParams.seqSize;
    tools_.toolParams_.cmpRatio = tilingData_->baseParams.cmpRatio;
    tools_.Init(startPos, seqUsed, cuSeqlens);

    // remove invalid batches at the tail
    for (; constInfo.batchSize > 0; --constInfo.batchSize) {
        uint32_t bSeqUsed = tools_.GetSeqLength(constInfo.batchSize - 1);
        if (bSeqUsed > 0) {
            break;
        }
    }

    // when the valid sequences of all batches are 0, exit directly
    if (constInfo.batchSize == 0) {
        return;
    }

    // 0. Compute the start position of the last Tc block
    constInfo.bIdxOfLastTc = constInfo.batchSize - 1;
    // 1. Compute the head_dim split size and build the rest of ConstInfo
    SetBaseSize();  // set the base block size
    CalcSplitCoreInfo();
    // 2. Compute the number of loop iterations
    loopTimes = GetLoopTimes();
    // 3. Initialize workspace
    InitWorkspace(workspace);
    // 4. Initialize the block layer
    if ASCEND_IS_AIC {
#if __CCE_AICORE__ == 310
        blockCube_.InitParams(constInfo, tools_);
#else
        blockCube_.InitParams(constInfo, tools_);
#endif
        blockCube_.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens,
                        seqUsed, startPos, cmpKvOut);
        blockCube_.InitBuffers(pipe_);
#if __CCE_AICORE__ == 310
        blockCube_.InitGlobalBuffers(mm1KvResGm, mm1ScoreResGm);
#else
        blockCube_.InitGlobalBuffers(mm1KvResGm, mm1ScoreResGm);
#endif
    } else {
        blockVec_.InitParams(constInfo, tools_);
        blockVec_.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens,
                       seqUsed, startPos, cmpKvOut);
        blockVec_.InitBuffers(pipe_);
#if __CCE_AICORE__ == 310
        blockVec_.InitVec1GlobalTensor(Vec1InputKvGm, Vec1InputScoreGm, vec1KvCacheGm, vec1ScoreCacheGm, vec1ResGm,
                                       vec2InputGm);
#else
        blockVec_.InitVec1GlobalTensor(Vec1InputKvGm, Vec1InputScoreGm, vec1KvCacheGm, vec1ScoreCacheGm, vec1ResGm,
                                       vec2InputGm);
#endif
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::InitTilingData()
{
    constInfo.cmpRatio = tilingData_->baseParams.cmpRatio;
    constInfo.batchSize = tilingData_->baseParams.batchSize;
    constInfo.mBaseSize = tilingData_->innerSplitParams.mBaseSize;
    constInfo.headDim = tilingData_->baseParams.headDim;
    constInfo.hSize = tilingData_->baseParams.hiddenSize;
    constInfo.sSize = tilingData_->baseParams.seqSize;
    constInfo.ropeHeadDim = tilingData_->baseParams.ropeHeadDim;
    constInfo.normEps = tilingData_->baseParams.normEps;
    constInfo.reciprocalD = tilingData_->baseParams.reciprocalD;
    constInfo.usedCoreNum = tilingData_->baseParams.usedCoreNum;

    constInfo.blockNum = tilingData_->pageAttentionParams.blockNum;
    constInfo.blockSize = tilingData_->pageAttentionParams.blockSize;
    constInfo.maxBlockNumPerBatch = tilingData_->pageAttentionParams.maxBlockNumPerBatch;

    constInfo.nSize = tilingData_->baseParams.nSize;
    constInfo.vec1TailCacheSize = tilingData_->workspaceParams.vec1TailCacheSize;
    constInfo.dbWorkspaceRatio = tilingData_->workspaceParams.dbWorkspaceRatio;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::SetBaseSize()
{
    uint32_t mSize = 0;
    uint32_t minMBaseSize = 0;
    bool sameSeqUsed = true;
    uint32_t firstBatchSeqUsed = tools_.GetSeqLength(0);
    for (uint32_t i = 0; i < constInfo.batchSize; i++) {
        uint32_t bSeqUsed = tools_.GetSeqLength(i);
        uint32_t bStartPos = tools_.GetStartPos(i);
        // get the size of m
        mSize += bSeqUsed;
        // get whether lengths are equal
        if (sameSeqUsed && (bSeqUsed != firstBatchSeqUsed)) {
            sameSeqUsed = false;
        }
        // get the minimum split size of the m axis
        if (minMBaseSize != constInfo.cmpRatio) {
            uint32_t startCmpIdx = bStartPos / constInfo.cmpRatio;
            uint32_t endCmpIdx = (bStartPos + bSeqUsed) / constInfo.cmpRatio;
            if (startCmpIdx == endCmpIdx) {
                if (bSeqUsed > minMBaseSize) {
                    minMBaseSize = bSeqUsed;
                }
            } else if (startCmpIdx + 1 == endCmpIdx) {
                uint32_t startCmpValidSeqCnt = constInfo.cmpRatio - (bStartPos % constInfo.cmpRatio);
                uint32_t endCmpValidSeqCnt = (bStartPos + bSeqUsed) % constInfo.cmpRatio;
                if (startCmpValidSeqCnt > minMBaseSize) {
                    minMBaseSize = startCmpValidSeqCnt;
                }
                if (endCmpValidSeqCnt > minMBaseSize) {
                    minMBaseSize = endCmpValidSeqCnt;
                }
            } else {
                minMBaseSize = constInfo.cmpRatio;
            }
        }
    }

    uint32_t aiCoreNum = constInfo.usedCoreNum;
    constInfo.dBaseSize = 64;
    uint32_t dBaseBlockNum = constInfo.headDim / constInfo.dBaseSize;
    if (sameSeqUsed && mSize <= (constInfo.mBaseSize * (aiCoreNum / dBaseBlockNum))) {
        if constexpr (COMP::coff == COFF::OVERLAP) {
            if (constInfo.headDim == 128) {
                dBaseBlockNum = 8;
            } else if (constInfo.headDim == 512) {
                dBaseBlockNum = 16;
            }
        } else {
            if (constInfo.headDim == 128) {
                dBaseBlockNum = 8;
            } else if (constInfo.headDim == 512) {
                dBaseBlockNum = 16;
            }
        }
        // the modification takes effect only when there are enough cores
        if (aiCoreNum >= dBaseBlockNum) {
            constInfo.dBaseSize = constInfo.headDim / dBaseBlockNum;
            // enable all cores
            uint32_t coreGroupNum = aiCoreNum / dBaseBlockNum;
            uint32_t newMBaseSize = (constInfo.batchSize + coreGroupNum - 1) / coreGroupNum * firstBatchSeqUsed;
            if (newMBaseSize > minMBaseSize && newMBaseSize < constInfo.mBaseSize) {
                constInfo.mBaseSize = newMBaseSize;
            }
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::SkipInvalidBatch(BatchInfo &batchInfo)
{
    for (; batchInfo.bIdx < constInfo.batchSize; ++batchInfo.bIdx) {
        batchInfo.seqCnt = tools_.GetSeqLength(batchInfo.bIdx);
        if (batchInfo.seqCnt > 0) {
            break;
        }
    }
    batchInfo.remSeqCnt = batchInfo.seqCnt;
    if (tools_.isExistSeqUsed_) {
        batchInfo.seqUsedCnt = tools_.GetSeqUsed(batchInfo.bIdx);
    } else {
        batchInfo.seqUsedCnt = batchInfo.seqCnt;
    }
    if (batchInfo.bIdx < constInfo.batchSize) {
        batchInfo.bStartPos = tools_.GetStartPos(batchInfo.bIdx);
        batchInfo.sIdx = 0;
        batchInfo.headHolderSeq = batchInfo.bStartPos & (constInfo.cmpRatio - 1);
        batchInfo.tcNum = (batchInfo.bStartPos + batchInfo.seqCnt + constInfo.cmpRatio - 1) / constInfo.cmpRatio -
                          batchInfo.bStartPos / constInfo.cmpRatio;
        batchInfo.compressedTcNum = (batchInfo.bStartPos + batchInfo.seqUsedCnt) / constInfo.cmpRatio -
                                    batchInfo.bStartPos / constInfo.cmpRatio;
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::UpdateCurGroup(BasicBlockInfo &basicBlockInfo, BatchInfo batchInfo,
                                                                  uint32_t &curGroupQuota, uint32_t curDealSeq)
{
    // update information of the current group
    if (curGroupQuota == 0 && !isFirstUpdateCurGroup) {
        return;
    }
    isFirstUpdateCurGroup = false;
    basicBlockInfo.bIdx = batchInfo.bIdx;
    uint32_t curGroupDealSeq = curGroupQuota < curDealSeq ? curGroupQuota : curDealSeq;
    basicBlockInfo.sIdx = batchInfo.sIdx + curGroupDealSeq;
    basicBlockInfo.dealSeqCnt += curGroupDealSeq;
    curGroupQuota -= curGroupDealSeq;
    // the end needs to jump batch; must consider that the current group start is at the end, or the current group start
    // is greater than the entire M axis
    if ((curGroupQuota == 0 || basicBlockInfo.bIdx == constInfo.batchSize - 1) &&
        basicBlockInfo.sIdx == batchInfo.seqCnt) {
        basicBlockInfo.sIdx = 0;
        for (basicBlockInfo.bIdx++; basicBlockInfo.bIdx < constInfo.batchSize; ++basicBlockInfo.bIdx) {
            uint32_t seqCnt = tools_.GetSeqLength(basicBlockInfo.bIdx);
            if (seqCnt > 0) {
                break;
            }
        }
    }
}

template <typename COMP>
__aicore__ inline BasicBlockInfo CompressorKernelPerf<COMP>::SkipOneLoop(BatchInfo &batchInfo)
{
    BasicBlockInfo basicBlockInfo{};
    isFirstUpdateCurGroup = true;
    uint32_t curGroupQuota = constInfo.mBaseSize * constInfo.curGroupIdx;  // m-axis start of the current group
    bool curGroupStartFlag = false;
    uint32_t quota = constInfo.coreGroupNum * constInfo.mBaseSize;

    for (; batchInfo.bIdx < constInfo.batchSize;) {
        uint32_t curDealSeq = 0;
        uint32_t curDealTcNum = 0;
        uint32_t curDealCompressedTcNum = 0;
        // cannot finish processing the entire current batch
        if (quota < batchInfo.remSeqCnt) {
            // align r downward,
            if (quota > constInfo.cmpRatio - batchInfo.headHolderSeq) {
                uint32_t delta = (batchInfo.bStartPos + batchInfo.sIdx + quota) &
                                 (constInfo.cmpRatio - 1);  // the part exceeding the alignment
                curDealSeq = quota - delta;
                quota -= curDealSeq;
                curDealTcNum = (curDealSeq + constInfo.cmpRatio - 1) / constInfo.cmpRatio;
                curDealCompressedTcNum = min(curDealTcNum, batchInfo.compressedTcNum);
                // update information needed by the current group
                UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
                // update batch information
                batchInfo.remSeqCnt = batchInfo.remSeqCnt - curDealSeq;
                batchInfo.sIdx = batchInfo.sIdx + curDealSeq;
                batchInfo.compressedTcNum -= curDealCompressedTcNum;
                batchInfo.tcNum -= curDealTcNum;
                // update loop information
                basicBlockInfo.dealTcNum += curDealTcNum;
                basicBlockInfo.compressedTcNum += curDealCompressedTcNum;
            }
            break;
        } else {
            // process the entire batch
            quota -= batchInfo.remSeqCnt;
            curDealSeq = batchInfo.remSeqCnt;
            curDealTcNum = batchInfo.tcNum;
            // update information needed by the current group
            UpdateCurGroup(basicBlockInfo, batchInfo, curGroupQuota, curDealSeq);
            // update batch and loop information
            batchInfo.remSeqCnt = 0;
            basicBlockInfo.dealTcNum += batchInfo.tcNum;
            basicBlockInfo.compressedTcNum += batchInfo.compressedTcNum;
            batchInfo.bIdx++;
            SkipInvalidBatch(batchInfo);
        }
    }
    uint32_t totalDataSize = constInfo.coreGroupNum * constInfo.mBaseSize - quota;
    // 2. Start offset of the current group
    uint32_t currentGroupStart = constInfo.curGroupIdx * constInfo.mBaseSize;

    // 3. Safety check
    if (currentGroupStart >= totalDataSize) {
        // exceeds the tail block
        basicBlockInfo.dealSeqCnt = 0;
    } else {
        // still within the valid range; compute the remaining amount
        uint32_t remaining = totalDataSize - currentGroupStart;
        basicBlockInfo.dealSeqCnt = (remaining < constInfo.mBaseSize) ? remaining : constInfo.mBaseSize;
    }

    return basicBlockInfo;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorKernelPerf<COMP>::GetLoopTimes()
{
    // compute the number of main loop iterations
    uint32_t loopTimes = 0;
    BatchInfo batchInfo{};
    SkipInvalidBatch(batchInfo);
    for (; batchInfo.bIdx < constInfo.batchSize; ++loopTimes) {
        SkipOneLoop(batchInfo);
    }
    return loopTimes;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcSplitCoreInfo()
{
    // number of base blocks in the D direction
    constInfo.dBasicBlockNum = constInfo.headDim / constInfo.dBaseSize;
    // number of core groups
    constInfo.coreGroupNum = constInfo.usedCoreNum / constInfo.dBasicBlockNum;
    // index in the d direction processed by each core
    constInfo.dIdx = (constInfo.aiCoreIdx % constInfo.dBasicBlockNum) * constInfo.dBaseSize;
    // current group id
    constInfo.curGroupIdx = constInfo.aiCoreIdx / constInfo.dBasicBlockNum;

    constInfo.mm1ResSize = constInfo.mBaseSize * constInfo.headDim * constInfo.coreGroupNum;

    uint32_t coff = (uint32_t)COMP::coff;
    constInfo.mm1KvResSize = constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.mm1ScoreResSize = constInfo.mBaseSize * constInfo.headDim * coff;
    constInfo.vec1ResSize = constInfo.mBaseSize * constInfo.headDim * constInfo.nSize;

    constInfo.dbSize = constInfo.coreGroupNum * constInfo.mm1KvResSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::InitWorkspace(__gm__ uint8_t *workspace)
{
    uint64_t offset = 0;
    uint64_t mm1KvResStartOffset = offset;
    // mm1KvResGm
    mm1KvResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset + constInfo.curGroupIdx * constInfo.mm1KvResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1KvResSize * sizeof(MM1_OUT_T);

    uint64_t mm1ScoreResStartOffset = offset;
    // mm1ScoreResGm
    mm1ScoreResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             constInfo.curGroupIdx * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T)));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.mm1ScoreResSize * sizeof(MM1_OUT_T);

    Vec1InputKvGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1KvResStartOffset));

    Vec1InputScoreGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + mm1ScoreResStartOffset));

    vec1KvCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    vec1ScoreCacheGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.vec1TailCacheSize * sizeof(MM1_OUT_T);

    uint64_t beforeVecOffset = offset;

    // vec1Res
    vec1ResGm.SetGlobalBuffer((__gm__ VEC1_OUT_T *)(workspace + offset));
    offset += constInfo.dbWorkspaceRatio * constInfo.coreGroupNum * constInfo.vec1ResSize * sizeof(VEC1_OUT_T);
    // vec2Input
    vec2InputGm.SetGlobalBuffer((__gm__ VEC1_OUT_T *)(workspace + beforeVecOffset));
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeMm1(const RunInfo &info, bool isNeedExcute)
{
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG + info.cubeDbIdx);
    if (isNeedExcute) {
        blockCube_.ComputeMm1(info);
    }
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_FIX>(SYNC_C1_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG + info.cubeDbIdx);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_C1_V1_FLAG + info.c1v1DbIdx);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG2 + info.c1v1DbIdx);
    blockVec_.ComputeVec1(info);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG);
    if constexpr (COMP::cacheMode == CACHE_MODE::EXPLICIT && COMP::coff == COFF::DISABLE) {
        SyncAll();
        blockVec_.CommitC128State(info);
    }
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG + info.c1v1DbIdx);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_V1_FLAG2 + (info.c1v1DbIdx + 1) % constInfo.dbWorkspaceRatio);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::ComputeVec2(const Vec2RunInfo &info)
{
    blockVec_.ComputeVec2(info);
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::AllocEventID()
{
    if ASCEND_IS_AIC {
        blockCube_.AllocEventID(pipe_);
    } else {
        blockVec_.AllocEventID();
        for (int i = 0; i < constInfo.dbWorkspaceRatio; ++i) {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_V1_C1_FLAG + i);
        }
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_V1_FLAG2);
    }
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::FreeEventID()
{
    if ASCEND_IS_AIC {
        for (int i = 0; i < constInfo.dbWorkspaceRatio; ++i) {
            CrossCoreWaitFlag<SYNC_MODE2, PIPE_FIX>(SYNC_V1_C1_FLAG + i);
        }
        blockCube_.FreeEventID(pipe_);
    } else {
        CrossCoreWaitFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_V1_FLAG2 + loopTimes % constInfo.dbWorkspaceRatio);
        blockVec_.FreeEventID();
    }
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedExcuteC1(RunInfo info)
{
    // If B is out of range, cube does not execute
    return info.bStart < constInfo.batchSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::CalcC1V1Params(RunInfo &info, Vec1RunInfo &vec1Info,
                                                                  BatchInfo &batchInfo, uint32_t loopIdx)
{
    vec1Info.bStart = batchInfo.bIdx;
    vec1Info.sStart = batchInfo.sIdx;
    vec1Info.resetResFlag = (loopIdx & (constInfo.nSize - 1)) == 0;
    vec1Info.c1v1DbIdx = (vec1Loop++ & (constInfo.dbWorkspaceRatio - 1));
    vec1Info.v1v2DbIdx = (vec2Loop & (constInfo.dbWorkspaceRatio - 1));
    BasicBlockInfo basicBlockInfo = SkipOneLoop(batchInfo);
    info.cubeDbIdx = (cubeLoop++ & (constInfo.dbWorkspaceRatio - 1));
    info.dealSeqCnt = basicBlockInfo.dealSeqCnt;
    info.dealTcNum = basicBlockInfo.dealTcNum;
    info.bStart = basicBlockInfo.bIdx;
    info.sStart = basicBlockInfo.sIdx;
    vec1Info.dealTcNum = basicBlockInfo.dealTcNum;
    vec1Info.dealScSize = basicBlockInfo.compressedTcNum;
    allCompressedTcNum_ += basicBlockInfo.compressedTcNum;
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedExcuteV2(Vec2RunInfo &vec2Info)
{
    return (vec2Info.dealScSize > 0);
}

template <typename COMP>
__aicore__ inline bool CompressorKernelPerf<COMP>::IsNeedSyncAll(uint32_t curBasicBlockIdx)
{
    if (allCompressedTcNum_ == 0) {
        return false;
    }

    uint32_t cnt = curBasicBlockIdx + 1;
    if ((cnt == loopTimes) || (cnt % constInfo.nSize == 0)) {
        return true;
    }
    return false;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::UpdateVec2Info(Vec2RunInfo &vec2Info, uint32_t curBasicBlockIdx,
                                                                  const Vec1RunInfo &info)
{
    // reset v2Info information at the start of each nSize round
    if (curBasicBlockIdx % constInfo.nSize == 0) {
        vec2Info.v2DbIdx = (vec2Loop & (constInfo.dbWorkspaceRatio - 1));
        vec2Info.bStart = info.bStart;
        vec2Info.sStart = info.sStart;
        // convert sStart to bCompressedId
        uint32_t startPos = tools_.GetStartPos(info.bStart);
        if (tools_.isExistSeqUsed_) {
            uint32_t seqUsed = tools_.GetSeqUsed(info.bStart);
            if (vec2Info.sStart >= seqUsed) {
                vec2Info.bStart++;
                vec2Info.sStart = 0;
            }
        }
        vec2Info.bCompressedId = (startPos + vec2Info.sStart) / constInfo.cmpRatio - startPos / constInfo.cmpRatio;

        vec2Info.dealScSize = 0;
    } else if ((curBasicBlockIdx + 1) % constInfo.nSize == 0) {
        vec2Loop++;
    }
    vec2Info.dealScSize += info.dealScSize;
    vec2Info.compressedId += info.dealScSize;
}

template <typename COMP>
__aicore__ inline void CompressorKernelPerf<COMP>::Process()
{
    // when the valid sequences of all batches are 0, exit directly
    if (constInfo.batchSize == 0) {
        return;
    }
    AllocEventID();

    BatchInfo batchInfo{};

    RunInfo extraInfo[1];
    Vec1RunInfo vec1Info{};
    Vec2RunInfo vec2Info{};
    SkipInvalidBatch(batchInfo);
    for (uint32_t i = 0; i < loopTimes; ++i) {
        RunInfo &extraInfo0 = extraInfo[0];
        CalcC1V1Params(extraInfo0, vec1Info, batchInfo, i);
        bool isNeedExcuteC1 = IsNeedExcuteC1(extraInfo0);

        if ASCEND_IS_AIC {
            ComputeMm1(extraInfo0, isNeedExcuteC1);
        } else {
            ComputeVec1(vec1Info);
            UpdateVec2Info(vec2Info, i, vec1Info);

            if (IsNeedSyncAll(i)) {
                SyncAll();
                if (IsNeedExcuteV2(vec2Info)) {
                    ComputeVec2(vec2Info);
                }
            }
        }
    }
    FreeEventID();
}

}  // namespace Compressor

#endif  // COMPRESSOR_KERNEL_PERF_H
