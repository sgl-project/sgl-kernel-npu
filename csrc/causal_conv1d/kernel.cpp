/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_common.h
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
constexpr int32_t MAX_BLOCK_DIM = 4096;
constexpr int32_t RING_SLOTS = 5;

__aicore__ inline int32_t SlotCurr(int32_t t)
{
    return (t + 3) % RING_SLOTS;
}

__aicore__ inline int32_t SlotHist(int32_t t, int32_t i)
{
    return (t + 3 - i) % RING_SLOTS;
}

__aicore__ inline int32_t SlotPrefetch(int32_t t)
{
    return (t + 4) % RING_SLOTS;
}

struct CalcBufLayout {
    AscendC::LocalTensor<float> weightF;
    AscendC::LocalTensor<float> biasF;
    AscendC::LocalTensor<float> accF;
    AscendC::LocalTensor<float> tmpF;
    AscendC::LocalTensor<float> currF;

    __aicore__ inline CalcBufLayout() = default;

    __aicore__ static inline CalcBufLayout FromCalcBuf(AscendC::TBuf<AscendC::QuePosition::VECCALC> &calcBuf)
    {
        CalcBufLayout layout;
        AscendC::LocalTensor<float> calc = calcBuf.template Get<float>();
        layout.weightF = calc;
        layout.biasF = calc[MAX_WIDTH * MAX_BLOCK_DIM];
        layout.accF = layout.biasF[MAX_BLOCK_DIM];
        layout.tmpF = layout.accF[MAX_BLOCK_DIM];
        layout.currF = layout.tmpF[MAX_BLOCK_DIM];
        return layout;
    }
};

} // namespace NsCausalConv1dCommon

#endif // CAUSAL_CONV1D_COMMON_H


/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file causal_conv1d_fn_tasks.h
 */

 #ifndef CAUSAL_CONV1D_FN_TASKS_H
 #define CAUSAL_CONV1D_FN_TASKS_H
 
 struct FnDirectBlockTask {
     bool valid = false;
     int32_t tokenTileId = 0;
     int32_t baseDimIdx = 0;
     int32_t tokenStart = 0;
     int32_t tokenEnd = 0;
     int32_t channelStart = 0;
     int32_t baseDimSize = 0;
 };
 
 __aicore__ inline FnDirectBlockTask ResolveFnDirectBlockTask(int32_t blockIdx, int32_t tokenBlockCnt, int32_t tokenBlockSize,
                                                              int32_t cuSeqlen, int32_t baseDimCnt, int32_t baseDim,
                                                              int32_t dim)
 {
     FnDirectBlockTask task;
     if (blockIdx < 0 || tokenBlockCnt <= 0 || tokenBlockSize <= 0 || cuSeqlen <= 0 || baseDimCnt <= 0 || baseDim <= 0 ||
         dim <= 0) {
         return task;
     }
 
     const int64_t phase1Grid = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
     if (phase1Grid <= 0 || static_cast<int64_t>(blockIdx) >= phase1Grid) {
         return task;
     }
 
     task.tokenTileId = blockIdx / baseDimCnt;
     task.baseDimIdx = blockIdx % baseDimCnt;
     task.channelStart = task.baseDimIdx * baseDim;
     if (task.channelStart >= dim) {
         return task;
     }
 
     task.baseDimSize = (task.channelStart + baseDim <= dim) ? baseDim : (dim - task.channelStart);
     task.tokenStart = task.tokenTileId * tokenBlockSize;
     if (task.tokenStart >= cuSeqlen) {
         return task;
     }
 
     const int32_t tokenEndRaw = task.tokenStart + tokenBlockSize;
     task.tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
     if (task.baseDimSize <= 0 || task.tokenEnd <= task.tokenStart) {
         return {};
     }
 
     task.valid = true;
     return task;
 }
 
 __aicore__ inline bool IsFnInitStateSnapshotOwnerBlock(const FnDirectBlockTask &task)
 {
     return task.valid && task.tokenTileId == 0;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline int32_t CAUSAL_CONV1D_CLASS::FindVarlenSeqByToken(int32_t tokenIdx) const
 {
     int32_t left = 0;
     int32_t right = static_cast<int32_t>(tilingData_->batch);
     while (left < right) {
         const int32_t mid = left + ((right - left) >> 1);
         const int32_t endVal = static_cast<int32_t>(queryStartLocGm.GetValue(mid + 1));
         if (tokenIdx < endVal) {
             right = mid;
         } else {
             left = mid + 1;
         }
     }
     return left;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq,
                                                                               int32_t &endSeq) const
 {
     if (!HasExplicitFnTokenSeqRanges() || tokenTileId < 0 || tokenTileId >= tilingData_->explicitTokenSeqRangeCount) {
         return false;
     }
     startSeq = static_cast<int32_t>(tilingData_->tokenTileStartSeq[tokenTileId]);
     endSeq = static_cast<int32_t>(tilingData_->tokenTileEndSeq[tokenTileId]);
     return (startSeq >= 0) && (endSeq >= startSeq);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::InitRingSeqSplit(int32_t seq, int32_t cacheIdx, bool hasInit,
                                                              int32_t seqStart, int32_t tileStart, int32_t tileLen,
                                                              int32_t channelStart, int32_t baseDim, int32_t dim)
 {
     const int32_t stateLen = tilingData_->stateLen;
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t historyCount = width - 1;
     const int32_t ringStart = MAX_WIDTH - width;
     const int32_t historyStartTok = tileStart - historyCount;
     LocalTensor<T> ring = inBuf.Get<T>();
     bool hasGmHistoryCopy = false;
     bool hasVectorInit = false;
     const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;
     int64_t xHistoryOffset = static_cast<int64_t>(historyStartTok) * dim + channelStart;
 
     for (int32_t i = 0; i < ringStart; ++i) {
         Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
         hasVectorInit = true;
     }
 
     for (int32_t i = 0, srcTok = historyStartTok; i < historyCount; ++i, ++srcTok, xHistoryOffset += dim) {
         LocalTensor<T> histSlot = ring[(ringStart + i) * MAX_BLOCK_DIM];
         if (srcTok >= seqStart) {
             DataCopy(histSlot, xGm[xHistoryOffset], baseDim);
             hasGmHistoryCopy = true;
         } else if (hasInit) {
             const int32_t statePos = srcTok - seqStart + historyCount;
             const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
             if (tilingData_->hasInitStateWorkspace != 0) {
                 const int64_t snapshotOffset =
                     (static_cast<int64_t>(seq) * historyCount + statePos) * dim + channelStart;
                 DataCopy(histSlot, initStateWorkspaceGm_[snapshotOffset], baseDim);
             } else {
                 DataCopy(histSlot, convStatesGm[stateOffset], baseDim);
             }
             hasGmHistoryCopy = true;
         } else {
             Duplicate(histSlot, static_cast<T>(0), baseDim);
             hasVectorInit = true;
         }
     }
 
     if (hasGmHistoryCopy) {
         SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
         WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
     }
     if (hasVectorInit) {
         PipeBarrier<PIPE_V>();
     }
 
     if (tileLen > 0) {
         const int32_t slot0 = SlotCurr(0);
         const int64_t xOffset = static_cast<int64_t>(tileStart) * dim + channelStart;
         DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);
         SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
     }
 
     if (tileLen > 1) {
         SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessFnChunk(int32_t seq, int32_t cacheIdx, bool hasInit,
                                                            int32_t seqStart, int32_t seqLen, int32_t chunkStart,
                                                            int32_t chunkLen, int32_t channelStart, int32_t baseDim,
                                                            int32_t dim)
 {
     LoadWeightAndBias(channelStart, baseDim);
     InitRingSeqSplit(seq, cacheIdx, hasInit, seqStart, chunkStart, chunkLen, channelStart, baseDim, dim);
 
     RunSeq(chunkStart, chunkLen, channelStart, baseDim, dim);
 
     MaybeWriteBackSeqSplitTailChunk(chunkStart, chunkLen, seqStart, seqLen, cacheIdx, channelStart, baseDim, dim);
     DrainTaskMte3();
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void
 CAUSAL_CONV1D_CLASS::MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                      int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                      int32_t baseDim, int32_t dim)
 {
     if (chunkStart + chunkLen != seqStart + seqLen) {
         return;
     }
 
     DrainTaskMte3();
     WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize)
 {
     if (tilingData_->hasInitStateWorkspace == 0) {
         return;
     }
 
     const int32_t dim = tilingData_->dim;
     const int32_t historyCount = static_cast<int32_t>(tilingData_->width - 1);
     const int32_t batch = tilingData_->batch;
     const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
     const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
     LocalTensor<T> tmpBuf = inBuf.Get<T>()[0 * MAX_BLOCK_DIM];
 
     for (int32_t seq = 0; seq < batch; ++seq) {
         if (!ResolveSeqHasInit(seq, hasInitialStateMode)) {
             continue;
         }
 
         int32_t cacheIdx = 0;
         if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
             continue;
         }
 
         const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * tilingData_->stateLen * dim + channelStart;
         const int64_t snapshotBaseOffset = static_cast<int64_t>(seq) * historyCount * dim + channelStart;
         for (int32_t statePos = 0; statePos < historyCount; ++statePos) {
             const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
             const int64_t snapshotOffset = snapshotBaseOffset + static_cast<int64_t>(statePos) * dim;
             DataCopy(tmpBuf, convStatesGm[stateOffset], baseDimSize);
             SetFlag<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
             WaitFlag<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
             DataCopy(initStateWorkspaceGm_[snapshotOffset], tmpBuf, baseDimSize);
             SetFlag<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
             WaitFlag<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
         }
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessVarlenTokenTiled()
 {
     const int32_t dim = tilingData_->dim;
     const int32_t batch = tilingData_->batch;
     const int32_t seqLen = tilingData_->seqLen;
     const int32_t cuSeqlen = tilingData_->cuSeqlen;
     const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
     const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
     const int32_t tokenBlockSize = static_cast<int32_t>(tilingData_->tokenBlockSize);
     const int32_t tokenBlockCnt = static_cast<int32_t>(tilingData_->tokenBlockCnt);
     const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
     const bool hasInitialStateMode = (tilingData_->hasInitialStateMode != 0);
     const bool isVarlenMode = (tilingData_->inputMode == 0);
 
     const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
     const auto blockTask = ResolveFnDirectBlockTask(blockIdx, tokenBlockCnt, tokenBlockSize, cuSeqlen, baseDimCnt,
                                                     baseDim, dim);
     if (tilingData_->hasInitStateWorkspace != 0) {
         if (IsFnInitStateSnapshotOwnerBlock(blockTask)) {
             PrefetchInitStatesToWorkspace(blockTask.channelStart, blockTask.baseDimSize);
         }
         SyncAll();
     }
     if (!blockTask.valid) {
         return;
     }
 
     int32_t seq = 0;
     int32_t seqUpperBound = batch;
     if (isVarlenMode) {
         if (!ResolveExplicitTokenTileSeqRange(blockTask.tokenTileId, seq, seqUpperBound)) {
             seq = FindVarlenSeqByToken(blockTask.tokenStart);
         }
     } else {
         seq = (seqLen > 0) ? (blockTask.tokenStart / seqLen) : 0;
     }
 
     int32_t cursor = blockTask.tokenStart;
     while (cursor < blockTask.tokenEnd && seq < seqUpperBound) {
         int32_t seqStart = 0;
         int32_t curSeqLen = 0;
         if (!ResolveSeqTaskWindow(seq, tilingData_->inputMode, seqLen, seqStart, curSeqLen)) {
             ++seq;
             continue;
         }
         const int32_t curSeqEnd = seqStart + curSeqLen;
         if (cursor < seqStart) {
             cursor = seqStart;
         }
         if (cursor >= curSeqEnd) {
             ++seq;
             continue;
         }
 
         const int32_t tileEnd = (blockTask.tokenEnd <= curSeqEnd) ? blockTask.tokenEnd : curSeqEnd;
         const int32_t tileLen = tileEnd - cursor;
         if (tileLen <= 0) {
             ++seq;
             continue;
         }
 
         int32_t cacheIdx = 0;
         if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
             cursor = tileEnd;
             ++seq;
             continue;
         }
 
         const bool hasInit = ResolveSeqHasInit(seq, hasInitialStateMode);
         ProcessFnChunk(seq, cacheIdx, hasInit, seqStart, curSeqLen, cursor, tileLen, blockTask.channelStart,
                        blockTask.baseDimSize, dim);
 
         cursor = tileEnd;
         ++seq;
     }
 }
 
 #endif
 

 /**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file causal_conv1d_fn.h
 */
 #ifndef CAUSAL_CONV1D_FN_H
 #define CAUSAL_CONV1D_FN_H
 
 #include "causal_conv1d.h"
 
 namespace NsCausalConv1d {
 
 template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
 class CausalConv1dFn : public CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_FN, widthKey, fnPlanKey> {
 public:
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                 GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                 GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
     {
         (void)numAcceptedTokens;
         this->ResetRuntimeState(tilingData);
         this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
         this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
         this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
         this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
         this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
         this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
         this->initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initialStateMode));
         this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
         if (tilingData->hasInitStateWorkspace != 0) {
             const uint64_t syncElems =
                 static_cast<uint64_t>(GetBlockNum()) * INIT_STATE_SYNCALL_NEED_SIZE;
             const uint64_t syncBytes = syncElems * sizeof(int32_t);
             const uint64_t workspaceElems =
                 static_cast<uint64_t>(tilingData->batch) *
                 static_cast<uint64_t>(tilingData->width - 1) *
                 static_cast<uint64_t>(tilingData->dim);
             this->initStateSyncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), syncElems);
             auto *workspaceBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
             this->initStateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspaceBytes + syncBytes),
                                                         workspaceElems);
         }
         this->InitSharedBuffersAndEvents();
     }
 
     __aicore__ inline void Process()
     {
         this->ProcessVarlenTokenTiled();
         this->ReleaseEvents();
     }
 };
 
 template <typename T, uint32_t widthKey, uint32_t fnPlanKey>
 __aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                          GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                          GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                          const CausalConv1dTilingData *tilingData)
 {
     CausalConv1dFn<T, widthKey, fnPlanKey> op;
     op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
             tilingData);
     op.Process();
 }
 
 }
 
 #endif
 

 /**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_data.h
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

enum FnExecutionPlan : int64_t {
    FN_EXECUTION_PLAN_INVALID = 0,
    FN_EXECUTION_PLAN_CUTBS = 1,
    FN_EXECUTION_PLAN_CUTBSD = 2,
};

inline constexpr int64_t ResolveFnExecutionPlan(int64_t baseDimCnt)
{
    return (baseDimCnt <= 0) ? FN_EXECUTION_PLAN_INVALID
        : (baseDimCnt <= 1) ? FN_EXECUTION_PLAN_CUTBS
        :                     FN_EXECUTION_PLAN_CUTBSD;
}


struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;
    int64_t batch;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t hasBias;
    int64_t baseDim;
    int64_t baseDimCnt;
    int64_t hasNumAcceptedTokens;
    int64_t hasCacheIndices;
    int64_t hasInitialStateMode;
    int64_t tokenBlockSize;
    int64_t tokenBlockCnt;
    int64_t hasExplicitTokenSeqRanges;
    int64_t explicitTokenSeqRangeCount;
    int64_t tokenTileStartSeq[128];
    int64_t tokenTileEndSeq[128];
    int64_t hasInitStateWorkspace;
};
#endif // CAUSAL_CONV1D_TILING_DATA_H_


/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_key.h
 * \brief causal_conv1d tiling key declare
 */

#ifndef __CAUSAL_CONV1D_TILING_KEY_H__
#define __CAUSAL_CONV1D_TILING_KEY_H__

#include "causal_conv1d_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

#define CAUSAL_CONV1D_TPL_RUN_MODE_FN 0
#define CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE 1
#define CAUSAL_CONV1D_TPL_WIDTH_RUNTIME 0
#define CAUSAL_CONV1D_TPL_WIDTH_2 1
#define CAUSAL_CONV1D_TPL_WIDTH_3 2
#define CAUSAL_CONV1D_TPL_WIDTH_4 3
#define CAUSAL_CONV1D_TPL_FN_PLAN_INVALID 0
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS 1
#define CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD 2
ASCENDC_TPL_ARGS_DECL(CausalConv1d,
                      ASCENDC_TPL_UINT_DECL(runModeKey, 1, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_RUN_MODE_FN,
                                            CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE),
                      ASCENDC_TPL_UINT_DECL(widthKey, 2, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                            CAUSAL_CONV1D_TPL_WIDTH_2, CAUSAL_CONV1D_TPL_WIDTH_3,
                                            CAUSAL_CONV1D_TPL_WIDTH_4),
                      ASCENDC_TPL_UINT_DECL(fnPlanKey, 2, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_FN_PLAN_INVALID,
                                            CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS, CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#define CAUSAL_CONV1D_TPL_SEL_ENTRY(RUN_MODE, WIDTH, FN_PLAN)                                                 \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(runModeKey, ASCENDC_TPL_UI_LIST, RUN_MODE),                    \
                         ASCENDC_TPL_UINT_SEL(widthKey, ASCENDC_TPL_UI_LIST, WIDTH),                          \
                         ASCENDC_TPL_UINT_SEL(fnPlanKey, ASCENDC_TPL_UI_LIST, FN_PLAN),                       \
                         ASCENDC_TPL_TILING_STRUCT_SEL(CausalConv1dTilingData))

// Keep entries in encoded tiling-key order: real-device sub-kernel dispatch is sensitive to declaration order.
ASCENDC_TPL_SEL(
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                                CAUSAL_CONV1D_TPL_FN_PLAN_INVALID),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_2,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_3,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_4,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_2,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_3,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD),
    CAUSAL_CONV1D_TPL_SEL_ENTRY(CAUSAL_CONV1D_TPL_RUN_MODE_FN, CAUSAL_CONV1D_TPL_WIDTH_4,
                                CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD));

#undef CAUSAL_CONV1D_TPL_SEL_ENTRY

#endif // __CAUSAL_CONV1D_TILING_KEY_H__


/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file causal_conv1d_update.h
 * \brief causal_conv1d update
 */
#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T>
class CausalConv1dUpdate
    : public CausalConv1d<T, CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME,
                          CAUSAL_CONV1D_TPL_FN_PLAN_INVALID> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)workspace;
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
        this->numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(numAcceptedTokens));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        const CausalConv1dTilingData *tilingData = this->GetTilingData();
        const int32_t dim = tilingData->dim;
        const int32_t baseDimCnt = static_cast<int32_t>(tilingData->baseDimCnt);
        const int32_t width = static_cast<int32_t>(tilingData->width);
        const int32_t baseDim = static_cast<int32_t>(tilingData->baseDim);
        if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH || dim <= 0 ||
            tilingData->batch <= 0) {
            this->ReleaseEvents();
            return;
        }

        this->ProcessDefault();
        this->ReleaseEvents();
    }
};

template <typename T>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                             const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H


/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.cpp
 * \brief
 */

#include "causal_conv1d_fn.h"
#include "causal_conv1d_update.h"

namespace {

template <typename T, uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                       GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                       const CausalConv1dTilingData *tilingData)
{
    if constexpr (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        NsCausalConv1d::RunCausalConv1dFn<T, widthKey, fnPlanKey>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    } else {
        NsCausalConv1d::RunCausalConv1dUpdate<T>(
            x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    }
}

} // namespace

template <uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey>
__global__ __aicore__ void causal_conv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CausalConv1dTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = AscendC::GetUserWorkspace(workspace);
    }

    RunCausalConv1d<DTYPE_X, runModeKey, widthKey, fnPlanKey>(
        x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y,
        userWorkspace, &tilingData);
}


/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d.h
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/causal_conv1d_regbase.h"
#endif
 
 namespace NsCausalConv1d {
 
 using namespace AscendC;
 using namespace NsCausalConv1dCommon;
 
 #define CAUSAL_CONV1D_TEMPLATE_ARGS typename T, uint32_t runModeKey, uint32_t widthKey, uint32_t fnPlanKey
 #define CAUSAL_CONV1D_CLASS CausalConv1d<T, runModeKey, widthKey, fnPlanKey>
 
 enum SeqTaskWindowMode : int32_t {
     SEQ_TASK_WINDOW_MODE_VARLEN = 0,
     SEQ_TASK_WINDOW_MODE_BATCH = 1,
     SEQ_TASK_WINDOW_MODE_DECODE2D = 2,
 };
 
 inline constexpr int32_t INIT_STATE_SYNCALL_NEED_SIZE = 8;
 inline constexpr int32_t INIT_STATE_SYNCALL_MAX_BLOCKS = 64;
 
 struct SeqTaskWindow {
     bool valid = false;
     int32_t start = 0;
     int32_t len = 0;
 };
 
 __aicore__ inline int32_t GetSeqTaskWindowMode(int32_t inputMode)
 {
     if (inputMode == 0) {
         return SEQ_TASK_WINDOW_MODE_VARLEN;
     }
     if (inputMode == 2) {
         return SEQ_TASK_WINDOW_MODE_DECODE2D;
     }
     return SEQ_TASK_WINDOW_MODE_BATCH;
 }
 
 __aicore__ inline SeqTaskWindow BuildSeqTaskWindowVarlen(int32_t startVal, int32_t endVal)
 {
     SeqTaskWindow window;
     window.start = startVal;
     window.len = endVal - startVal;
     window.valid = (window.len > 0);
     return window;
 }
 
 __aicore__ inline int32_t RetreatRingSlot(int32_t slot, int32_t delta)
 {
     int32_t prev = slot - delta;
     return (prev >= 0) ? prev : (prev + RING_SLOTS);
 }
 
 __aicore__ inline SeqTaskWindow BuildSeqTaskWindowBatch(int32_t seq, int32_t seqLen)
 {
     SeqTaskWindow window;
     window.start = seq * seqLen;
     window.len = seqLen;
     window.valid = (window.len > 0);
     return window;
 }
 
 __aicore__ inline SeqTaskWindow BuildSeqTaskWindowDecode2D(int32_t seq)
 {
     SeqTaskWindow window;
     window.valid = true;
     window.start = seq;
     window.len = 1;
     return window;
 }
 
 __aicore__ inline constexpr int32_t DecodeWidthTplKey(uint32_t widthKey)
 {
     switch (widthKey) {
         case CAUSAL_CONV1D_TPL_WIDTH_2:
             return 2;
         case CAUSAL_CONV1D_TPL_WIDTH_3:
             return 3;
         case CAUSAL_CONV1D_TPL_WIDTH_4:
             return 4;
         default:
             return 0;
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 class CausalConv1d {
 public:
     __aicore__ inline CausalConv1d() = default;
 
 protected:
     static constexpr bool kIsUpdateMode = (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_UPDATE);
     static constexpr int32_t kTemplateWidth = DecodeWidthTplKey(widthKey);
     static constexpr bool kHasCompileTimeWidth =
         (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN) && (kTemplateWidth >= 2) && (kTemplateWidth <= MAX_WIDTH);
     static constexpr FnExecutionPlan kFnExecutionPlan = static_cast<FnExecutionPlan>(fnPlanKey);
 
     __aicore__ inline void ResetRuntimeState(const CausalConv1dTilingData *tilingData);
     __aicore__ inline void InitSharedBuffersAndEvents();
     __aicore__ inline void LoadWeightAndBias(int32_t channelStart, int32_t baseDim);
     __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                     int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
     __aicore__ inline void InitRingSeqSplit(int32_t seq, int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                             int32_t tileStart, int32_t tileLen, int32_t channelStart, int32_t baseDim,
                                             int32_t dim);
     __aicore__ inline void PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize);
     __aicore__ inline void RestoreFnLocalPartials(int32_t baseDim);
     __aicore__ inline void ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim);
     __aicore__ inline void AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim);
     __aicore__ inline void RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim,
                                            int32_t dim);
     __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
     __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                           int32_t dim);
     __aicore__ inline void WriteBackStateSpec(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                               int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
     __aicore__ inline void DrainTaskMte3();
     __aicore__ inline void AllocEvents();
     __aicore__ inline void ReleaseEvents();
     __aicore__ inline int32_t FindVarlenSeqByToken(int32_t tokenIdx) const;
     __aicore__ inline bool ResolveExplicitTokenTileSeqRange(int32_t tokenTileId, int32_t &startSeq, int32_t &endSeq) const;
     __aicore__ inline bool ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen, int32_t &start,
                                                 int32_t &len) const;
     template <int32_t kWindowMode>
     __aicore__ inline bool ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start, int32_t &len) const;
     __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices, int32_t &cacheIdx) const;
     __aicore__ inline bool ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const;
     __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                            int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                            int32_t baseDim, int32_t dim);
     __aicore__ inline void ProcessDefault();
     template <int32_t kWindowMode>
     __aicore__ inline void ProcessDefaultByWindowMode();
     __aicore__ inline void ProcessVarlenTokenTiled();
     __aicore__ inline void ProcessFnChunk(int32_t seq, int32_t cacheIdx, bool hasInit, int32_t seqStart,
                                           int32_t seqLen, int32_t chunkStart, int32_t chunkLen, int32_t channelStart,
                                           int32_t baseDim, int32_t dim);
     __aicore__ inline const CausalConv1dTilingData *GetTilingData() const;
     __aicore__ inline bool HasActivation() const;
     __aicore__ inline bool HasBias() const;
     __aicore__ inline bool IsUpdateMode() const;
     __aicore__ inline bool IsFnRollingFastPathEnabled() const;
     __aicore__ inline bool HasExplicitFnTokenSeqRanges() const;
     __aicore__ inline bool IsUpdateSpecDecodingEnabled() const;
 
 protected:
     TPipe pipe;
     TBuf<QuePosition::VECIN> inBuf;
     TBuf<QuePosition::VECOUT> outBuf;
     TBuf<QuePosition::VECCALC> calcBuf;
 
     TEventID weightBiasMte2ToVEvent_;
     TEventID stateMte2ToVEvent_;
     TEventID inputMte2ToVEvent_[RING_SLOTS];
     TEventID inputVToMte2Event_;
     TEventID outMte3ToVEvent_[2];
     TEventID outVToMte3Event_[2];
     TEventID stateWritebackMte3ToVEvent_;
     TEventID stateWritebackMte3ToMte2Event_;
     TEventID stateShiftMte2ToMte3Event_;
     TEventID stateShiftVToMte3Event_;
     TEventID stateShiftMte3ToMte2Event_;
     TEventID initSnapshotMte2ToMte3Event_;
     TEventID initSnapshotMte3ToMte2Event_;
     TEventID initSyncVToMte3Event_;
     TEventID initSyncMte3ToVEvent_;
     TEventID specWritebackMte2ToMte3Event_[2];
     TEventID specWritebackMte3ToMte2Event_[2];
 
     GlobalTensor<T> xGm;
     GlobalTensor<T> weightGm;
     GlobalTensor<T> biasGm;
     GlobalTensor<T> convStatesGm;
     GlobalTensor<int64_t> queryStartLocGm;
     GlobalTensor<int64_t> cacheIndicesGm;
     GlobalTensor<int64_t> initialStateModeGm;
     GlobalTensor<int64_t> numAcceptedTokensGm;
     GlobalTensor<T> yGm;
     GlobalTensor<int32_t> initStateSyncGm_;
     GlobalTensor<T> initStateWorkspaceGm_;
 
     const CausalConv1dTilingData *tilingData_{nullptr};
 };
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ResetRuntimeState(const CausalConv1dTilingData *tilingData)
 {
     tilingData_ = tilingData;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::InitSharedBuffersAndEvents()
 {
     pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
     pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
     pipe.InitBuffer(calcBuf, (MAX_WIDTH + 4) * MAX_BLOCK_DIM * sizeof(float));
     AllocEvents();
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::AllocEvents()
 {
     weightBiasMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
     stateMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
     for (int32_t i = 0; i < RING_SLOTS; ++i) {
         inputMte2ToVEvent_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
     }
     inputVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
     outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
     outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
     outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
     outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
     stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
     stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
     stateShiftMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
     stateShiftVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
     stateShiftMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
     initSnapshotMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
     initSnapshotMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
     initSyncVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
     initSyncMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
     specWritebackMte2ToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
     specWritebackMte2ToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
     specWritebackMte3ToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
     specWritebackMte3ToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ReleaseEvents()
 {
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(stateMte2ToVEvent_);
     for (int32_t i = 0; i < RING_SLOTS; ++i) {
         GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[i]);
     }
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(initSnapshotMte2ToMte3Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(initSnapshotMte3ToMte2Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(initSyncVToMte3Event_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(initSyncMte3ToVEvent_);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[1]);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
     GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::LoadWeightAndBias(int32_t channelStart, int32_t baseDim)
 {
     const int32_t dim = tilingData_->dim;
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t jStart = MAX_WIDTH - width;
     const bool hasBias = HasBias();
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &weightF = cl.weightF;
     LocalTensor<float> &biasF = cl.biasF;
     LocalTensor<T> weightT;
     LocalTensor<T> biasT;
 
     if constexpr (!std::is_same<T, float>::value) {
         weightT = weightF.ReinterpretCast<T>();
         biasT = biasF.ReinterpretCast<T>();
     }
 
     for (int32_t j = 0; j < jStart; ++j) {
         Duplicate(weightF[j * MAX_BLOCK_DIM], 0.0f, baseDim);
     }
 
     for (int32_t j = 0; j < width; ++j) {
         const int32_t jDst = jStart + j;
         const int64_t weightOffset = static_cast<int64_t>(j) * dim + channelStart;
 
         if constexpr (std::is_same<T, float>::value) {
             DataCopy(weightF[jDst * MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
         } else {
             DataCopy(weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], weightGm[weightOffset], baseDim);
         }
     }
 
     if (hasBias) {
         if constexpr (std::is_same<T, float>::value) {
             DataCopy(biasF, biasGm[channelStart], baseDim);
         } else {
             DataCopy(biasT[MAX_BLOCK_DIM], biasGm[channelStart], baseDim);
         }
     }
 
     SetFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
     WaitFlag<HardEvent::MTE2_V>(weightBiasMte2ToVEvent_);
 
     if constexpr (!std::is_same<T, float>::value) {
         for (int32_t j = 0; j < width; ++j) {
             const int32_t jDst = jStart + j;
             Cast(weightF[jDst * MAX_BLOCK_DIM], weightT[jDst * MAX_BLOCK_DIM * 2 + MAX_BLOCK_DIM], RoundMode::CAST_NONE,
                  baseDim);
         }
         if (hasBias) {
             Cast(biasF, biasT[MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         }
         PipeBarrier<PIPE_V>();
     }
 
     if (!hasBias) {
         Duplicate(biasF, 0.0f, baseDim);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                      int32_t start, int32_t len, int32_t channelStart,
                                                      int32_t baseDim, int32_t dim)
 {
     const int32_t stateLen = tilingData_->stateLen;
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t ringStart = MAX_WIDTH - width;
     LocalTensor<T> ring = inBuf.Get<T>();
 
     for (int32_t i = 0; i < ringStart; ++i) {
         Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
     }
     if (ringStart > 0) {
         PipeBarrier<PIPE_V>();
     }
 
     if (hasInit) {
         for (int32_t i = 0; i < (width - 1); ++i) {
             const int32_t pos = stateTokenOffset + i;
             const int64_t stateOffset =
                 static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + channelStart;
             DataCopy(ring[(ringStart + i) * MAX_BLOCK_DIM], convStatesGm[stateOffset], baseDim);
         }
         SetFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
         WaitFlag<HardEvent::MTE2_V>(stateMte2ToVEvent_);
     } else {
         for (int32_t i = 0; i < (width - 1); ++i) {
             Duplicate(ring[(ringStart + i) * MAX_BLOCK_DIM], static_cast<T>(0), baseDim);
         }
         PipeBarrier<PIPE_V>();
     }
 
     if (len > 0) {
         const int32_t slot0 = SlotCurr(0);
         const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
         DataCopy(ring[slot0 * MAX_BLOCK_DIM], xGm[xOffset], baseDim);
         SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slot0]);
     }
 
     if (len > 1) {
         SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeq(int32_t start, int32_t len, int32_t channelStart,
                                                    int32_t baseDim, int32_t dim)
 {
     if (IsFnRollingFastPathEnabled()) {
         RunSeqFnRolling(start, len, channelStart, baseDim, dim);
         return;
     }
 
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t jStart = MAX_WIDTH - width;
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &weightF = cl.weightF;
     LocalTensor<float> &biasF = cl.biasF;
     LocalTensor<float> &accF = cl.accF;
     LocalTensor<float> &tmpF = cl.tmpF;
     LocalTensor<T> ring = inBuf.Get<T>();
     LocalTensor<T> outT = outBuf.Get<T>();
     const bool hasBias = HasBias();
     const bool hasActivation = HasActivation();
     for (int32_t t = 0; t < len; ++t) {
         const int32_t slotCurr = SlotCurr(t);
 
         WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);
 
         if (t + 1 < len) {
             const int32_t slotNext = SlotPrefetch(t);
             const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
             WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
             DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], baseDim);
             SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
         }
 
         bool accInitialized = false;
         if (hasBias) {
             Adds(accF, biasF, 0.0f, baseDim);
             PipeBarrier<PIPE_V>();
             accInitialized = true;
         }
 
         for (int32_t j = jStart; j < MAX_WIDTH; ++j) {
             const int32_t tap = (MAX_WIDTH - 1) - j;
             const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
             Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
             PipeBarrier<PIPE_V>();
             if (!accInitialized) {
                 Mul(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
                 accInitialized = true;
             } else {
                 MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], baseDim);
             }
         }
 
         PipeBarrier<PIPE_V>();
 
         if (hasActivation) {
             Silu(tmpF, accF, baseDim);
         }
 
         const int32_t outSlot = t & 1;
         LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
         if (t >= 2) {
             WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
         }
 
         if constexpr (IsSameType<T, float>::value) {
             if (hasActivation) {
                 DataCopy(outSlotT, tmpF, baseDim);
             } else {
                 DataCopy(outSlotT, accF, baseDim);
             }
         } else {
             if (hasActivation) {
                 Cast(outSlotT, tmpF, RoundMode::CAST_RINT, baseDim);
             } else {
                 Cast(outSlotT, accF, RoundMode::CAST_RINT, baseDim);
             }
         }
 
         SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
 
         const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
         WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
         DataCopy(yGm[outOffset], outSlotT, baseDim);
         if (t + 2 < len) {
             SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
         }
 
         if (t + 2 < len) {
             SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
         }
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::RestoreFnLocalPartials(int32_t baseDim)
 {
     if constexpr (!kHasCompileTimeWidth) {
         return;
     }
 
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &weightF = cl.weightF;
     LocalTensor<float> &state2F = cl.biasF;
     LocalTensor<float> &state1F = cl.accF;
     LocalTensor<float> &state0F = cl.tmpF;
     LocalTensor<float> &currF = cl.currF;
     LocalTensor<T> ring = inBuf.Get<T>();
     constexpr int32_t ringStart = MAX_WIDTH - kTemplateWidth;
     constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;
 
     if constexpr (kTemplateWidth == 2) {
         Duplicate(state2F, 0.0f, baseDim);
         Duplicate(state1F, 0.0f, baseDim);
         PipeBarrier<PIPE_V>();
 
         Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
     } else if constexpr (kTemplateWidth == 3) {
         Duplicate(state2F, 0.0f, baseDim);
         PipeBarrier<PIPE_V>();
 
         Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
 
         Cast(currF, ring[(ringStart + 1) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
         MulAddDst(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
     } else if constexpr (kTemplateWidth == 4) {
         Cast(currF, ring[ringStart * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
 
         Cast(currF, ring[(ringStart + 1) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
         MulAddDst(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
 
         Cast(currF, ring[(ringStart + 2) * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
         PipeBarrier<PIPE_V>();
         Mul(state2F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
         MulAddDst(state1F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
         MulAddDst(state0F, currF, weightF[(w0Idx + 2) * MAX_BLOCK_DIM], baseDim);
         PipeBarrier<PIPE_V>();
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim)
 {
     if constexpr (!kHasCompileTimeWidth) {
         return;
     }
 
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &weightF = cl.weightF;
     LocalTensor<float> &state0F = cl.tmpF;
     LocalTensor<float> &currF = cl.currF;
     LocalTensor<T> ring = inBuf.Get<T>();
 
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    const bool hasActivation = HasActivation();
    if (hasActivation) {
        ComputeFnRollingOutputRegbase<T, true>(ring[slotCurr * MAX_BLOCK_DIM], currF, state0F, weightF[3 * MAX_BLOCK_DIM], baseDim);
    } else {
        ComputeFnRollingOutputRegbase<T, false>(ring[slotCurr * MAX_BLOCK_DIM], currF, state0F, weightF[3 * MAX_BLOCK_DIM], baseDim);
    }
#else
    Cast(currF, ring[slotCurr * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
    PipeBarrier<PIPE_V>();
    MulAddDst(state0F, currF, weightF[3 * MAX_BLOCK_DIM], baseDim);
    PipeBarrier<PIPE_V>();

    const bool hasActivation = HasActivation();
    if (hasActivation) {
        PipeBarrier<PIPE_V>();
        Silu(currF, state0F, baseDim);
    }
#endif
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim)
 {
     if constexpr (!kHasCompileTimeWidth) {
         return;
     }
 
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &weightF = cl.weightF;
     LocalTensor<float> &state2F = cl.biasF;
     LocalTensor<float> &state1F = cl.accF;
     LocalTensor<float> &state0F = cl.tmpF;
     LocalTensor<float> &currF = cl.currF;
     LocalTensor<T> ring = inBuf.Get<T>();
     constexpr int32_t w0Idx = MAX_WIDTH - kTemplateWidth;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    AdvanceFnLocalPartialsRegbase<T, kTemplateWidth>(ring[slotCurr * MAX_BLOCK_DIM], weightF[w0Idx * MAX_BLOCK_DIM], 
        state0F, state1F, state2F, baseDim, MAX_BLOCK_DIM);
#else
    Cast(currF, ring[slotCurr * MAX_BLOCK_DIM], RoundMode::CAST_NONE, baseDim);
    PipeBarrier<PIPE_V>();

    if constexpr (kTemplateWidth == 2) {
        Mul(state0F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 3) {
        Mul(state0F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state0F, state0F, state1F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state1F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    } else if constexpr (kTemplateWidth == 4) {
        Mul(state0F, currF, weightF[(w0Idx + 2) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state0F, state0F, state1F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state1F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
        Add(state1F, state1F, state2F, baseDim);
        PipeBarrier<PIPE_V>();

        Mul(state2F, currF, weightF[w0Idx * MAX_BLOCK_DIM], baseDim);
        PipeBarrier<PIPE_V>();
    }
#endif
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart,
                                                             int32_t baseDim, int32_t dim)
 {
     if constexpr (!kHasCompileTimeWidth) {
         return;
     }
 
     auto cl = CalcBufLayout::FromCalcBuf(calcBuf);
     LocalTensor<float> &state0F = cl.tmpF;
     LocalTensor<float> &currF = cl.currF;
     LocalTensor<T> ring = inBuf.Get<T>();
     LocalTensor<T> outT = outBuf.Get<T>();
     const bool hasActivation = HasActivation();
     RestoreFnLocalPartials(baseDim);
 
     for (int32_t t = 0; t < len; ++t) {
         const int32_t slotCurr = SlotCurr(t);
 
         WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotCurr]);
 
         if (t + 1 < len) {
             const int32_t slotNext = SlotPrefetch(t);
             const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
             WaitFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
             DataCopy(ring[slotNext * MAX_BLOCK_DIM], xGm[xOffsetNext], baseDim);
             SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_[slotNext]);
         }
 
         ComputeFnRollingOutput(slotCurr, baseDim);
 
         const int32_t outSlot = t & 1;
         LocalTensor<T> outSlotT = outT[outSlot * MAX_BLOCK_DIM];
         if (t >= 2) {
             WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
         }
 
         if constexpr (IsSameType<T, float>::value) {
             if (hasActivation) {
                 DataCopy(outSlotT, currF, baseDim);
             } else {
                 DataCopy(outSlotT, state0F, baseDim);
             }
         } else {
             if (hasActivation) {
                 Cast(outSlotT, currF, RoundMode::CAST_RINT, baseDim);
             } else {
                 Cast(outSlotT, state0F, RoundMode::CAST_RINT, baseDim);
             }
         }
 
         AdvanceFnLocalPartials(slotCurr, baseDim);
 
         SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
 
         const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
         WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
         DataCopy(yGm[outOffset], outSlotT, baseDim);
         if (t + 2 < len) {
             SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
         }
 
         if (t + 2 < len) {
             SetFlag<HardEvent::V_MTE2>(inputVToMte2Event_);
         }
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::DrainTaskMte3()
 {
     SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
     WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
     SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
     WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart,
                                                            int32_t baseDim, int32_t dim)
 {
     const int32_t stateLen = tilingData_->stateLen;
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     if (len <= 0) {
         return;
     }
 
     const int32_t lastT = len - 1;
     LocalTensor<T> ring = inBuf.Get<T>();
     const int32_t lastSlot = SlotCurr(lastT);
     const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;
 
     for (int32_t pos = 0; pos < (width - 1); ++pos) {
         const int32_t tap = (width - 2) - pos;
         const int32_t slot = RetreatRingSlot(lastSlot, tap);
         const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * dim;
         DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], baseDim);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::WriteBackStateSpec(int32_t cacheIdx, bool hasInit,
                                                                int32_t stateTokenOffset, int32_t start, int32_t len,
                                                                int32_t channelStart, int32_t baseDim, int32_t dim)
 {
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const int32_t stateLen = tilingData_->stateLen;
     if (len <= 0) {
         return;
     }
 
     if (width != 4) {
         WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
         return;
     }
 
     constexpr int32_t keep = MAX_WIDTH - 2;
     const int32_t reqStateLen = keep + len;
     if (reqStateLen > stateLen) {
         WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
         return;
     }
 
     LocalTensor<T> ring = inBuf.Get<T>();
     LocalTensor<T> buf0 = ring[0 * MAX_BLOCK_DIM];
     LocalTensor<T> buf1 = ring[1 * MAX_BLOCK_DIM];
 
     if (hasInit) {
         const int32_t srcPos0 = stateTokenOffset + 1;
         const int32_t srcPos1 = stateTokenOffset + 2;
         const int64_t srcOffset0 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos0) * dim + channelStart;
         const int64_t srcOffset1 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos1) * dim + channelStart;
         DataCopy(buf0, convStatesGm[srcOffset0], baseDim);
         DataCopy(buf1, convStatesGm[srcOffset1], baseDim);
         SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
         WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
         const int64_t dstOffset0 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
         const int64_t dstOffset1 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
         DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
         DataCopy(convStatesGm[dstOffset1], buf1, baseDim);
         SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
         WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
     } else {
         Duplicate(buf0, static_cast<T>(0), baseDim);
         SetFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
         WaitFlag<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
         const int64_t dstOffset0 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
         const int64_t dstOffset1 =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
         DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
         DataCopy(convStatesGm[dstOffset1], buf0, baseDim);
         SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
         WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
     }
 
     const int64_t xOffset0 = static_cast<int64_t>(start) * dim + channelStart;
     DataCopy(buf0, xGm[xOffset0], baseDim);
     SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
 
     for (int32_t t = 0; t < len; ++t) {
         const int32_t curr = t & 1;
         const int32_t next = curr ^ 1;
         LocalTensor<T> currBuf = (curr == 0) ? buf0 : buf1;
         LocalTensor<T> nextBuf = (next == 0) ? buf0 : buf1;
 
         WaitFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[curr]);
 
         if (t + 1 < len) {
             const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
             if (t > 0) {
                 WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[next]);
             }
             DataCopy(nextBuf, xGm[xOffsetNext], baseDim);
             SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[next]);
         }
 
         const int64_t dstOffset =
             static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(keep + t) * dim + channelStart;
         DataCopy(convStatesGm[dstOffset], currBuf, baseDim);
         SetFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[curr]);
     }
 
     WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
     if (len > 1) {
         WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqTaskWindow(int32_t seq, int32_t inputMode, int32_t seqLen,
                                                                  int32_t &start, int32_t &len) const
 {
     switch (GetSeqTaskWindowMode(inputMode)) {
         case SEQ_TASK_WINDOW_MODE_VARLEN:
             return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_VARLEN>(seq, seqLen, start, len);
         case SEQ_TASK_WINDOW_MODE_DECODE2D:
             return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_DECODE2D>(seq, seqLen, start, len);
         default:
             return ResolveSeqTaskWindowByMode<SEQ_TASK_WINDOW_MODE_BATCH>(seq, seqLen, start, len);
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 template <int32_t kWindowMode>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqTaskWindowByMode(int32_t seq, int32_t seqLen, int32_t &start,
                                                                        int32_t &len) const
 {
     SeqTaskWindow window;
     if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_VARLEN) {
         const int32_t startVal = queryStartLocGm.GetValue(seq);
         const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
         window = BuildSeqTaskWindowVarlen(startVal, endVal);
     } else if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_DECODE2D) {
         window = BuildSeqTaskWindowDecode2D(seq);
     } else {
         window = BuildSeqTaskWindowBatch(seq, seqLen);
     }
 
     if (!window.valid) {
         return false;
     }
     start = window.start;
     len = window.len;
     return true;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices,
                                                                  int32_t &cacheIdx) const
 {
     cacheIdx = seq;
     if (!hasCacheIndices) {
         return true;
     }
 
     const int64_t cacheIdx64 = cacheIndicesGm.GetValue(seq);
     if (cacheIdx64 == tilingData_->padSlotId) {
         return false;
     }
     cacheIdx = static_cast<int32_t>(cacheIdx64);
     return true;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const
 {
     return hasInitialStateMode ? (initialStateModeGm.GetValue(seq) != 0) : false;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefault()
 {
     switch (GetSeqTaskWindowMode(tilingData_->inputMode)) {
         case SEQ_TASK_WINDOW_MODE_VARLEN:
             ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
             return;
         case SEQ_TASK_WINDOW_MODE_DECODE2D:
             ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_DECODE2D>();
             return;
         default:
             ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
             return;
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 template <int32_t kWindowMode>
 __aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefaultByWindowMode()
 {
     const int32_t dim = tilingData_->dim;
     const int32_t batch = tilingData_->batch;
     const int32_t seqLen = tilingData_->seqLen;
     const int32_t baseDim = static_cast<int32_t>(tilingData_->baseDim);
     const int32_t baseDimCnt = static_cast<int32_t>(tilingData_->baseDimCnt);
     const int32_t width = static_cast<int32_t>(tilingData_->width);
     const bool hasCacheIndices = (tilingData_->hasCacheIndices != 0);
     const bool hasInit = true;
     const bool isSpecDecodingGlobal = IsUpdateSpecDecodingEnabled();
 
     const uint32_t blockIdx = GetBlockIdx();
     const uint32_t blockNum = GetBlockNum();
 
     if (baseDim <= 0 || baseDimCnt <= 0 || baseDim > MAX_BLOCK_DIM || width < 2 || width > MAX_WIDTH) {
         ReleaseEvents();
         return;
     }
 
     const int64_t gridSize = static_cast<int64_t>(batch) * baseDimCnt;
     for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
         const int32_t seq = static_cast<int32_t>(task / baseDimCnt);
         const int32_t baseDimIdx = static_cast<int32_t>(task % baseDimCnt);
         const int32_t channelStart = baseDimIdx * baseDim;
         if (channelStart >= dim) {
             continue;
         }
         const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);
 
         int32_t start = 0;
         int32_t len = 0;
         if (!ResolveSeqTaskWindowByMode<kWindowMode>(seq, seqLen, start, len)) {
             continue;
         }
 
         int32_t cacheIdx = 0;
         if (!ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
             continue;
         }
 
         int32_t stateTokenOffset = 0;
         if (isSpecDecodingGlobal) {
             int32_t accepted = static_cast<int32_t>(numAcceptedTokensGm.GetValue(seq));
             stateTokenOffset = accepted - 1;
             const int32_t maxOffset = static_cast<int32_t>(tilingData_->stateLen - (width - 1));
             if (stateTokenOffset < 0) {
                 stateTokenOffset = 0;
             } else if (stateTokenOffset > maxOffset) {
                 stateTokenOffset = maxOffset;
             }
         }
 
         LoadWeightAndBias(channelStart, curBaseDim);
 
         InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
         RunSeq(start, len, channelStart, curBaseDim, dim);
 
         if (isSpecDecodingGlobal) {
             DrainTaskMte3();
             WriteBackStateSpec(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
         } else {
             WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
         }
 
         DrainTaskMte3();
     }
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline const CausalConv1dTilingData *CAUSAL_CONV1D_CLASS::GetTilingData() const
 {
     return tilingData_;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::HasActivation() const
 {
     return (tilingData_ != nullptr) && (tilingData_->activationMode != 0);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::HasBias() const
 {
     return (tilingData_ != nullptr) && (tilingData_->hasBias != 0);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateMode() const
 {
     return kIsUpdateMode;
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::IsFnRollingFastPathEnabled() const
 {
     return !kIsUpdateMode && (tilingData_ != nullptr) && (kFnExecutionPlan != FN_EXECUTION_PLAN_INVALID) &&
            (tilingData_->hasNumAcceptedTokens == 0) && !HasBias();
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::HasExplicitFnTokenSeqRanges() const
 {
     return !kIsUpdateMode && (tilingData_ != nullptr) && (tilingData_->inputMode == 0) &&
            (tilingData_->hasExplicitTokenSeqRanges != 0) &&
            (tilingData_->explicitTokenSeqRangeCount >= tilingData_->tokenBlockCnt);
 }
 
 template <CAUSAL_CONV1D_TEMPLATE_ARGS>
 __aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateSpecDecodingEnabled() const
 {
     return kIsUpdateMode && (tilingData_->hasNumAcceptedTokens != 0) && (tilingData_->width == 4);
 }
 
 #include "causal_conv1d_fn_tasks.h"
 
 #undef CAUSAL_CONV1D_CLASS
 #undef CAUSAL_CONV1D_TEMPLATE_ARGS

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H
