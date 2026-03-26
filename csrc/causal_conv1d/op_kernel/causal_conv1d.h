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
 * \brief CausalConv1D (prefill/extend) AscendC kernel implementation.
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_common.h"

// #define ENABLE_CAUSAL_CONV1D_DEBUG

#ifdef ENABLE_CAUSAL_CONV1D_DEBUG
#define CCONV_PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define CCONV_PRINTF(fmt, ...)
#endif

#define CCONV_PRINT_IF(cond, fmt, ...)   \
    do {                                 \
        if (cond) {                      \
            CCONV_PRINTF(fmt, ##__VA_ARGS__); \
        }                                \
    } while (0)

#ifdef ENABLE_CAUSAL_CONV1D_DEBUG

#define CCONV_DUMP_TENSOR_IF(cond, tensor, size) \
    do {                                         \
        if (cond) {                              \
            DumpTensor(tensor, __LINE__, size);  \
        }                                        \
    } while (0)
#else
constexpr int32_t CCONV_DBG_SEQ = -1;
constexpr int32_t CCONV_DBG_C0 = -1;
constexpr int32_t CCONV_DBG_MAX_TOKENS = 0;
constexpr int32_t CCONV_DBG_VERBOSE_TOKENS = 0;
constexpr int32_t CCONV_DBG_DUMP_SIZE = 0;
constexpr bool CCONV_DBG_PRINT_SYNC = false;
constexpr bool CCONV_DBG_DUMP_WEIGHTS = false;
constexpr bool CCONV_DBG_DUMP_BIAS = false;
constexpr bool CCONV_DBG_DUMP_INIT_RING = false;
constexpr bool CCONV_DBG_DUMP_RUNSEQ = false;
constexpr bool CCONV_DBG_DUMP_PREFETCH = false;
constexpr bool CCONV_DBG_DUMP_STATE = false;

#define CCONV_DUMP_TENSOR_IF(cond, tensor, size) \
    do {                                         \
    } while (0)
#endif

namespace NsCausalConv1d {

using namespace AscendC;
using namespace NsCausalConv1dCommon;
using sglang::npu_kernel::CausalConv1dTilingData;

template <typename T>
class CausalConv1d
{
public:
    __aicore__ inline CausalConv1d() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y,
                                const CausalConv1dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                    int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                          int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID tempVToMte2Event_;
    TEventID tempMte2ToVEvent_;
    TEventID inputMte2ToVEvent_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> cacheIndicesGm;
    GlobalTensor<bool> hasInitialStateGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData* tilingData_ {nullptr};
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                            GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                            GM_ADDR y, const CausalConv1dTilingData* tilingData)
{
    tilingData_ = tilingData;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    if (tilingData_->hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(convStates));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cacheIndices));
    hasInitialStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool*>(hasInitialState));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    pipe.InitBuffer(inBuf, RING_SLOTS * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * MAX_BLOCK_DIM * sizeof(T));
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float));

    AllocEvents();

    CCONV_PRINT_IF(GetBlockIdx() == 0U, "[Init] dim=%d, dimTileSize=%d, blocksPerSeq=%d, batch=%d\n",
                   tilingData_->dim, tilingData_->dimTileSize, tilingData_->blocksPerSeq, tilingData_->batch);
    CCONV_PRINT_IF(GetBlockIdx() == 0U, "[Init] hasBias=%d, activationMode=%d, stateLen=%d, inputMode=%d\n",
                   tilingData_->hasBias, tilingData_->activationMode, tilingData_->stateLen, tilingData_->inputMode);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
{
    tempVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    tempMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    inputMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(tempVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg)
{
    const int32_t dim = tilingData_->dim;
    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;
    (void)dbgSync;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<T> tempT = outBuf.Get<T>();

    CCONV_PRINT_IF(dbg, "[LoadWeightAndBias] c0=%d, dimTileSize=%d\n", c0, dimTileSize);

    for (int32_t j = 0; j < MAX_WIDTH; ++j) {
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, weightGm[weightOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(weightF[j * MAX_BLOCK_DIM], tempT, RoundMode::CAST_NONE, dimTileSize);
        PipeBarrier<PIPE_ALL>();
        if (dbg && CCONV_DBG_DUMP_WEIGHTS) {
            CCONV_PRINTF("[Dump][weightF] j=%d\n", j);
            CCONV_DUMP_TENSOR_IF(true, weightF[j * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        }
    }

    if (tilingData_->hasBias != 0) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(tempT, biasGm[c0], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        Cast(biasF, tempT, RoundMode::CAST_NONE, dimTileSize);
        PipeBarrier<PIPE_ALL>();
        if (dbg && CCONV_DBG_DUMP_BIAS) {
            CCONV_PRINTF("[Dump][biasF]\n");
            CCONV_DUMP_TENSOR_IF(true, biasF, CCONV_DBG_DUMP_SIZE);
        }
    } else {
        Duplicate(biasF, 0.0f, dimTileSize);
        CCONV_PRINT_IF(dbg, "[LoadWeightAndBias] bias=0 (no bias)\n");
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len,
                                                 int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    LocalTensor<T> ring = inBuf.Get<T>();

    CCONV_PRINT_IF(dbg, "[InitRing] cacheIdx=%d, hasInit=%d, start=%d, len=%d, c0=%d\n",
                   cacheIdx, hasInit ? 1 : 0, start, len, c0);

    PipeBarrier<PIPE_ALL>();
    if (hasInit) {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(i) * dim + c0;
            DataCopy(ring[i * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize);
            if (dbg && CCONV_DBG_DUMP_INIT_RING) {
                CCONV_PRINTF("[Dump][init_state] pos=%d\n", i);
                CCONV_DUMP_TENSOR_IF(true, ring[i * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
            }
        }
    } else {
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), dimTileSize);
        }
        CCONV_PRINT_IF(dbg, "[InitRing] ring slots 0-2 zeroed (no init state)\n");
        if (dbg && CCONV_DBG_DUMP_INIT_RING) {
            for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
                CCONV_PRINTF("[Dump][ring_zero] slot=%d\n", i);
                CCONV_DUMP_TENSOR_IF(true, ring[i * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();

    if (len > 0) {
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(ring[SlotCurr(0) * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
        PipeBarrier<PIPE_ALL>();
        CCONV_PRINT_IF(dbg, "[InitRing] x[0] loaded to slot %d\n", SlotCurr(0));
        if (dbg && CCONV_DBG_DUMP_INIT_RING) {
            CCONV_PRINTF("[Dump][ring_x0] slot=%d\n", SlotCurr(0));
            CCONV_DUMP_TENSOR_IF(true, ring[SlotCurr(0) * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        }
    }

}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize,
                                               int32_t dim, bool dbg)
{
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * MAX_BLOCK_DIM];
    LocalTensor<float> accF = biasF[MAX_BLOCK_DIM];
    LocalTensor<float> tmpF = accF[MAX_BLOCK_DIM];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;
    (void)dbgSync;
    const bool hasActivation = (tilingData_->activationMode != 0);
    const int32_t dbgMaxTokens = CCONV_DBG_MAX_TOKENS;
    const int32_t dbgVerboseTokens = CCONV_DBG_VERBOSE_TOKENS;

    for (int32_t t = 0; t < len; ++t) {
        const bool dbgTok = dbg && (t < dbgMaxTokens);
        const bool dbgVerbose = dbg && CCONV_DBG_DUMP_RUNSEQ && (t < dbgVerboseTokens);
        const bool dbgStep = dbgVerbose && (t == 0);
        const int32_t slotCurr = SlotCurr(t);
        const int32_t slotH1 = SlotHist(t, 1);
        const int32_t slotH2 = SlotHist(t, 2);
        const int32_t slotH3 = SlotHist(t, 3);
        const int32_t slotPref = (t + 1 < len) ? SlotPrefetch(t) : -1;
        const int32_t outSlot = t & 1;

        CCONV_PRINT_IF(dbgTok,
                       "[RunSeq][t=%d] curr=%d h1=%d h2=%d h3=%d pref=%d outSlot=%d\n",
                       t, slotCurr, slotH1, slotH2, slotH3, slotPref, outSlot);

        if (dbgVerbose) {
            CCONV_PRINTF("[Dump][ring_x] t=%d slot=%d\n", t, slotCurr);
            CCONV_DUMP_TENSOR_IF(true, ring[slotCurr * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        }

        if (t + 1 < len) {
            const int64_t xOffset = static_cast<int64_t>(start + t + 1) * dim + c0;
            PipeBarrier<PIPE_ALL>();
            DataCopy(ring[slotPref * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);
            PipeBarrier<PIPE_ALL>();
            CCONV_PRINT_IF(dbgTok, "[RunSeq][t=%d] prefetch x[%d] -> slot %d\n", t, t + 1, slotPref);
            if (dbgVerbose && CCONV_DBG_DUMP_PREFETCH) {
                CCONV_PRINTF("[Dump][ring_prefetch] t=%d slot=%d\n", t, slotPref);
                CCONV_DUMP_TENSOR_IF(true, ring[slotPref * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
            }
        }

        DataCopy(accF, biasF, dimTileSize);
        if (dbgStep) {
            CCONV_PRINTF("[Dump][acc_init] (bias)\n");
            CCONV_DUMP_TENSOR_IF(true, accF, CCONV_DBG_DUMP_SIZE);
        }

        for (int32_t j = 0; j < MAX_WIDTH; ++j) {
            const int32_t tap = (MAX_WIDTH - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            PipeBarrier<PIPE_ALL>();
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize);
            PipeBarrier<PIPE_ALL>();
            if (tap == (MAX_WIDTH - 1)) {
                SetFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
            }
            if (dbgStep) {
                CCONV_PRINTF("[Dump][x_cast] j=%d tap=%d slot=%d\n", j, tap, slot);
                CCONV_DUMP_TENSOR_IF(true, tmpF, CCONV_DBG_DUMP_SIZE);
            }
            PipeBarrier<PIPE_ALL>();
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);
            PipeBarrier<PIPE_ALL>();
            if (dbgStep) {
                CCONV_PRINTF("[Dump][acc] after j=%d\n", j);
                CCONV_DUMP_TENSOR_IF(true, accF, CCONV_DBG_DUMP_SIZE);
            }
        }

        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);
            if (dbgStep) {
                CCONV_PRINTF("[Dump][act] silu\n");
                CCONV_DUMP_TENSOR_IF(true, tmpF, CCONV_DBG_DUMP_SIZE);
            }
        }

        PipeBarrier<PIPE_ALL>();
        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], tmpF, dimTileSize);
            } else {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outT[outSlot * MAX_BLOCK_DIM], tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outT[outSlot * MAX_BLOCK_DIM], accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }
        PipeBarrier<PIPE_ALL>();

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;
        CCONV_PRINT_IF(dbgTok, "[RunSeq][t=%d] outOffset=%lld\n", t, (long long)outOffset);
        if (dbgVerbose) {
            CCONV_PRINTF("[Dump][outT] t=%d outSlot=%d\n", t, outSlot);
            CCONV_DUMP_TENSOR_IF(true, outT[outSlot * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(yGm[outOffset], outT[outSlot * MAX_BLOCK_DIM], dimTileSize);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0,
                                                       int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    if (len <= 0) {
        return;
    }

    CCONV_PRINT_IF(dbg, "[WriteBackState] cacheIdx=%d, len=%d, c0=%d\n", cacheIdx, len, c0);

    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    for (int32_t pos = 0; pos < (MAX_WIDTH - 1); ++pos) {
        const int32_t tap = (MAX_WIDTH - 2) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        CCONV_PRINT_IF(dbg, "[WriteBackState] pos=%d, tap=%d, slot=%d\n", pos, tap, slot);
        const int64_t stateOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                    static_cast<int64_t>(pos) * dim + c0;
        if (dbg && CCONV_DBG_DUMP_STATE) {
            CCONV_PRINTF("[Dump][state] pos=%d slot=%d\n", pos, slot);
            CCONV_DUMP_TENSOR_IF(true, ring[slot * MAX_BLOCK_DIM], CCONV_DBG_DUMP_SIZE);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::Process()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t inputMode = tilingData_->inputMode;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t dimTileSize = static_cast<int32_t>(tilingData_->dimTileSize);
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_->blocksPerSeq);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    CCONV_PRINT_IF(blockIdx == 0U,
                   "[Process] blockIdx=%u, blockNum=%u, dim=%d, batch=%d, dimTileSize=%d, blocksPerSeq=%d\n",
                   blockIdx, blockNum, dim, batch, dimTileSize, blocksPerSeq);

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || blocksPerSeq * dimTileSize != dim) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        const bool dbg = (seq == CCONV_DBG_SEQ) && (c0 == CCONV_DBG_C0);

        CCONV_PRINT_IF(dbg, "[Debug] task=%lld, seq=%d, dimBlockId=%d, c0=%d\n",
                       (long long)task, seq, dimBlockId, c0);
        CCONV_PRINT_IF(dbg, "[Debug] dbgMaxTokens=%d dbgVerboseTokens=%d dumpSize=%d\n",
                       CCONV_DBG_MAX_TOKENS, CCONV_DBG_VERBOSE_TOKENS, CCONV_DBG_DUMP_SIZE);

        LoadWeightAndBias(c0, dimTileSize, dbg);

        int32_t start = 0;
        int32_t len = 0;
        if (inputMode == 0) {
            const int32_t startVal = queryStartLocGm.GetValue(seq);
            const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
            start = startVal;
            len = endVal - startVal;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }
        CCONV_PRINT_IF(dbg, "[Process] start=%d, len=%d, inputMode=%d\n", start, len, inputMode);

        if (len <= 0) {
            continue;
        }

        const int32_t cacheIdx = cacheIndicesGm.GetValue(seq);
        if (cacheIdx == tilingData_->padSlotId) {
            continue;
        }

        const bool hasInit = hasInitialStateGm.GetValue(seq);
        CCONV_PRINT_IF(dbg, "[Process] cacheIdx=%d, hasInit=%d\n", cacheIdx, hasInit ? 1 : 0);

        InitRing(cacheIdx, hasInit, start, len, c0, dimTileSize, dim, dbg);
        RunSeq(start, len, c0, dimTileSize, dim, dbg);
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim, dbg);
    }

    ReleaseEvents();
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_H
