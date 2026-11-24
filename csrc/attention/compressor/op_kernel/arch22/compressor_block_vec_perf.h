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
 * \file compressor_block_vec_perf.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_PERF_H
#define COMPRESSOR_BLOCK_VEC_PERF_H

#include "compressor_comm.h"
#include "compressor_tools.h"
#include "compressor_vector_comm.h"
#include "rms_norm.h"
#include "rope.h"
#include "soft_max.h"

using namespace AscendC;

namespace Compressor {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

struct LoopInfo {
    uint32_t groupSize = 0U;
    uint32_t groupNum = 0U;
    uint32_t coreRowIdx = 0U;
    uint32_t coreColIdx = 0U;
    bool isCoreRowFirst = false;
    bool isCoreRowLast = false;
    bool isCoreLoopFirst = false;
    bool isCoreLoopLast = false;
};

struct Vec1SplitInfo {
    uint32_t dealSeqStartIdx = 0;
    uint32_t dBaseSize = 0;
    uint32_t vec1GroupSize = 0;
    uint32_t vec1GroupNum = 0;
    uint32_t dealTcSize = 0;
    uint32_t preDealTcSize = 0;
    uint32_t curBStart = 0;
    uint32_t curSStart = 0;
    uint32_t curCompressedCnt = 0;
    uint32_t totalCompressedCnt = 0;
    uint32_t tcSplitSize = 0;
    uint32_t dSplitSize = 0;
    uint32_t dLoopCount = 0;
    bool isWrapped = false;
};

template <typename COMP>
class CompressorBlockVectorPerf
{
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024;  // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    float SOFTMAX_MIN_NUM = static_cast<float>(-1.0 / 0.0);
    // =================================Type Definitions=================================
    // intermediate computation data type is float, high-precision mode
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVectorPerf(){};
    // =================================Set Parameters=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const CompressorTools<COMP> &tools);
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate,
                                __gm__ uint8_t *stateCache, __gm__ uint8_t *ape, __gm__ uint8_t *normWeight,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                __gm__ uint8_t *cmpKvOut);
    // =================================Resource Management=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================Execute Computation=================================
    __aicore__ inline void ComputeVec1(const Vec1RunInfo &info);
    __aicore__ inline void CommitC128State(const Vec1RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline void GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart,
                                        uint32_t v2TcEnd, uint32_t &outputBStart, uint32_t &outputSStart,
                                        uint32_t &outputScSize);
    __aicore__ inline void CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd,
                                        uint32_t &scEnd);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm,
                                                GlobalTensor<T> kvCacheTcGm, GlobalTensor<T> scoreCacheTcGm,
                                                GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void ComputeVec2(const Vec2RunInfo &info);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> vec2InputGm_;
    GlobalTensor<T> scoreMm1ResGm_;
    GlobalTensor<T> kvMm1ResGm_;
    GlobalTensor<T> kvCacheTcGm_;
    GlobalTensor<T> scoreCacheTcGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetBsLength(uint32_t index);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd,
                                             uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize,
                                           uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const Vec1RunInfo &info, CompressorVec1SliceIterator<COMP> &sliceIterator,
                                             const LoopInfo &loopInfo, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t dBaseSize);
    __aicore__ inline void CommitC128StateBaseBlock(const Vec1RunInfo &info,
                                                    CompressorVec1SliceIterator<COMP> &sliceIterator,
                                                    uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                         const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    __aicore__ inline void AddSingleApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
                                               const Vec1SliceInfo &sliceInfo, uint32_t dDealSize);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O> dstLocal, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O> dstLocal, const GlobalTensor<O> srcGm,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                               uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                               uint32_t dstSingleRowCount);
    template <typename O>
    __aicore__ inline void DataCopyWithOutputQue(const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal,
                                                 uint32_t copyRowCount, uint32_t copyColCount,
                                                 uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void PadAlign(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    template <bool IS_SCORE>
    __aicore__ inline void OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
                                   const GlobalTensor<T> &srcGm, const GlobalTensor<T> &stateGm,
                                   const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
                                   const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                   uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void FromWokrSpaceToUb(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGm,
                                             const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo,
                                             uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
                                             const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx,
                                             uint32_t endSeqIdx, uint32_t dStartIdx, uint32_t dDealSize,
                                             uint32_t stateIdx);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state,
                                              const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx,
                                              uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStartIdx,
                                              uint32_t dDealSize, uint32_t stateIdx);
    __aicore__ inline void SaveToWorkSpace(const LocalTensor<T> srcLocal, const GlobalTensor<T> &cacheTcGm,
                                           const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
                                           uint32_t dDealSize);
    __aicore__ inline void LoadFromWorkSpace(const LocalTensor<T> dstLocal, const GlobalTensor<T> &cacheTcGm,
                                             const GlobalTensor<T> &srcGm, const LocalTensor<T> srcLocal,
                                             const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo,
                                             uint32_t dStartIdx, uint32_t globalSeqIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                     uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T> &scoreLocal,
                                            const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb,
                                            uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void OverLapScoreKv(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal,
                                          const Vec1RunInfo &info, const LoopInfo &loopInfo,
                                          const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo,
                                          uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize,
                                          uint32_t needDealTcSize);
    __aicore__ inline void CopyOutVec1Res(const GlobalTensor<T> &resGm, const Vec1RunInfo &info,
                                          const LocalTensor<T> comperssoredUb, uint32_t compressTcSize,
                                          uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTaskDistribution(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void UpdateIteratorState(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo);
    __aicore__ inline void CalcTilingStrategy(Vec1SplitInfo &splitInfo);
    __aicore__ inline Vec1SplitInfo SplitCoreV1(const Vec1RunInfo &info);
    __aicore__ inline void SplitCoreV2(const Compressor::Vec2RunInfo &info);
    __aicore__ inline void CopyFinalResultOut(const Compressor::Vec2RunInfo &info, const LocalTensor<X_T> &cmpKvOutUb,
                                              uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void DealVec2BaseBlock(const Compressor::Vec2RunInfo &info, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb,
                                          const LocalTensor<T> &normWeightUb, const LocalTensor<T> &tempLocal,
                                          uint32_t dealRowCount);
    __aicore__ inline void SingleCalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
                                         uint32_t rowCnt, uint32_t curDealScSize, uint32_t globalScStart);
    __aicore__ inline void CalRope(const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
                                   uint32_t dealRowCount);
    __aicore__ inline void SaveState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    template <bool IS_SCORE>
    __aicore__ inline void DuplicateFirstBlock(const LocalTensor<T> &dstLocal, uint32_t duplicateRowCount,
                                               uint32_t duplicateColCount, uint32_t singleRowCount);
    template <bool IS_SCORE>
    __aicore__ inline void ReadState(const LocalTensor<T> &srcLocal, const GlobalTensor<T> &stateGm,
                                     const GlobalTensor<int32_t> &blockTableGm, const Vec1SliceInfo &sliceInfo,
                                     uint32_t dStartIdx, uint32_t dDealSize, uint32_t stateIdx);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t compressedCnt_ = 0;
    uint32_t v1SplitSize_ = 0;
    uint32_t v1ScLoopTimes_ = 0;
    uint32_t v1DLoopTimes_ = 0;
    uint32_t dealTcNum_ = 0;
    bool apeIsLoad_ = false;
    bool isExistSeqUsed = false;
    bool isExistStartPos = false;
    // vec2
    uint32_t v2MBaseSize = 16;  // number of Tc blocks: 32 * 1024 / (512 * 4)
    uint32_t v2TcStartIdx = 0U;
    uint32_t v2TcEndIdx = 0U;
    uint32_t mmResColSize_ = 128;
    int64_t vec1ResGmStart = 0U;
    uint32_t OutputBStartIdx, OutputSStartIdx, OutputSize;
    CompressorTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> stateBlockTableGm_;
    GlobalTensor<T> stateCacheGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<T> normWeightGm_;
    GlobalTensor<T> ropeSinGm_;
    GlobalTensor<T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer====================================
    // TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> mm1ResTensor;
    LocalTensor<T> leftStateTensor;
    LocalTensor<T> rightStateTensor;
    LocalTensor<T> normWeightUb;
    LocalTensor<T> apeUb;
    LocalTensor<uint32_t> gatherOffsetCastUb;
    // temporary tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;
    TBuf<TPosition::VECCALC> apeBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TBuf<TPosition::VECIN> normWeightBuf;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitParams(const ConstInfo &constInfo,
                                                                   const CompressorTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::Init(
    __gm__ uint8_t *x, __gm__ uint8_t *wKv, __gm__ uint8_t *wGate, __gm__ uint8_t *stateCache, __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos, __gm__ uint8_t *cmpKvOut)
{
    stateBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
    stateCacheGm_.SetGlobalBuffer((__gm__ T *)stateCache);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    normWeightGm_.SetGlobalBuffer((__gm__ T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
    coff_ = static_cast<uint32_t>(COMP::coff);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(gatherOffsetBuf, BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_32K);
    normWeightUb = normWeightBuf.Get<T>();
    apeUb = apeBuf.Get<T>();
    LocalTensor<T> normweightInUb = inputQue1.AllocTensor<T>();
    LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf.Get<int32_t>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim);  // get normWeight, kept resident
    inputQue1.EnQue(normweightInUb);
    inputQue1.DeQue<T>();
    DataCopy(normWeightUb, normweightInUb, constInfo_.headDim);
    inputQue1.FreeTensor(normweightInUb);
    if constexpr (COMP::rotaryMode == Compressor::ROTARY_MODE::INTERLEAVE) {
        SetGatherSrcOffset<float>(gatherOffsetUb, constInfo_.ropeHeadDim);
    }
    gatherOffsetCastUb = gatherOffsetUb.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::AllocEventID()
{}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::FreeEventID()
{}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitVec1GlobalTensor(
    GlobalTensor<T> kvMm1ResGm, GlobalTensor<T> scoreMm1ResGm, GlobalTensor<T> kvCacheTcGm,
    GlobalTensor<T> scoreCacheTcGm, GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm)
{
    this->kvMm1ResGm_ = kvMm1ResGm;
    this->scoreMm1ResGm_ = scoreMm1ResGm;
    this->kvCacheTcGm_ = kvCacheTcGm;
    this->scoreCacheTcGm_ = scoreCacheTcGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec2InputGm_ = vec2InputGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return sequsedGm_.GetValue(bIdx);
    } else if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetBsLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index);
    } else {
        return index * constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetBasicNum()
{
    // get the number of basic units Tc in the m direction
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // add the middle full block and the tail block
    curBasicNum += CeilDivT(curActSeqLength_ - headSize, constInfo_.cmpRatio);
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetScSize()
{
    uint32_t curBasicNum = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    return curBasicNum;
}

// compute the Tc start and end indices
template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcScEndIdx(uint32_t bStart, uint32_t scStart,
                                                                     uint32_t dealScSize, uint32_t &bEnd,
                                                                     uint32_t &scEnd)
{
    uint32_t accScSize = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        // compute the remaining blocks of the starting batch
        if (bIdx == bStart) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            accScSize += GetScSize() - scStart;
            if (accScSize >= dealScSize) {
                scEnd = scStart + dealScSize;
                return;
            }
        } else {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetScSize();
            uint32_t curBasicNumEnd = dealScSize - accScSize;

            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = curBasicNumEnd;
                return;
            }
            accScSize += curBasicNum;
        }
    }
}

// compute the b and sc indices for vec output based on the sc start index
template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::GetScIdxInfo(uint32_t bStart, uint32_t scStart,
                                                                     uint32_t dealScSize, uint32_t v2TcStart,
                                                                     uint32_t v2TcEnd, uint32_t &outputBStart,
                                                                     uint32_t &outputSStart, uint32_t &outputScSize)
{
    outputScSize = v2TcEnd - v2TcStart;
    uint32_t scEnd = 0;
    uint32_t bEnd = 0;
    CalcScEndIdx(bStart, scStart, v2TcStart, bEnd, scEnd);
    outputSStart = scEnd;
    outputBStart = bEnd;
    // handle batch jumping
    curActSeqLength_ = GetSeqLength(bEnd);
    curStartPos_ = GetStartPos(bEnd);
    uint32_t curScSize = GetScSize();
    if (curScSize == scEnd) {
        outputSStart = 0;
        outputBStart++;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx,
                                                                  uint32_t dDealSize)
{
    LocalTensor<T> apeUbTmp = inputQue1.AllocTensor<T>();

    uint32_t copyRowCount = coff_ * constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = dDealSize;
    uint32_t srcSingleRowCount = constInfo_.headDim;

    uint64_t gmOffset = dStartIdx;
    DataCopyAlignGmToUb(apeUbTmp, apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    inputQue1.EnQue(apeUbTmp);
    inputQue1.DeQue<T>();
    DataCopy(apeUb, apeUbTmp, coff_ * dDealSize * constInfo_.cmpRatio);
    inputQue1.FreeTensor(apeUbTmp);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal,
                                                                      const LocalTensor<T> &apeUb,
                                                                      const Vec1SliceInfo &sliceInfo,
                                                                      uint32_t dDealSize)
{
    uint32_t singleRowElemNum = dDealSize * coff_;
    uint64_t scoreOffset = sliceInfo.dealedSeqCnt * singleRowElemNum;

    uint32_t tcDealSize = sliceInfo.dealTcSize;
    if (sliceInfo.headHolderSeqCnt > 0) {
        uint64_t apeOffset = sliceInfo.headHolderSeqCnt * singleRowElemNum;
        uint32_t rCnt = tcDealSize == 1 ? sliceInfo.validSeqCnt * singleRowElemNum
                                        : (constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt) * singleRowElemNum;
        Add(scoreLocal[scoreOffset], scoreLocal[scoreOffset], apeUb[apeOffset], rCnt);
        scoreOffset += rCnt;
        tcDealSize -= 1;
    }
    if (tcDealSize == 0) {
        return;
    }
    if (sliceInfo.tailHolderSeqCnt > 0) {
        tcDealSize -= 1;
        uint64_t apeOffset = 0;
        uint32_t rCnt = (constInfo_.cmpRatio - sliceInfo.tailHolderSeqCnt) * singleRowElemNum;
        uint32_t tailScoreOffset = scoreOffset + tcDealSize * constInfo_.cmpRatio * singleRowElemNum;
        Add(scoreLocal[tailScoreOffset], scoreLocal[tailScoreOffset], apeUb[apeOffset], rCnt);
    }
    if (tcDealSize == 0) {
        return;
    }
    uint32_t rCnt = constInfo_.cmpRatio * singleRowElemNum;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        Add(scoreLocal[scoreOffset + r * rCnt], scoreLocal[scoreOffset + r * rCnt], apeUb, rCnt);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::AddSingleApeToScore(const LocalTensor<T> &scoreLocal,
                                                                            const LocalTensor<T> &apeUb,
                                                                            const Vec1SliceInfo &sliceInfo,
                                                                            uint32_t dDealSize)
{
    uint32_t SingleRowElemNum = dDealSize * coff_;
    uint32_t dealRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t scoreOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    uint64_t apeOffset = (constInfo_.cmpRatio - dealRowCount) * SingleRowElemNum;
    for (uint32_t dOffset = 0; dOffset < dDealSize; dOffset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curAddColCount = min(dDealSize - dOffset, FP32_REPEAT_ELEMENT_NUM);
        Add(scoreLocal[scoreOffset + dOffset], scoreLocal[scoreOffset + dOffset], apeUb[apeOffset + dOffset],
            curAddColCount, dealRowCount,
            {1, 1, 1, static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM),
             static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM),
             static_cast<uint8_t>(SingleRowElemNum / FP32_BLOCK_ELEMENT_NUM)});
    }
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyAlignUbToUb(
    const LocalTensor<O> dstLocal, const LocalTensor<O> srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyAlignGmToUb(
    const LocalTensor<O> dstLocal, const GlobalTensor<O> srcGm, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyAlignUbToGm(
    const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstGm, srcLocal, intriParams);
}

template <typename COMP>
template <typename O>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyWithOutputQue(
    const GlobalTensor<O> dstGm, const LocalTensor<O> srcLocal, uint32_t copyRowCount, uint32_t copyColCount,
    uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return;
    }
    uint32_t singleCopyRowCount = BUFFER_SIZE_BYTE_16K / (copyColCount * sizeof(O));
    for (uint32_t rowCount = 0; rowCount < copyRowCount; rowCount += singleCopyRowCount) {
        uint64_t srcOffset = rowCount * srcSingleRowCount;
        uint64_t dstOffset = rowCount * dstSingleRowCount;
        uint32_t curCopyRowCount = min(singleCopyRowCount, copyRowCount - rowCount);

        LocalTensor<O> outputUb = outputQue1.AllocTensor<O>();

        DataCopyAlignUbToUb(outputUb, srcLocal[srcOffset], curCopyRowCount, copyColCount, srcSingleRowCount,
                            copyColCount);
        PipeBarrier<PIPE_V>();

        outputQue1.EnQue(outputUb);
        outputQue1.DeQue<O>();

        DataCopyAlignUbToGm(dstGm[dstOffset], outputUb, curCopyRowCount, copyColCount, copyColCount, dstSingleRowCount);
        outputQue1.FreeTensor(outputUb);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::PadAlign(const LocalTensor<T> dstLocal,
                                                                 const LocalTensor<T> srcLocal,
                                                                 const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                                 uint32_t dDealSize)
{
    // Ub data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = sliceInfo.compressTcSize * constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = srcSingleRowElemNum;  // left and right are stored interleaved in the seq direction
    uint64_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum;

    uint64_t dstUbOffset = sliceInfo.compressoredScCnt * constInfo_.cmpRatio * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // left side
        uint64_t preSrcLocalOffset = srcLocalOffset;
        uint64_t preDstUbOffset = dstUbOffset + (sliceInfo.headHolderSeqCnt + constInfo_.cmpRatio) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
                            copyRowCount - min(copyRowCount, constInfo_.cmpRatio), copyColCount, srcSingleRowCount,
                            dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += dDealSize;
    }
    // right side
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset], copyRowCount, copyColCount, srcSingleRowCount,
                        dstSingleRowCount);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::OverLap(
    const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal, const GlobalTensor<T> &srcGm,
    const GlobalTensor<T> &stateGm, const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<T> &cacheTcGm,
    const Vec1RunInfo &info, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.dealTcSize == 0) {
        return;
    }

    if constexpr (IS_SCORE) {
        AddApeToScore(srcLocal, apeUb, sliceInfo, dDealSize);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (COMP::cacheMode == CACHE_MODE::EXPLICIT) {
        if constexpr (COMP::coff == COFF::OVERLAP) {
            if (!sliceInfo.isWrapped) {
                SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                          static_cast<uint32_t>(IS_SCORE));
            }
        } else {
            // C128 explicit locations may map the history head and the new tail
            // to the same physical ring rows. Defer the tail commit until the
            // kernel-level SyncAll, so every ReadState observes the old history.
        }
    } else {
        SaveState(srcLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize, static_cast<uint32_t>(IS_SCORE));
    }

    event_t eventId_V_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventId_V_MTE2);
    ReadState<IS_SCORE>(dstLocal, stateGm, blockTableGm, sliceInfo, dStartIdx, dDealSize,
                        static_cast<uint32_t>(IS_SCORE));

    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t nextC1V1DbIdx = (info.c1v1DbIdx + 1) % constInfo_.dbWorkspaceRatio;
        GlobalTensor<T> nextCacheTcGm = cacheTcGm[nextC1V1DbIdx * constInfo_.cmpRatio * constInfo_.headDim];
        SaveToWorkSpace(srcLocal, nextCacheTcGm, sliceInfo, loopInfo, dStartIdx, dDealSize);
    }
    if (sliceInfo.compressTcSize > 0) {
        PadAlign(dstLocal, srcLocal, sliceInfo, dStartIdx, dDealSize);
        if constexpr (COMP::coff == COFF::OVERLAP) {
            event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
            GlobalTensor<T> curCacheTcGm = cacheTcGm[info.c1v1DbIdx * constInfo_.cmpRatio * constInfo_.headDim];
            LoadFromWorkSpace(dstLocal, curCacheTcGm, srcGm, srcLocal, sliceInfo, loopInfo, dStartIdx, globalSeqIdx,
                              dDealSize);
        }
    }
    event_t eventId_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::FromWokrSpaceToUb(const LocalTensor<T> &dstLocal,
                                                                          const GlobalTensor<T> &srcGm,
                                                                          const Vec1SliceInfo &sliceInfo,
                                                                          const StatisticInfo &statisticInfo,
                                                                          uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.headDim;
    uint32_t copyRowCount = statisticInfo.dealSeqCnt * coff_;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize;
    uint64_t srcGmOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum * coff_ + dStartIdx;
    DataCopyAlignGmToUb(dstLocal, srcGm[srcGmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SaveToWorkSpace(const LocalTensor<T> srcLocal,
                                                                        const GlobalTensor<T> &cacheTcGm,
                                                                        const Vec1SliceInfo &sliceInfo,
                                                                        const LoopInfo &loopInfo, uint32_t dStartIdx,
                                                                        uint32_t dDealSize)
{
    uint32_t curSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt;
    uint32_t totalSeqLen = sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.bSeqUsed;
    if (!loopInfo.isCoreRowLast || !loopInfo.isCoreLoopLast || !sliceInfo.isLast || totalSeqLen < constInfo_.cmpRatio ||
        curSeqLen > Trunc(totalSeqLen, constInfo_.cmpRatio) - constInfo_.cmpRatio) {
        return;
    }
    uint32_t srcSingleRowElemNum = dDealSize * coff_;
    uint64_t srcLocalOffset =
        (sliceInfo.dealedSeqCnt + sliceInfo.validSeqCnt - min(sliceInfo.validSeqCnt, constInfo_.cmpRatio)) *
        srcSingleRowElemNum;
    DataCopyWithOutputQue(cacheTcGm[dStartIdx], srcLocal[srcLocalOffset],
                          curSeqLen - max(curSeqLen - constInfo_.cmpRatio, sliceInfo.bStartPos), dDealSize,
                          coff_ * dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::LoadFromWorkSpace(
    const LocalTensor<T> dstLocal, const GlobalTensor<T> &cacheTcGm, const GlobalTensor<T> &srcGm,
    const LocalTensor<T> srcLocal, const Vec1SliceInfo &sliceInfo, const LoopInfo &loopInfo, uint32_t dStartIdx,
    uint32_t globalSeqIdx, uint32_t dDealSize)
{
    if (sliceInfo.sIdx == 0) {
        return;
    }
    uint32_t dstSingleRowElemNum = dDealSize * coff_;
    uint32_t copyRowCount = min(sliceInfo.sIdx, constInfo_.cmpRatio);
    uint64_t dstLocalOffset =
        (sliceInfo.compressoredScCnt * constInfo_.cmpRatio + constInfo_.cmpRatio - copyRowCount) * dstSingleRowElemNum;
    if (loopInfo.isCoreRowFirst && loopInfo.isCoreLoopFirst && sliceInfo.isFirst) {  // get from cacheGm
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset = dStartIdx;
        DataCopyAlignGmToUb(dstLocal[dstLocalOffset], cacheTcGm[srcLocalOffset], copyRowCount, dDealSize,
                            constInfo_.headDim, coff_ * dDealSize);
    } else if (sliceInfo.isFirst) {  // get from the WorkSpace holding the MatMul result
        uint32_t srcSingleRowElemNum = constInfo_.headDim * coff_;
        uint64_t srcLocalOffset =
            (globalSeqIdx + sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum + dStartIdx;
        DataCopyAlignGmToUb(dstLocal[dstLocalOffset], srcGm[srcLocalOffset], copyRowCount, dDealSize,
                            coff_ * constInfo_.headDim, coff_ * dDealSize);
    } else {  // get from UB
        uint32_t srcSingleRowElemNum = dDealSize * coff_;
        uint64_t srcLocalOffset = (sliceInfo.dealedSeqCnt - copyRowCount) * srcSingleRowElemNum;
        DataCopyAlignUbToUb(dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], copyRowCount, dDealSize,
                            coff_ * dDealSize, coff_ * dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
                                                                           const GlobalTensor<T> &state,
                                                                           const GlobalTensor<int32_t> &blockTableGm,
                                                                           uint32_t batchIdx, uint32_t startSeqIdx,
                                                                           uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                           uint32_t dDealSize, uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::EXPLICIT) {
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        uint64_t tableBaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
        int64_t tableColumn = static_cast<int64_t>(coff_ * constInfo_.cmpRatio) + static_cast<int64_t>(curSeqIdx) -
                              static_cast<int64_t>(GetStartPos(batchIdx));
        while (copyFinishRowCnt < seqCnt) {
            uint64_t stateLoc = static_cast<uint64_t>(blockTableGm.GetValue(tableBaseOffset + tableColumn));
            uint32_t copyRowCount = 1;
            while (copyFinishRowCnt + copyRowCount < seqCnt) {
                uint64_t nextStateLoc =
                    static_cast<uint64_t>(blockTableGm.GetValue(tableBaseOffset + tableColumn + copyRowCount));
                if (nextStateLoc != stateLoc + copyRowCount) {
                    break;
                }
                copyRowCount++;
            }
            uint64_t stateOffset =
                stateLoc * 2 * coff_ * constInfo_.headDim + stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            DataCopyAlignGmToUb(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount,
                                dDealSize, coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
            tableColumn += copyRowCount;
        }
        return;
    }

    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * 2 * coff_ * constInfo_.headDim +
                               remainRowCnt * 2 * coff_ * constInfo_.headDim + stateIdx * coff_ * constInfo_.headDim +
                               dStartIdx;

        DataCopyAlignGmToUb(output[copyFinishRowCnt * coff_ * dDealSize], state[stateOffset], copyRowCount, dDealSize,
                            coff_ * constInfo_.headDim * 2, coff_ * dDealSize);
        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::WriteToCacheState(const GlobalTensor<T> &state,
                                                                          const GlobalTensor<int32_t> &blockTableGm,
                                                                          const LocalTensor<T> &input,
                                                                          uint32_t batchIdx, uint32_t startSeqIdx,
                                                                          uint32_t endSeqIdx, uint32_t dStartIdx,
                                                                          uint32_t dDealSize, uint32_t stateIdx)
{
    if constexpr (COMP::cacheMode == CACHE_MODE::EXPLICIT) {
        uint32_t curSeqIdx = startSeqIdx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t seqCnt = endSeqIdx - startSeqIdx;
        uint64_t tableBaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
        int64_t tableColumn = static_cast<int64_t>(coff_ * constInfo_.cmpRatio) + static_cast<int64_t>(curSeqIdx) -
                              static_cast<int64_t>(GetStartPos(batchIdx));
        while (copyFinishRowCnt < seqCnt) {
            uint64_t stateLoc = static_cast<uint64_t>(blockTableGm.GetValue(tableBaseOffset + tableColumn));
            uint32_t copyRowCount = 1;
            while (copyFinishRowCnt + copyRowCount < seqCnt) {
                uint64_t nextStateLoc =
                    static_cast<uint64_t>(blockTableGm.GetValue(tableBaseOffset + tableColumn + copyRowCount));
                if (nextStateLoc != stateLoc + copyRowCount) {
                    break;
                }
                copyRowCount++;
            }
            uint64_t stateOffset =
                stateLoc * 2 * coff_ * constInfo_.headDim + stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                  dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * 2);
            copyFinishRowCnt += copyRowCount;
            curSeqIdx += copyRowCount;
            tableColumn += copyRowCount;
        }
        return;
    }

    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCount = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCount > seqCnt) {
            copyRowCount = seqCnt - copyFinishRowCnt;
        }
        if (idInBlockTable != 0) {  // 32
            uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * 2 * coff_ * constInfo_.headDim +
                                   remainRowCnt * 2 * coff_ * constInfo_.headDim +
                                   stateIdx * coff_ * constInfo_.headDim + dStartIdx;
            DataCopyWithOutputQue(state[stateOffset], input[copyFinishRowCnt * coff_ * dDealSize], copyRowCount,
                                  dDealSize, coff_ * dDealSize, coff_ * constInfo_.headDim * 2);
        }

        copyFinishRowCnt += copyRowCount;
        curSeqIdx += copyRowCount;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SaveState(const LocalTensor<T> &srcLocal,
                                                                  const GlobalTensor<T> &stateGm,
                                                                  const GlobalTensor<int32_t> &blockTableGm,
                                                                  const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                                  uint32_t dDealSize, uint32_t stateIdx)
{
    uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
    uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
    uint64_t srcBaseOffset = sliceInfo.dealedSeqCnt * coff_ * dDealSize;

    if constexpr (COMP::cacheMode == CACHE_MODE::EXPLICIT) {
        // Rows strictly before the compress boundary become c-state and do not
        // need to stay in the raw state -- EXCEPT during MTP verify, where a
        // partially-accepted round must be able to re-compress from the
        // actually-accepted offset and re-reads raw rows that sat between the
        // accepted position and this round's compress boundary. Keep those tail
        // rows the ring can hold beyond one window (mirrors sglang's
        // get_compress_state_write_pad / c_plan `mtp_pad`):
        //     pad = ring - window + 2   (0 when ring <= window)
        const uint32_t window = (coff_ == 2U ? 2U : 1U) * constInfo_.cmpRatio;
        const uint32_t pad = constInfo_.blockSize > window ? constInfo_.blockSize - window + 2U : 0U;
        const uint32_t batchEnd = sliceInfo.bStartPos + sliceInfo.bSeqUsed;
        const uint32_t compressSeqIdx = Trunc(batchEnd, constInfo_.cmpRatio);
        uint32_t writeSeqStartIdx = compressSeqIdx > (coff_ - 1U) * constInfo_.cmpRatio
                                        ? compressSeqIdx - (coff_ - 1U) * constInfo_.cmpRatio
                                        : 0U;
        if (pad != 0U) {
            writeSeqStartIdx = batchEnd > pad ? min(writeSeqStartIdx, batchEnd - pad) : 0U;
        }
        if (endSeqIdx <= writeSeqStartIdx) {
            return;
        }
        srcBaseOffset += (max(startSeqIdx, writeSeqStartIdx) - startSeqIdx) * coff_ * dDealSize;
        startSeqIdx = max(startSeqIdx, writeSeqStartIdx);
    }

    if constexpr (COMP::coff == COFF::OVERLAP) {
        WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                          dStartIdx, dDealSize, stateIdx);
        srcBaseOffset += dDealSize;
        dStartIdx += constInfo_.headDim;
    }

    WriteToCacheState(stateGm, blockTableGm, srcLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx,
                      dDealSize, stateIdx);
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DuplicateFirstBlock(const LocalTensor<T> &dstLocal,
                                                                            uint32_t duplicateRowCount,
                                                                            uint32_t duplicateColCount,
                                                                            uint32_t singleRowCount)
{
    for (uint32_t offset = 0; offset < duplicateColCount; offset += FP32_REPEAT_ELEMENT_NUM) {
        uint32_t curDuplicateColCount = min(duplicateColCount - offset, FP32_REPEAT_ELEMENT_NUM);
        if constexpr (IS_SCORE) {
            Duplicate(dstLocal[offset], SOFTMAX_MIN_NUM, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        } else {
            Duplicate(dstLocal[offset], FLOAT_ZERO, curDuplicateColCount, duplicateRowCount, 1,
                      singleRowCount / REPEAT_STRIDE_NUM);
        }
    }
}

template <typename COMP>
template <bool IS_SCORE>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ReadState(const LocalTensor<T> &dstLocal,
                                                                  const GlobalTensor<T> &stateGm,
                                                                  const GlobalTensor<int32_t> &blockTableGm,
                                                                  const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx,
                                                                  uint32_t dDealSize, uint32_t stateIdx)
{
    // when there is no block to compress, there is no need to read state information
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    // fill the right side
    if (sliceInfo.headHolderSeqCnt > 0) {
        // the first block of the entire batch
        uint32_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
        if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
            dstBaseOffset += (coff_ - 1) * dDealSize;
        }
        ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                           dStartIdx + (coff_ - 1) * constInfo_.headDim, dDealSize, stateIdx);
    }

    // fill the left side
    if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < constInfo_.cmpRatio;
        if (isFirst) {
            // no historical data
            // dDealSize must be 64
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            DuplicateFirstBlock<IS_SCORE>(dstLocal[dstBaseOffset], constInfo_.cmpRatio, dDealSize, coff_ * dDealSize);
        }
        if (sliceInfo.sIdx < constInfo_.cmpRatio && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx =
                sliceInfo.bStartPos < constInfo_.cmpRatio
                    ? 0
                    : Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio) - constInfo_.cmpRatio;
            uint32_t endSeqIdx =
                min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, constInfo_.cmpRatio) -
                        constInfo_.cmpRatio,
                    sliceInfo.bStartPos);
            uint64_t dstBaseOffset = sliceInfo.compressoredScCnt * constInfo_.cmpRatio * coff_ * dDealSize;
            if (isFirst) {
                dstBaseOffset += constInfo_.cmpRatio * coff_ * dDealSize;
            }
            ReadFromCacheState(dstLocal[dstBaseOffset], stateGm, blockTableGm, sliceInfo.bIdx, startSeqIdx, endSeqIdx,
                               dStartIdx, dDealSize, stateIdx);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SoftmaxDN(const LocalTensor<T> &scoreLocal,
                                                                  const LocalTensor<T> &tmpUb, uint32_t tcDealSize,
                                                                  uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::KvMulReduceScore(const LocalTensor<T> &kvLocal,
                                                                         const LocalTensor<T> &scoreLocal,
                                                                         const LocalTensor<T> &dstLocal,
                                                                         const LocalTensor<T> &tmpUb,
                                                                         uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = coff_ * constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    Mul(kvLocal, kvLocal, scoreLocal, tcDealSize * rCnt);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSum(dstLocal[r * dDealSize], kvLocal[r * rCnt], tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyOutVec1Res(const GlobalTensor<T> &resGm,
                                                                       const Vec1RunInfo &info,
                                                                       const LocalTensor<T> comperssoredUb,
                                                                       uint32_t compressTcSize, uint32_t dStartIdx,
                                                                       uint32_t dDealSize)
{
    uint64_t outGmOffset = compressedCnt_ * constInfo_.headDim + dStartIdx;
    DataCopyAlignUbToGm(resGm[outGmOffset], comperssoredUb, compressTcSize, dDealSize, dDealSize, constInfo_.headDim);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::OverLapScoreKv(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &kvLocal, const Vec1RunInfo &info, const LoopInfo &loopInfo,
    const StatisticInfo &statisticInfo, const Vec1SliceInfo &originSliceInfo, uint32_t dStartIdx, uint32_t dDealSize,
    uint32_t dBaseSize, uint32_t needDealTcSize)
{
    CompressorVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    overLapSliceIterator.SetRunStartBatch(info.bStart);
    Vec1SliceInfo &overLapSliceInfo = overLapSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = scoreMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(scoreUb, scoreDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(scoreUb);
    inputQue1.DeQue<T>();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetWrapped(originSliceInfo.isWrapped);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<true>(scoreLocal, scoreUb, scoreDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, scoreCacheTcGm_, info,
                      overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(scoreUb);

    if constexpr (COMP::coff == COFF::OVERLAP) {
        if (originSliceInfo.sIdx != 0 && originSliceInfo.compressTcSize > 0 &&
            (!loopInfo.isCoreRowFirst || !loopInfo.isCoreLoopFirst)) {
            AddSingleApeToScore(scoreLocal, apeUb, originSliceInfo, dDealSize);
        }
    }

    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);

    inputQue1.EnQue(kvUb);
    inputQue1.DeQue<T>();
    overLapSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetWrapped(originSliceInfo.isWrapped);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap<false>(kvLocal, kvUb, kvDBMm1ResGm, stateCacheGm_, stateBlockTableGm_, kvCacheTcGm_, info,
                       overLapSliceInfo, loopInfo, dStartIdx, originSliceInfo.dealedSeqCnt, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(kvUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DealVec1BaseBlock(
    const Vec1RunInfo &info, CompressorVec1SliceIterator<COMP> &sliceIterator, const LoopInfo &loopInfo,
    uint32_t dStartIdx, uint32_t dDealSize, uint32_t dBaseSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }
    LocalTensor<T> scoreLocal = tmpBuff1.Get<T>();
    LocalTensor<T> kvLocal = tmpBuff2.Get<T>();

    OverLapScoreKv(scoreLocal, kvLocal, info, loopInfo, statisticInfo, originSliceInfo, dStartIdx, dDealSize, dBaseSize,
                   needDealTcSize);

    if (statisticInfo.compressorScCnt > 0) {
        LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
        SoftmaxDN(scoreLocal, tmpUb, statisticInfo.compressorScCnt, dDealSize);
        LocalTensor<T> comperssoredUb = outputQue1.AllocTensor<T>();
        PipeBarrier<PIPE_V>();
        KvMulReduceScore(kvLocal, scoreLocal, comperssoredUb, tmpUb, statisticInfo.compressorScCnt, dDealSize);
        PipeBarrier<PIPE_V>();
        outputQue1.EnQue(comperssoredUb);
        outputQue1.DeQue<T>();
        GlobalTensor<T> resGm = vec1ResGm_[info.v1v2DbIdx * constInfo_.dbSize];
        CopyOutVec1Res(resGm, info, comperssoredUb, statisticInfo.compressorScCnt, dStartIdx, dDealSize);
        outputQue1.FreeTensor(comperssoredUb);
    }
    compressedCnt_ += statisticInfo.compressorScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CommitC128StateBaseBlock(
    const Vec1RunInfo &info, CompressorVec1SliceIterator<COMP> &sliceIterator, uint32_t dStartIdx, uint32_t dDealSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    if (statisticInfo.actualTcCnt == 0) {
        return;
    }

    CompressorVec1SliceIterator commitSliceIterator(tools_);
    commitSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    commitSliceIterator.SetRunStartBatch(info.bStart);
    Vec1SliceInfo &commitSliceInfo = commitSliceIterator.GetSlice();

    GlobalTensor<T> scoreDBMm1ResGm = scoreMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(scoreUb, scoreDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(scoreUb);
    inputQue1.DeQue<T>();
    commitSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    commitSliceIterator.SetWrapped(originSliceInfo.isWrapped);
    commitSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!commitSliceIterator.IsEnd()) {
        commitSliceIterator.GetSlice();
        AddApeToScore(scoreUb, apeUb, commitSliceInfo, dDealSize);
        PipeBarrier<PIPE_V>();
        SaveState(scoreUb, stateCacheGm_, stateBlockTableGm_, commitSliceInfo, dStartIdx, dDealSize, 1U);
        commitSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(scoreUb);

    GlobalTensor<T> kvDBMm1ResGm = kvMm1ResGm_[info.c1v1DbIdx * constInfo_.dbSize];
    LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(kvUb, kvDBMm1ResGm, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(kvUb);
    inputQue1.DeQue<T>();
    commitSliceIterator.Reset(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    commitSliceIterator.SetWrapped(originSliceInfo.isWrapped);
    commitSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!commitSliceIterator.IsEnd()) {
        commitSliceIterator.GetSlice();
        SaveState(kvUb, stateCacheGm_, stateBlockTableGm_, commitSliceInfo, dStartIdx, dDealSize, 0U);
        commitSliceIterator.IteratorSlice();
    }
    inputQue1.FreeTensor(kvUb);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcGroupInfo(const Vec1RunInfo &info, Vec1SplitInfo &splitInfo)
{
    uint32_t aiCoreNum = constInfo_.usedCoreNum * 2;
    splitInfo.dBaseSize = constInfo_.headDim / min(FloorPow2(aiCoreNum), CeilPow2(CeilDivT(aiCoreNum, info.dealTcNum)));
    splitInfo.dBaseSize = max(splitInfo.dBaseSize, FP32_BLOCK_ELEMENT_NUM);
    splitInfo.vec1GroupSize = constInfo_.headDim / splitInfo.dBaseSize;
    splitInfo.vec1GroupNum = min(static_cast<uint32_t>(aiCoreNum / splitInfo.vec1GroupSize), info.dealTcNum);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcTaskDistribution(const Vec1RunInfo &info,
                                                                             Vec1SplitInfo &splitInfo)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t groupSize = splitInfo.vec1GroupSize;
    uint32_t groupNum = splitInfo.vec1GroupNum;
    uint32_t dealTcNum = info.dealTcNum;

    if (blockIdx < groupSize * (dealTcNum % groupNum)) {
        splitInfo.dealTcSize = dealTcNum / groupNum + 1;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize);
    } else if (blockIdx < groupSize * groupNum) {
        splitInfo.dealTcSize = dealTcNum / groupNum;
        splitInfo.preDealTcSize = splitInfo.dealTcSize * (blockIdx / groupSize) + dealTcNum % groupNum;
    } else {
        splitInfo.dealTcSize = 0;
        splitInfo.preDealTcSize = dealTcNum;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::UpdateIteratorState(const Vec1RunInfo &info,
                                                                            Vec1SplitInfo &splitInfo)
{
    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.SetRunStartBatch(info.bStart);
    sliceIterator.Reset(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo &sliceInfo = sliceIterator.GetSlice();

    // process the preceding workload and update the start index
    if (splitInfo.preDealTcSize > 0) {
        sliceIterator.SetNeedDealTcSize(splitInfo.preDealTcSize);
        StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
        splitInfo.curCompressedCnt = statisticInfo.compressorScCnt;
        splitInfo.dealSeqStartIdx = sliceInfo.dealedSeqCnt;
        splitInfo.curBStart = sliceInfo.bIdx;
        splitInfo.curSStart = sliceInfo.sIdx;
        splitInfo.isWrapped = sliceInfo.isWrapped;
    } else {
        splitInfo.curCompressedCnt = 0;
        splitInfo.dealSeqStartIdx = 0;
        splitInfo.curBStart = info.bStart;
        splitInfo.curSStart = info.sStart;
        splitInfo.isWrapped = false;
    }

    // process the actual workload to run on the current core
    sliceIterator.SetNeedDealTcSize(info.dealTcNum - splitInfo.preDealTcSize);
    StatisticInfo &statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    splitInfo.totalCompressedCnt = splitInfo.curCompressedCnt + statisticInfo.compressorScCnt;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcTilingStrategy(Vec1SplitInfo &splitInfo)
{
    // compute the split size in the headDim and Tc directions
    uint32_t maxDealColNum = BUFFER_SIZE_BYTE_32K / (constInfo_.cmpRatio * coff_ * sizeof(T));

    // tiling logic
    if (maxDealColNum < splitInfo.dBaseSize) {
        splitInfo.tcSplitSize = 1;
        splitInfo.dLoopCount = CeilDivT(splitInfo.dBaseSize, maxDealColNum);
        splitInfo.dSplitSize = splitInfo.dBaseSize / splitInfo.dLoopCount;
    } else {
        splitInfo.dSplitSize = splitInfo.dBaseSize;
        splitInfo.dLoopCount =
            splitInfo.dBaseSize / splitInfo.dSplitSize;  // this is usually equal to 1 here; keep the original logic
        splitInfo.tcSplitSize = maxDealColNum / splitInfo.dBaseSize;
    }
}

template <typename COMP>
__aicore__ inline Vec1SplitInfo CompressorBlockVectorPerf<COMP>::SplitCoreV1(const Vec1RunInfo &info)
{
    Vec1SplitInfo splitInfo;

    // 1. Compute the base grouping and slice size
    CalcGroupInfo(info, splitInfo);

    // 2. Compute task assignment based on the current BlockIdx (load balancing)
    CalcTaskDistribution(info, splitInfo);

    // 3. Refresh the iterator and get the start position state of the current core
    UpdateIteratorState(info, splitInfo);

    if (splitInfo.dealTcSize == 0) {
        return splitInfo;
    }

    // 4. Compute the concrete in-memory tiling logic
    CalcTilingStrategy(splitInfo);

    return splitInfo;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ComputeVec1(const Vec1RunInfo &info)
{
    if (info.dealTcNum == 0) {
        return;
    }
    if (info.resetResFlag) {
        compressedCnt_ = 0;
    }
    uint32_t preCompressedCnt = compressedCnt_;
    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    // compute the workload of the current VecCore
    if (splitInfo.dealTcSize == 0) {
        compressedCnt_ += info.dealScSize;
        return;
    }

    LoopInfo loopInfo;
    loopInfo.groupSize = splitInfo.vec1GroupSize;
    loopInfo.groupNum = splitInfo.vec1GroupNum;
    loopInfo.coreRowIdx = GetBlockIdx() / splitInfo.vec1GroupSize;
    loopInfo.coreColIdx = GetBlockIdx() % splitInfo.vec1GroupSize;
    loopInfo.isCoreRowLast = loopInfo.coreRowIdx == splitInfo.vec1GroupNum - 1;
    loopInfo.isCoreRowFirst = loopInfo.coreRowIdx == 0;

    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.SetRunStartBatch(info.bStart);
    // tiling loop
    uint64_t baseOffset = loopInfo.coreColIdx * splitInfo.dBaseSize;
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;

        CopyInApe(apeUb, dBaseOffset, splitInfo.dSplitSize);

        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        sliceIterator.SetWrapped(splitInfo.isWrapped);
        compressedCnt_ = preCompressedCnt + splitInfo.curCompressedCnt;
        for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize; tcIdx += splitInfo.tcSplitSize) {
            uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcSize - tcIdx);

            loopInfo.isCoreLoopFirst = tcIdx == 0;
            loopInfo.isCoreLoopLast = tcIdx + splitInfo.tcSplitSize >= splitInfo.dealTcSize;
            // process a single tile
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            sliceIterator.SetDealedTcCnt(0U);
            DealVec1BaseBlock(info, sliceIterator, loopInfo, dBaseOffset, splitInfo.dSplitSize, splitInfo.dBaseSize);
        }
    }
    compressedCnt_ = preCompressedCnt + info.dealScSize;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CommitC128State(const Vec1RunInfo &info)
{
    if constexpr (COMP::cacheMode != CACHE_MODE::EXPLICIT || COMP::coff != COFF::DISABLE) {
        return;
    }
    if (info.dealTcNum == 0) {
        return;
    }

    Vec1SplitInfo splitInfo = SplitCoreV1(info);
    if (splitInfo.dealTcSize == 0) {
        return;
    }

    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.SetRunStartBatch(info.bStart);
    uint64_t baseOffset = (GetBlockIdx() % splitInfo.vec1GroupSize) * splitInfo.dBaseSize;
    for (uint32_t dLoopIdx = 0; dLoopIdx < splitInfo.dLoopCount; dLoopIdx++) {
        uint64_t dBaseOffset = baseOffset + dLoopIdx * splitInfo.dSplitSize;
        CopyInApe(apeUb, dBaseOffset, splitInfo.dSplitSize);

        sliceIterator.Reset(splitInfo.curBStart, splitInfo.curSStart, splitInfo.dealSeqStartIdx, 0U);
        sliceIterator.SetWrapped(splitInfo.isWrapped);
        for (uint32_t tcIdx = 0; tcIdx < splitInfo.dealTcSize; tcIdx += splitInfo.tcSplitSize) {
            uint32_t actDealTcSize = min(splitInfo.tcSplitSize, splitInfo.dealTcSize - tcIdx);
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            sliceIterator.SetDealedTcCnt(0U);
            CommitC128StateBaseBlock(info, sliceIterator, dBaseOffset, splitInfo.dSplitSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ComputeVec2(const Compressor::Vec2RunInfo &info)
{
    SplitCoreV2(info);
    uint32_t vec2DealM = v2TcEndIdx - v2TcStartIdx;
    uint32_t loopCount = CeilDivT(vec2DealM, v2MBaseSize);
    for (uint32_t v2LoopIdx = 0, dealSize = v2MBaseSize; v2LoopIdx < loopCount; ++v2LoopIdx) {
        if (v2LoopIdx == loopCount - 1) {
            dealSize = vec2DealM - v2LoopIdx * v2MBaseSize;
        }
        DealVec2BaseBlock(info, v2TcStartIdx + v2LoopIdx * v2MBaseSize, dealSize);
    }
    v2TcStartIdx = 0;
    v2TcEndIdx = 0;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DealVec2BaseBlock(const Compressor::Vec2RunInfo &info,
                                                                          uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t computeSize = dealRowCount * constInfo_.headDim;
    int64_t inGmOffset = startRow * constInfo_.headDim;
    GlobalTensor<T> vec2InputGm = vec2InputGm_[info.v2DbIdx * constInfo_.dbSize];
    // CopyIn
    LocalTensor<T> vec1ResUb = inputQue1.AllocTensor<T>();
    DataCopy(vec1ResUb, vec2InputGm[inGmOffset], computeSize);
    inputQue1.EnQue(vec1ResUb);
    inputQue1.DeQue<T>();

    // RmsNorm
    LocalTensor<T> normResUb = tmpBuff1.Get<T>();
    LocalTensor<T> tempLocal = tmpBuff2.Get<T>();
    PipeBarrier<PIPE_V>();
    MultRowRmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, dealRowCount);
    inputQue1.FreeTensor(vec1ResUb);

    // rope: apply rope only to the latter RD; cast the first headDim -
    // ropeHeadDim elements of each row of normResUb to X_T, then combine with the rope result and store to outputUb
    LocalTensor<X_T> outputUb = outputQue1.AllocTensor<X_T>();
    PipeBarrier<PIPE_V>();
    CalRope(outputUb, normResUb, dealRowCount);
    PipeBarrier<PIPE_V>();
    // CopyOut
    outputQue1.EnQue(outputUb);
    outputQue1.DeQue<X_T>();
    CopyFinalResultOut(info, outputUb, startRow - v2TcStartIdx, dealRowCount);
    outputQue1.FreeTensor(outputUb);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::MultRowRmsNorm(const LocalTensor<T> &normResUb,
                                                                       const LocalTensor<T> &vec1ResUb,
                                                                       const LocalTensor<T> &normWeightUb,
                                                                       const LocalTensor<T> &tempLocal,
                                                                       uint32_t dealRowCount)
{
    RmsNormParam rmsNormParams;
    rmsNormParams.reciprocal = constInfo_.reciprocalD;
    rmsNormParams.epsilon = constInfo_.normEps;
    rmsNormParams.row = dealRowCount;
    rmsNormParams.col = constInfo_.headDim;
    RmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, rmsNormParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SingleCalRope(const LocalTensor<X_T> &outputUb,
                                                                      const LocalTensor<T> &normResUb, uint32_t rowCnt,
                                                                      uint32_t curDealScSize, uint32_t globalScStart)
{
    uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
    uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
    // sin and cos each take half; in practice each uses at most 16K, total 32K
    LocalTensor<T> cosUb = inputQue1.AllocTensor<T>();
    LocalTensor<T> sinUb = cosUb[BUFFER_SIZE_BYTE_16K / sizeof(T)];
    DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
    DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
    inputQue1.EnQue(sinUb);
    inputQue1.DeQue<T>();
    LocalTensor<T> tempLocal = tmpBuff2.Get<T>();
    PipeBarrier<PIPE_V>();
    RotaryPosEmb<COMP::rotaryMode>(normResUb[rowCnt * constInfo_.headDim], normResUb[rowCnt * constInfo_.headDim],
                                   cosUb, sinUb, tempLocal, gatherOffsetCastUb, curDealScSize, constInfo_.ropeHeadDim,
                                   constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
    inputQue1.FreeTensor(sinUb);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalRope(const LocalTensor<X_T> &outputUb,
                                                                const LocalTensor<T> &normResUb, uint32_t dealRowCount)
{
    uint32_t bStartIdx = OutputBStartIdx;
    uint32_t sStartIdx = OutputSStartIdx;
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, bStartIdx, sStartIdx, globalScStart);
    uint32_t totalSize = dealRowCount * constInfo_.headDim;
    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        // TH mode logic: execute the core computation at once, then idle-update the Index
        curDealScSize = dealRowCount;
        SingleCalRope(outputUb, normResUb, 0, curDealScSize, globalScStart);  // rowOffset passed as 0

        while (dealScSize > 0) {
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // BSH mode logic: execute the core computation in blocks (loop)
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;

        while (dealScSize > 0) {
            preOutputBStartIdx = bStartIdx;
            preOutputSStartIdx = sStartIdx;
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);

            if (curDealScSize > 0) {
                uint32_t rowCnt = dealRowCount - dealScSize - curDealScSize;
                SingleCalRope(outputUb, normResUb, rowCnt, curDealScSize, globalScStart);
            }
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, bStartIdx, sStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
    Cast(outputUb, normResUb, RoundMode::CAST_RINT, totalSize);
    PipeBarrier<PIPE_V>();
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SplitCoreV2(const Compressor::Vec2RunInfo &info)
{
    // after accumulating N base block data, do vec2, N=2; the passed RunInfo contains the bStart, bEnd, sStart, sEnd,
    // and dealTcCount of the data blocks processed by this group of cores; each group of cores splits the M direction
    // and distributes the data after C1/V1 to 8 * 2 vec cores for V2 computation each v2 computation redistributes the
    // data processed by each group of cores in the workspace to the vec cores of the current group based on the current
    // situation

    // Input: before syncAll, the start/end idx in batch and s directions and the actual data amount (m direction) of
    // the actual data blocks processed by each group of cube cores Output: the start/end positions in the m direction
    // of the data blocks processed by each vec core and the start position output to Gm
    uint32_t coreNum = constInfo_.usedCoreNum * 2;  // total number of cores, vec*2
    uint32_t currCoreIdx = GetBlockIdx();           // current vec core ID
    // 1. Compute the total number of vec2 base blocks
    uint32_t totalBaseNum = info.dealScSize;  // actual amount of data accumulated by the current group of cores

    uint32_t usedCoreNum = min(totalBaseNum, coreNum);
    // 2. Amount of data assigned to each vec core
    uint32_t avgBaseNum = CeilDivT(totalBaseNum, coreNum);
    if (currCoreIdx % coreNum >= usedCoreNum) {
        return;
    }
    // 3. Compute the start and end positions of each vec core
    uint32_t accumBaseNum = 0;  // number of base blocks accumulated so far
    uint32_t targetBaseNum =
        (currCoreIdx % coreNum + 1) * avgBaseNum;  // target number of base blocks the current vec core must reach
    uint32_t targetStartBaseNum =
        targetBaseNum -
        avgBaseNum;  // number of base blocks already assigned to cores before the current vec core when splitting
    bool setStart = false;
    for (uint32_t i = 0; i < totalBaseNum; ++i) {
        if (accumBaseNum >= totalBaseNum) {
            return;
        }
        accumBaseNum += 1;
        if (!setStart && (accumBaseNum > targetStartBaseNum)) {
            v2TcStartIdx = i;
            setStart = true;
        }
        if (setStart && (accumBaseNum >= targetBaseNum || i == (totalBaseNum - 1))) {
            // update the End core-split information of the current core
            v2TcEndIdx = i + 1;
            GetScIdxInfo(info.bStart, info.bCompressedId, info.dealScSize, v2TcStartIdx, v2TcEndIdx, OutputBStartIdx,
                         OutputSStartIdx, OutputSize);
            return;
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcGlobalScStart(uint32_t bStart, uint32_t scStart,
                                                                          uint32_t bEnd, uint32_t scEnd,
                                                                          uint64_t &globalScStart)
{
    for (uint32_t bIdx = bStart; bIdx < bEnd; ++bIdx) {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            globalScStart += GetScSize();
        } else {
            curActSeqLength_ = constInfo_.sSize;
            globalScStart += CeilDivT(curActSeqLength_, constInfo_.cmpRatio);
        }
    }
    globalScStart -= scStart;
    globalScStart += scEnd;
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart,
                                                                        uint32_t &dealScSize, uint32_t &curDealScSize)
{
    curActSeqLength_ = GetSeqLength(outputBStart);
    curStartPos_ = GetStartPos(outputBStart);
    uint32_t curBatchScSize =
        (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    uint32_t curBatchRemainScSize = curBatchScSize - outputSStart;
    curDealScSize = curBatchRemainScSize > dealScSize ? dealScSize : curBatchRemainScSize;
    dealScSize -= curDealScSize;
    outputSStart += curDealScSize;
    if (outputSStart == curBatchScSize) {
        outputBStart++;
        outputSStart = 0;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyFinalResultOut(const Compressor::Vec2RunInfo &info,
                                                                           const LocalTensor<X_T> &cmpKvOutUb,
                                                                           uint32_t startRow, uint32_t dealRowCount)
{
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, OutputBStartIdx, OutputSStartIdx, globalScStart);
    uint64_t outOffset = globalScStart * constInfo_.headDim;
    uint32_t copySize = dealRowCount * constInfo_.headDim;

    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        DataCopy(cmpKvOutGm_[outOffset], cmpKvOutUb, copySize);
        while (dealScSize > 0) {
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // handle BSH valid data being non-contiguous in memory (padding may exist)
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // compute the write-out index batch by batch
            preOutputBStartIdx = OutputBStartIdx;
            preOutputSStartIdx = OutputSStartIdx;
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
            DataCopy(cmpKvOutGm_[globalScStart * constInfo_.headDim], cmpKvOutUb[ubProcessedCount * constInfo_.headDim],
                     curDealScSize * constInfo_.headDim);
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, OutputBStartIdx, OutputSStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
}
}  // namespace Compressor
#endif  // COMPRESSOR_BLOCK_VECTOR_PREF_H
