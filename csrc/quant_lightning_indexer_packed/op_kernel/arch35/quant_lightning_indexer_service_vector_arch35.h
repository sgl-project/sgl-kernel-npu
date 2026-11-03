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
 * \file quant_lightning_indexer_service_vector_arch35.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../quant_lightning_indexer_common.h"
#include "vf/quant_lightning_indexer_vector1.h"
#include "vf/quant_lightning_indexer_topk.h"

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_6K = 6144;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;
    using QK_T = typename QLIT::qkType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo& info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo& info);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo& constInfo,
                                      const QLITilingData* __restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;
    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo& runInfo, LocalTensor<SCALE_T>& kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    struct QLICommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));
    resMm1UB_ = resMm1Buf_.Get<QK_T>();
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightFloatUB_ = weightFloatBuf_.Get<float>();
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>();

    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>();

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_,
                     (topkCountAlign256_ + 64) *
                         sizeof(uint32_t)); // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);

    // 刷-1
    Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo& constInfo,
                                                   const QLITilingData* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    if constexpr (std::is_same_v<SCORE_T, uint32_t>) {
        trunkLen_ = TRUNK_LEN_6K;
    } else {
        trunkLen_ = TRUNK_LEN_16K;
    }
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                                           GlobalTensor<SCALE_T> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo& runInfo, LocalTensor<SCALE_T>& kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<SCALE_T> v1PadParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams v1CopyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t v1StartBlockIdx = startS2 / kCacheBlockSize_;
        int32_t v1StartBlockOffset = startS2 % kCacheBlockSize_;
        int32_t v1BatchBlockOffset = batchId * maxBlockNumPerBatch_;
        v1CopyInParams.blockCount = 1;
        v1CopyInParams.srcStride = 0;
        v1CopyInParams.dstStride = 0;
        v1CopyInParams.rsv = 0;
        int32_t v1UbBaseOffset = 0;
        if (v1StartBlockOffset > 0) {
            int32_t v1FirstPartLen =
                kCacheBlockSize_ - v1StartBlockOffset > getLen ? getLen : kCacheBlockSize_ - v1StartBlockOffset;
            v1CopyInParams.blockLen = v1FirstPartLen * sizeof(SCALE_T);
            int32_t v1BlockId = blockTableGm.GetValue(v1BatchBlockOffset + v1StartBlockIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                 kScaleGm[v1BlockId * constInfo_.keyDequantScaleStride0 + v1StartBlockOffset],
                                 v1CopyInParams, v1PadParams);
            v1StartBlockIdx++;
            getLen = getLen - v1FirstPartLen;
            v1UbBaseOffset = v1FirstPartLen;
        }
        int32_t v1CopyLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        v1CopyInParams.blockLen = kCacheBlockSize_ * sizeof(SCALE_T);
        for (int32_t v1LoopIdx = 0; v1LoopIdx < v1CopyLoopNum; v1LoopIdx++) {
            if (v1LoopIdx == v1CopyLoopNum - 1) {
                v1CopyInParams.blockLen = (getLen - v1LoopIdx * kCacheBlockSize_) * sizeof(SCALE_T);
            }
            int32_t v1BlockId = blockTableGm.GetValue(v1BatchBlockOffset + v1StartBlockIdx + v1LoopIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(
                kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ + v1UbBaseOffset + v1LoopIdx * kCacheBlockSize_],
                kScaleGm[v1BlockId * constInfo_.keyDequantScaleStride0], v1CopyInParams, v1PadParams);
        }
    } else {
        v1CopyInParams.blockCount = 1;
        v1CopyInParams.blockLen = getLen * sizeof(SCALE_T);
        v1CopyInParams.srcStride = 0;
        v1CopyInParams.dstStride = 0;
        v1CopyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                             kScaleGm[runInfo.tensorKeyScaleOffset], v1CopyInParams, v1PadParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo& info)
{
    auto pingpong = (info.loop % 2);
    auto qliQScalePingpong = (info.qScaleLoop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            QLICommon::ConstInfo::CROSS_CV_EVENT + pingpong); // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
            QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong); // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }

    if (info.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qliQScalePingpong);
        // weightsGm --> weightUB_
        int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
        DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockCount = curAivS1ProcNum;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
        wDataCopyExtParams.srcStride = 0;
        wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_[qliQScalePingpong * (UB_BANK_STRIDE / sizeof(W_T))], weightsGm[weightGmOffset],
                    wDataCopyExtParams, padWeightsParams);

        // qScaleGm  -->  qScaleUB_
        DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        DataCopyPad(qScaleUB_[qliQScalePingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))], qScaleGm[weightGmOffset],
                    qDataCopyExtParams, padQScaleParams);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qliQScalePingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qliQScalePingpong);
    }

    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_) ?
                              info.validS2Len - info.s2Idx * s2BaseSize_ :
                              16 * s2BaseSize_;
        // kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    // CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(
        QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2); // V核等C核计算完mm1，mm1Res已搬运到UB
    LocalTensor<SCORE_T> outBase;
    if (std::is_same_v<SCORE_T, uint16_t>) {
        outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    } else {
        outBase = vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_];
    }
    auto weightBase = weightUB_[qliQScalePingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[qliQScalePingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[qliQScalePingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLstride,
                                        (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)), weightBase,
                                        UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase, kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T), gSize_, curAivS1ProcNum);
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qliQScalePingpong);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    int64_t vec1OutGmOffset =
        blockId_ % 2 == 0 ?
            curS2Idx :
            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride =
        (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT +
                                                                   pingpong); // V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo& info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t qliCurS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t qliCurS2Idx = info.s2Idx * s2BaseSize_;
    int64_t qliCurS1ProcNum = qliCurS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t qliCurAivS1Idx = qliCurS1Idx + (blockId_ % 2) * CeilDiv(qliCurS1ProcNum, 2);
    int64_t qliCurAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(qliCurS1ProcNum, 2) : qliCurS1ProcNum / 2;

    AscendC::DataCopyExtParams qliCopyInParams;
    qliCopyInParams.blockCount = 1;
    qliCopyInParams.srcStride = 0;
    qliCopyInParams.dstStride = 0;
    qliCopyInParams.rsv = 0;

    AscendC::DataCopyParams qliCopyOutParams;
    qliCopyOutParams.blockCount = 1;
    qliCopyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    qliCopyOutParams.srcStride = 0;
    qliCopyOutParams.dstStride = 0;

    int32_t qliCuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        qliCuRealAcSeq = info.actS2SizeOrig - info.actS1Size + qliCurAivS1Idx + 1;
    }

    int32_t qliValidS2Len = qliCuRealAcSeq;
    for (uint32_t i = 0; i < qliCurAivS1ProcNum; i++) {
        uint32_t qliRowIdx = blockId_ % 2 * CeilDiv(qliCurS1ProcNum, 2) + i;
        uint32_t qliVecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        SCORE_T qliZero = 0;
        int32_t qliNeg = -1;
        if (constInfo_.attenMaskFlag) {
            qliValidS2Len = (int32_t)i + qliCuRealAcSeq;
        }
        if (qliValidS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), qliNeg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (qliCurS1Idx + qliRowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), qliCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> qliPadParams{true, 0, 0, 0};
        if (qliValidS2Len >= topkCount_) {
            uint32_t qliS2LoopNum = (qliValidS2Len + trunkLen_ - 1) / trunkLen_;
            if (qliS2LoopNum == 1) {
                uint32_t qliValidS2LenAlign = QLICommon::Align(qliValidS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[qliValidS2Len / 256 * 256], qliZero,
                          qliValidS2LenAlign - qliValidS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                qliCopyInParams.blockLen = qliValidS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> qliPadParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[qliVecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    qliCopyInParams, qliPadParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, qliValidS2LenAlign, 0, 1);
            } else {
                for (uint32_t qliLoopIdx = 0; qliLoopIdx < qliS2LoopNum; qliLoopIdx++) {
                    if (qliLoopIdx == 0) {
                        qliCopyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_,
                                             scoreGm[qliVecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                     (uint64_t)s2BaseSize_)],
                                             qliCopyInParams, qliPadParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, qliLoopIdx, qliS2LoopNum);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t qliValidTrunkLen =
                        (qliLoopIdx * trunkLen_ + trunkLen_) > qliValidS2Len ? qliValidS2Len % trunkLen_ : trunkLen_;
                    uint32_t qliOffset =
                        qliVecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                        qliLoopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t qliMask[1];
                        qliMask[0] = ~0;
                        qliMask[0] = qliMask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], qliZero, qliMask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], qliZero,
                                  topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    qliCopyInParams.blockLen = qliValidTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + qliValidTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + qliValidTrunkLen / 256 * 256], qliZero,
                                  QLICommon::Align(qliValidTrunkLen, (uint32_t)256) - qliValidTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[qliOffset], qliCopyInParams,
                                         qliPadParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            QLICommon::Align(topkCountAlign256_ + qliValidTrunkLen, (uint32_t)256), qliLoopIdx,
                            qliS2LoopNum);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)qliZero, qliValidS2Len);
        }

        if (qliValidS2Len < topkCount_) {
            uint64_t qliMask[1];
            qliMask[0] = ~0;
            qliMask[0] = qliMask[0] << (qliValidS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[qliValidS2Len / 8 * 8], qliNeg, qliMask, 1, 1, 0);
        }

        if (qliValidS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[qliValidS2Len / 8 * 8 + 64], qliNeg,
                      topkCount_ - (qliValidS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (qliCurS1Idx + qliRowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), qliCopyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}
} // namespace QLIKernel
#endif
