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
 * \file moe_distribute_combine_a5.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A5_CCU_H
#define MOE_DISTRIBUTE_COMBINE_A5_CCU_H

#include "lib/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_v2_base.h"

namespace MoeDistributeCombineA5CCUImpl {
constexpr uint8_t BUFFER_NUM = 2;  // 多buf
constexpr uint32_t UB_ALIGN = 32;  // UB按32字节对齐
constexpr uint32_t COUNT_OFFSET = 512;

constexpr uint32_t ALIGN_DOWN_TO_32_MASK = 31;
constexpr uint32_t NEED_THIRTY_FIRST = 31;
constexpr uint32_t RIGHT_SHIFT_BIT_FIVE = 5;
constexpr uint64_t HALF_DATA_DIV = 2;
constexpr uint32_t DUAL_DATA = 2;
constexpr uint16_t BLOCK_COUNT = 2;

template <typename T>
__aicore__ inline T Ceil32(T x)
{
    return (x + NEED_THIRTY_FIRST) >> RIGHT_SHIFT_BIT_FIVE;
}

template <typename T>
__aicore__ inline T Align32(T x)
{
    return (x + ALIGN_DOWN_TO_32_MASK) & (~ALIGN_DOWN_TO_32_MASK);
}

using namespace AscendC;
using namespace MoeDistributeV2Base;
#define TemplateMoeDistributeCombineA5TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMoeDistributeCombineA5TypeFunc ExpandXType, ExpandIdxType

template <TemplateMoeDistributeCombineA5TypeClass>
class MoeDistributeCombineA5
{
public:
    __aicore__ inline MoeDistributeCombineA5(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR tpSendCount, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR sharedExpertX,
                                GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    // 按照rank分核逻辑，得到每个核的起始和数量
    __aicore__ inline void SplitRank(uint32_t &start, uint32_t &count);
    // 准备发送数据的初始化
    __aicore__ inline void PrepareInit();
    // 数据复制与填充
    __aicore__ inline void CopyAndPadData(uint32_t &startRank, uint32_t &rankNum, uint32_t &eachSize,
                                          LocalTensor<uint64_t> &sizeLT, LocalTensor<uint64_t> &sendOffsetLT);
    // 共享专家卡和MOE专家卡通信准备
    __aicore__ inline void HandleSharedAndMoeRank(uint32_t &startRank, LocalTensor<uint64_t> &sizeLT, uint32_t &eachCnt,
                                                  uint32_t &rankNum, LocalTensor<uint32_t> &offsetLT,
                                                  LocalTensor<uint64_t> &sendOffsetLT);
    __aicore__ inline void SharedAndMoePrepare();
    // 准备发送数据
    __aicore__ inline void PrepareSendData();
    // 通信
    __aicore__ inline void Communication();
    // 计算的初始化
    __aicore__ inline void CalculateInit();
    // 计算并输出
    __aicore__ inline void HandleCalculateToken(uint32_t &beginIndex, uint32_t &endIndex, uint32_t &processLen,
                                                uint32_t &tokenOffset);
    __aicore__ inline void Calculate();

    __aicore__ inline void HandleSharedExpertX(uint32_t &tokenIndex, uint32_t &tokenOffset, uint32_t &processLen);

    __aicore__ inline void CalculateRecvBufAndRecvOffset(const MoeDistributeCombineV2TilingData *tilingData);

    __aicore__ inline void TokenMaskCalCnt();

    __aicore__ inline void CalculateMoeResult(uint32_t &tokenIndex, uint32_t &tokenOffset, uint32_t &processLen);
    TPipe *pipe_;

    GlobalTensor<ExpandXType> expandXGT_;
    GlobalTensor<int32_t> expertIdsGT_;
    GlobalTensor<int32_t> expandIdxGT_;
    GlobalTensor<float> expandScalesGT_;
    GlobalTensor<ExpandIdxType> epSendCountGT_;
    GlobalTensor<ExpandXType> expandOutGT_;
    GlobalTensor<bool> xActiveMaskGT_;

    LocalTensor<ExpandIdxType> inputCountLT_;
    LocalTensor<int32_t> expertIdsLT_;
    LocalTensor<int32_t> expandIdxLT_;
    LocalTensor<float> expandScalesLT_;
    LocalTensor<float> castLT_;
    LocalTensor<float> mulLT_;
    LocalTensor<float> sumLT_;

    uint32_t axisBS_;
    uint32_t axisH_;
    uint32_t axisK_;
    uint32_t aivNum_;
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t aivId_;
    uint32_t sharedExpertRankNum_;
    uint32_t sharedExpertNum_;
    uint32_t rankNumPerSharedExpert_;
    uint32_t moeExpertNum_;
    uint32_t moeExpertRankNum_;
    uint32_t localExpertNum_;
    uint32_t hasSharedExpertX_;
    uint64_t activeMaskBsCnt_;
    uint32_t axisMaxBS_;
    uint32_t axisBsAlignSize_;

    uint32_t bskNum_;
    uint32_t perTokenSize_;
    uint32_t perRankDataSize_;
    bool isShareExpertRank_;
    bool isInputTokenMaskFlag_ = false;

    TQue<QuePosition::VECIN, BUFFER_NUM> moeQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sharedQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sharedExpertXQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> tokenQue_;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    AscendC::HcclHandle hcclHandleId_;

    GM_ADDR sendBufGM_;
    GM_ADDR sendSizeGM_;
    GM_ADDR sendOffsetGM_;
    GM_ADDR recvBufGM_;
    GM_ADDR sharedExpertXGM_;
    uint64_t recvOffset_;

    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> xActMaskSumTBuf_;
};

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGT_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::CalculateRecvBufAndRecvOffset(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    GlobalTensor<int32_t> statusGT;
    __gm__ HcclCombineOpParam *context = (__gm__ HcclCombineOpParam *)(GetHcclContext<0>());
    // 结构体切换后，V1版本初始化方法不可用（已废弃），改用V2版本
    hccl_.InitV2((GM_ADDR)context, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineV2TilingData, mc2CcTiling1));
    GM_ADDR cclBuf = (GM_ADDR)context->windowsOut[0];
    uint64_t statusSize = aivNum_ * COUNT_OFFSET;
    statusGT.SetGlobalBuffer((__gm__ int32_t *)(cclBuf + aivId_ * COUNT_OFFSET));
    if (statusGT(0) == 0) {
        recvBufGM_ = cclBuf + statusSize;
        recvOffset_ = statusSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 1;
    } else {
        uint64_t halfDataSize = (context->winSize - statusSize) / HALF_DATA_DIV;
        recvBufGM_ = cclBuf + statusSize + halfDataSize;
        recvOffset_ = statusSize + halfDataSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 0;
    }

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGT);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeCombineV2TilingData *tilingData)
{
    pipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    axisMaxBS_ = tilingData->moeDistributeCombineV2Info.globalBs / epWorldSize_;
    localExpertNum_ = moeExpertNum_ / moeExpertRankNum_;
    isShareExpertRank_ = epRankId_ < sharedExpertRankNum_;
    bskNum_ = axisMaxBS_ * axisK_;
    perTokenSize_ = axisH_ * sizeof(ExpandXType);
    perRankDataSize_ = perTokenSize_ * axisMaxBS_ * localExpertNum_;

    expandXGT_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGT_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandIdxGT_.SetGlobalBuffer((__gm__ int32_t *)expandIdx);
    epSendCountGT_.SetGlobalBuffer((__gm__ ExpandIdxType *)epSendCount);
    expandScalesGT_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGT_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    xActiveMaskGT_.SetGlobalBuffer((__gm__ bool *)xActiveMask);

    sharedExpertXGM_ = sharedExpertX;
    sendBufGM_ = workspaceGM;
    sendSizeGM_ = sendBufGM_ + epWorldSize_ * perRankDataSize_;
    sendOffsetGM_ = sendSizeGM_ + epWorldSize_ * sizeof(uint64_t) * DUAL_DATA;

    CalculateRecvBufAndRecvOffset(tilingData);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::CopyAndPadData(
    uint32_t &startRank, uint32_t &rankNum, uint32_t &eachSize, LocalTensor<uint64_t> &sizeLT,
    LocalTensor<uint64_t> &sendOffsetLT)
{
    GlobalTensor<uint64_t> sendGT;
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendSizeGM_ + startRank * sizeof(uint64_t)));
    DataCopyExtParams params = {BLOCK_COUNT, static_cast<uint32_t>(rankNum * sizeof(uint64_t)),
                                static_cast<uint32_t>(Ceil32(eachSize) - Ceil32(rankNum * sizeof(uint64_t))),
                                static_cast<uint32_t>((epWorldSize_ - rankNum) * sizeof(uint64_t)), 0};
    DataCopyPad(sendGT, sizeLT, params);

    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendOffsetGM_ + startRank * sizeof(uint64_t)));
    DataCopyPad(sendGT, sendOffsetLT, params);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::SplitRank(uint32_t &start,
                                                                                                 uint32_t &count)
{
    count = epWorldSize_ / aivNum_;
    uint32_t remainCnt = epWorldSize_ % aivNum_;
    start = count * aivId_;

    if (aivId_ < remainCnt) {
        ++count;
        start += aivId_;
    } else {
        start += remainCnt;
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::PrepareInit()
{
    pipe_->InitBuffer(tokenQue_, BUFFER_NUM, perTokenSize_);
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, localExpertNum_ * epWorldSize_ * sizeof(ExpandIdxType));
    inputCountLT_ = tmpBuf.Get<ExpandIdxType>();
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(localExpertNum_ * epWorldSize_ * sizeof(ExpandIdxType)), 0U,
                                 0U};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyPad(inputCountLT_, epSendCountGT_, copyParams, padParams);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::HandleSharedAndMoeRank(
    uint32_t &startRank, LocalTensor<uint64_t> &sizeLT, uint32_t &eachCnt, uint32_t &rankNum,
    LocalTensor<uint32_t> &offsetLT, LocalTensor<uint64_t> &sendOffsetLT)
{
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(perTokenSize_), 0U, 0U};
    DataCopyPadParams padParams = {false, 0, 0, 0};
    GlobalTensor<ExpandXType> tokenGT;
    uint32_t inPreExpertCount = 0;
    uint32_t outPreExpertCount = 0;
    uint32_t localExpertNum = isShareExpertRank_ ? 1 : localExpertNum_;
    for (uint32_t k = 0; k < localExpertNum; ++k) {
        uint32_t expertOffset = k * epWorldSize_;
        uint32_t inPreCount = startRank > 0 ? inputCountLT_(expertOffset + startRank - 1) : inPreExpertCount;
        for (uint32_t i = 0; i < rankNum; ++i) {
            uint32_t curRankId = startRank + i;
            uint32_t curSumNum = inputCountLT_(expertOffset + curRankId);
            uint32_t curTokenNum = curSumNum - inPreCount;
            uint32_t tokenBeginIdx = inPreCount * axisH_;

            GM_ADDR dstGM = sendBufGM_ + curRankId * perRankDataSize_ + offsetLT(i) * perTokenSize_;
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            for (uint32_t j = 0; j < curTokenNum; ++j) {
                auto t = tokenQue_.AllocTensor<ExpandXType>();
                DataCopyPad(t, expandXGT_[tokenBeginIdx], copyParams, padParams);
                tokenQue_.EnQue(t);
                t = tokenQue_.DeQue<ExpandXType>();
                tokenGT.SetGlobalBuffer((__gm__ ExpandXType *)dstGM);
                DataCopyPad(tokenGT, t, copyParams);
                tokenQue_.FreeTensor<ExpandXType>(t);
                tokenBeginIdx += axisH_;
                dstGM += perTokenSize_;
            }

            offsetLT(i) += curTokenNum;

            inPreCount = curSumNum;
            uint64_t countByte = curTokenNum * perTokenSize_;
            uint64_t halfSize = countByte / HALF_DATA_DIV;
            sizeLT(i) += halfSize;
            sizeLT(i + eachCnt) += countByte - halfSize;

            sendOffsetLT(i) = curRankId * perRankDataSize_;
            sendOffsetLT(i + eachCnt) = curRankId * perRankDataSize_ + sizeLT(i);
        }

        inPreExpertCount = inputCountLT_((k + 1) * epWorldSize_ - 1);
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::SharedAndMoePrepare()
{
    uint32_t startRank;
    uint32_t rankNum;
    SplitRank(startRank, rankNum);
    if (rankNum == 0) {
        return;
    }
    uint32_t endRank = startRank + rankNum;
    uint32_t eachSize = Align32(rankNum * sizeof(uint64_t));
    uint32_t eachCnt = eachSize / sizeof(uint64_t);
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, eachSize * DUAL_DATA + rankNum * sizeof(uint32_t));
    LocalTensor<uint64_t> sizeLT = tmpBuf.GetWithOffset<uint64_t>(eachCnt * DUAL_DATA, 0);
    LocalTensor<uint32_t> offsetLT = tmpBuf.GetWithOffset<uint32_t>(rankNum, eachSize * DUAL_DATA);
    pipe_->InitBuffer(tmpBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sendOffsetLT = tmpBuf.Get<uint64_t>();
    uint32_t rankCntSize = Align32(localExpertNum_ * sizeof(uint32_t));
    uint32_t rankCnt = rankCntSize / sizeof(uint32_t);
    LocalTensor<int32_t> sizeI32LT = sizeLT.ReinterpretCast<int32_t>();
    Duplicate(sizeI32LT, 0, DUAL_DATA * eachCnt * (sizeof(uint64_t) / sizeof(int32_t)) + rankNum);

    PipeBarrier<PIPE_ALL>();

    HandleSharedAndMoeRank(startRank, sizeLT, eachCnt, rankNum, offsetLT, sendOffsetLT);

    SyncFunc<AscendC::HardEvent::S_MTE3>();

    CopyAndPadData(startRank, rankNum, eachSize, sizeLT, sendOffsetLT);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::PrepareSendData()
{
    PrepareInit();
    SharedAndMoePrepare();
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::CalculateInit()
{
    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(int32_t));
    expertIdsLT_ = tmpBuf.Get<int32_t>();
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(ExpandIdxType));
    expandIdxLT_ = tmpBuf.Get<int32_t>();
    pipe_->InitBuffer(tmpBuf, bskNum_ * sizeof(float));
    expandScalesLT_ = tmpBuf.Get<float>();

    DataCopyExtParams bskParams = {1U, static_cast<uint32_t>(bskNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    DataCopyPad(expandIdxLT_, expandIdxGT_, bskParams, copyPadParams);
    DataCopyPad(expertIdsLT_, expertIdsGT_, bskParams, copyPadParams);
    DataCopyPad(expandScalesLT_, expandScalesGT_, bskParams, copyPadFloatParams);
    pipe_->InitBuffer(moeQue_, BUFFER_NUM, perTokenSize_);
    pipe_->InitBuffer(outQue_, BUFFER_NUM, perTokenSize_);
    if (sharedExpertRankNum_ > 0) {
        pipe_->InitBuffer(sharedQue_, BUFFER_NUM, perTokenSize_);
    }

    if (hasSharedExpertX_ > 0) {
        pipe_->InitBuffer(sharedExpertXQue_, BUFFER_NUM, perTokenSize_);
    }

    uint32_t tokenF32Size = axisH_ * sizeof(float);
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    castLT_ = tmpBuf.Get<float>();
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    mulLT_ = tmpBuf.Get<float>();
    pipe_->InitBuffer(tmpBuf, tokenF32Size);
    sumLT_ = tmpBuf.Get<float>();

    activeMaskBsCnt_ = axisBS_;
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        pipe_->InitBuffer(xActMaskTBuf_, bskNum_ * sizeof(int32_t));
        pipe_->InitBuffer(xActMaskCastTBuf_, bskNum_ * sizeof(int32_t));
        pipe_->InitBuffer(xActMaskSumTBuf_, bskNum_ * sizeof(int32_t));
        TokenMaskCalCnt();  // 计算一维mask
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::Communication()
{
    SyncAll<true>();
    if (aivId_ == 0) {
        uint64_t localDataSize = 0;
        int32_t idx = 0;
        int32_t count = 0;
        if (!isShareExpertRank_) {
            for (int i = 0; i < localExpertNum_; ++i) {
                idx = i * epWorldSize_ + epRankId_;
                count = idx > 0 ? (inputCountLT_(idx) - inputCountLT_(idx - 1)) : inputCountLT_(idx);
                localDataSize += count * perTokenSize_;
            }
        } else {
            idx = epRankId_;
            count = idx > 0 ? (inputCountLT_(idx) - inputCountLT_(idx - 1)) : inputCountLT_(idx);
            localDataSize += count * perTokenSize_;
        }

        hcclHandleId_ = hccl_.AlltoAllvWrite<true>(sendBufGM_, sendOffsetGM_, sendSizeGM_, recvOffset_, localDataSize);
    }

    CalculateInit();  // 在这里可以被通信开销掩盖

    if (aivId_ == 0) {
        hccl_.Wait(hcclHandleId_);
    }
    SyncAll<true>();
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::HandleSharedExpertX(
    uint32_t &tokenIndex, uint32_t &tokenOffset, uint32_t &processLen)
{
    DataCopyParams copyInParams = {1U, static_cast<uint16_t>(processLen * sizeof(ExpandXType)), 0U, 0U};
    DataCopyPadParams padInParams = {false, 0, 0, 0};
    GlobalTensor<ExpandXType> sharedExpertXGT;
    if (hasSharedExpertX_ > 0) {
        auto rowTmpLocal = sharedExpertXQue_.AllocTensor<ExpandXType>();
        GM_ADDR shareExpertXAddr = sharedExpertXGM_ + tokenIndex * perTokenSize_ + tokenOffset * sizeof(ExpandXType);
        sharedExpertXGT.SetGlobalBuffer((__gm__ ExpandXType *)(shareExpertXAddr));
        DataCopyPad(rowTmpLocal, sharedExpertXGT, copyInParams, padInParams);
        sharedExpertXQue_.EnQue(rowTmpLocal);
        rowTmpLocal = sharedExpertXQue_.DeQue<ExpandXType>();
        Cast(castLT_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
        AscendC::Add(sumLT_, sumLT_, castLT_, processLen);
        sharedExpertXQue_.FreeTensor<ExpandXType>(rowTmpLocal);
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::CalculateMoeResult(
    uint32_t &tokenIndex, uint32_t &tokenOffset, uint32_t &processLen)
{
    GlobalTensor<ExpandXType> rowTmpGT;
    DataCopyParams copyInParams = {1U, static_cast<uint16_t>(processLen * sizeof(ExpandXType)), 0U, 0U};
    DataCopyPadParams padInParams = {false, 0, 0, 0};
    uint32_t index = tokenIndex * axisK_;
    for (uint32_t i = 0; i < axisK_; i++) {
        int32_t moeExpert = expertIdsLT_(index);
        int32_t expertRank = moeExpert / localExpertNum_;
        int32_t dataSrcRank = expertRank + sharedExpertRankNum_;
        int32_t inPreCount = expandIdxLT_(index);
        GM_ADDR src = recvBufGM_ + dataSrcRank * perRankDataSize_ + inPreCount * perTokenSize_ +
                      tokenOffset * sizeof(ExpandXType);
        rowTmpGT.SetGlobalBuffer((__gm__ ExpandXType *)src);
        float scaleVal = expandScalesLT_(index);
        auto t0 = moeQue_.AllocTensor<ExpandXType>();
        DataCopyPad(t0, rowTmpGT, copyInParams, padInParams);
        moeQue_.EnQue(t0);

        t0 = moeQue_.DeQue<ExpandXType>();
        Cast(castLT_, t0, AscendC::RoundMode::CAST_NONE, processLen);
        AscendC::Muls(mulLT_, castLT_, scaleVal, processLen);
        AscendC::Add(sumLT_, sumLT_, mulLT_, processLen);
        moeQue_.FreeTensor<ExpandXType>(t0);
        ++index;
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::HandleCalculateToken(
    uint32_t &beginIndex, uint32_t &endIndex, uint32_t &processLen, uint32_t &tokenOffset)
{
    GM_ADDR sharedBase = nullptr;
    GlobalTensor<ExpandXType> rowTmpGT;
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(processLen * sizeof(ExpandXType)), 0, 0, 0};
    DataCopyParams copyInParams = {1U, static_cast<uint16_t>(processLen * sizeof(ExpandXType)), 0U, 0U};
    DataCopyPadParams padInParams = {false, 0, 0, 0};
    for (uint32_t tokenIndex = beginIndex; tokenIndex < endIndex; tokenIndex++) {
        Duplicate(sumLT_, (float)0, axisH_);
        CalculateMoeResult(tokenIndex, tokenOffset, processLen);
        for (uint32_t sharedExpertRankId = 0; sharedExpertRankId < sharedExpertNum_; sharedExpertRankId++) {
            int32_t moeOnShareRank = epRankId_ % sharedExpertNum_;
            sharedBase = isShareExpertRank_ ? ((GM_ADDR)expandXGT_.GetPhyAddr())
                                            : (recvBufGM_ + moeOnShareRank * perRankDataSize_);
            GM_ADDR shareAddr = sharedBase + tokenIndex * perTokenSize_ + tokenOffset * sizeof(ExpandXType) +
                                (sharedExpertRankId * rankNumPerSharedExpert_ * perRankDataSize_);
            rowTmpGT.SetGlobalBuffer((__gm__ ExpandXType *)(shareAddr));
            auto t0 = sharedQue_.AllocTensor<ExpandXType>();
            DataCopyPad(t0, rowTmpGT, copyInParams, padInParams);
            sharedQue_.EnQue(t0);
            t0 = sharedQue_.DeQue<ExpandXType>();
            Cast(castLT_, t0, AscendC::RoundMode::CAST_NONE, processLen);
            AscendC::Add(sumLT_, sumLT_, castLT_, processLen);
            sharedQue_.FreeTensor(t0);
        }
        HandleSharedExpertX(tokenIndex, tokenOffset, processLen);
        auto outLT = outQue_.AllocTensor<ExpandXType>();
        Cast(outLT, sumLT_, AscendC::RoundMode::CAST_RINT, processLen);
        outQue_.EnQue(outLT);
        outLT = outQue_.DeQue<ExpandXType>();
        DataCopyPad(expandOutGT_[tokenIndex * axisH_ + tokenOffset], outLT, copyOutParams);
        outQue_.FreeTensor(outLT);
    }
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::Calculate()
{
    if (activeMaskBsCnt_ == 0U) {
        return;
    }
    uint32_t beginIndex;
    uint32_t endIndex;
    uint32_t processLen;
    uint32_t tokenOffset;
    if (activeMaskBsCnt_ < aivNum_) {
        uint32_t aivNumPerToken = aivNum_ / activeMaskBsCnt_;
        if (aivId_ >= (activeMaskBsCnt_ * aivNumPerToken)) {
            return;
        }
        uint32_t tokenIndex = aivId_ / aivNumPerToken;
        uint32_t blockCnt = UB_ALIGN / sizeof(ExpandXType);
        processLen = (axisH_ / aivNumPerToken / blockCnt) * blockCnt;

        tokenOffset = processLen * (aivId_ % aivNumPerToken);
        if ((aivId_ % aivNumPerToken) == (aivNumPerToken - 1)) {
            processLen = axisH_ - ((aivNumPerToken - 1) * processLen);
        }
        beginIndex = tokenIndex;
        endIndex = beginIndex + 1;
    } else {
        uint32_t tokenPerAivNum = activeMaskBsCnt_ / aivNum_;
        uint32_t remainToken = activeMaskBsCnt_ % aivNum_;
        beginIndex = tokenPerAivNum * aivId_;
        if (aivId_ < remainToken) {
            tokenPerAivNum++;
            beginIndex = tokenPerAivNum * aivId_;
        } else {
            beginIndex += remainToken;
        }
        endIndex = beginIndex + tokenPerAivNum;
        processLen = axisH_;
        tokenOffset = 0;
    }
    if (processLen == 0) {
        return;
    }

    HandleCalculateToken(beginIndex, endIndex, processLen, tokenOffset);
}

template <TemplateMoeDistributeCombineA5TypeClass>
__aicore__ inline void MoeDistributeCombineA5<TemplateMoeDistributeCombineA5TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        PrepareSendData();
        Communication();
        Calculate();
        if (aivId_ == 0) {
            hccl_.Finalize();
        }
    }
}

}  // namespace MoeDistributeCombineA5CCUImpl
#endif  // MOE_DISTRIBUTE_COMBINE_IMPL_H
