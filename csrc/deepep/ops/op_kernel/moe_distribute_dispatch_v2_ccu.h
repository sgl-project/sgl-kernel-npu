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
 * \file moe_distribute_dispatch_v2_ccu.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A5_CCU_H
#define MOE_DISTRIBUTE_DISPATCH_A5_CCU_H

#include "lib/hccl/hccl.h"
#include "common.h"
#include "adv_api/reduce/sum.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_v2_base.h"
#include "quantize_functions.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchA5CCUImpl {
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint8_t QUANT_PADDING_VALUE = 0;
constexpr uint64_t HALF_DATA_SIZE_DIV = 2;
constexpr uint32_t DUAL_DATA = 2;
constexpr uint16_t BLOCK_COUNT = 2;
constexpr uint32_t COUNT_OFFSET = 512;
constexpr uint32_t STATUS_OFFSET = 512;
constexpr uint32_t VEC_LEN = 256;
constexpr uint32_t UB_ALIGN = 32;  // UB 按照 32 字节对齐
constexpr uint32_t ONE_REPEAT_SORT_NUM = 32;
constexpr float MAX_FP32 = 2147483647.0f;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;

constexpr uint32_t UNQUANT = 0;
constexpr uint32_t STATIC_QUANT = 1;
constexpr uint32_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr uint32_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr uint32_t MX_QUANT = 4;

#define TemplateMoeDistributeDispatchA5TypeClass \
    typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateMoeDistributeDispatchA5TypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
using namespace MoeDistributeV2Base;
template <TemplateMoeDistributeDispatchA5TypeClass>
class MoeDistributeDispatchA5
{
public:
    __aicore__ inline MoeDistributeDispatchA5(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR xActiveMask,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalAttrs(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void InitBuf();
    __aicore__ inline void InitCommAndStatus(GM_ADDR workspaceGM, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void CalcTokenActiveMask();
    __aicore__ inline uint32_t CalcToSharedRankId(uint32_t sendCnt);
    __aicore__ inline void TokenGatherProcess(int32_t sharedExpertRankNum, int32_t localExpertNum);

    __aicore__ inline void TokenScatter();
    __aicore__ inline void Communication();
    __aicore__ inline void TokenGather();

    __aicore__ inline void QuantStatic(LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal,
                                       int32_t expertIndex);
    __aicore__ inline void QuantDynamicPerToken(LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal,
                                                int32_t expertIndex);

    __aicore__ inline void QuantProcess(LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal,
                                        int32_t expertIndex);

    __aicore__ inline void CopyScalesToOut(uint32_t currentTokenIndex, LocalTensor<ExpandXOutType> &quantTok,
                                           int32_t expertIndex);

    __aicore__ inline void ShareScatter();
    __aicore__ inline void UpdateShareRankOutput();
    __aicore__ inline void ShareScatterCopyTokens();
    __aicore__ inline void SetShareExpertSendSize();
    __aicore__ inline void MoeScatterCopyTokens();
    __aicore__ inline void MoeScatter();
    __aicore__ inline void UpdateMoeRankOutput(int32_t expertIndex, uint32_t startRank, uint32_t rankPerAiv,
                                               uint32_t expertOffset, uint32_t expertTokenNum);
    __aicore__ inline void SaveExpandIdx();
    __aicore__ inline void TokenGatherInit(uint32_t localExpertNum);
    __aicore__ inline void MoeGatherPrepare(uint32_t localExpertNum, uint32_t startRankIdx);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void QuantInit();
    __aicore__ inline void ProcessToken(GlobalTensor<ExpandXOutType> &outTokenGT, uint32_t tokenIndex,
                                        DataCopyParams &tokenInParams, DataCopyPadParams &padParams,
                                        DataCopyParams &tokenOutParams, DataCopyParams &scaleInParams,
                                        uint32_t expertIndex);

    TPipe *pipe_;
    uint32_t epRankId_;
    uint32_t aivId_;
    uint32_t axisBS_;
    uint32_t axisH_;
    uint32_t axisK_;
    uint32_t epWorldSize_;
    uint32_t moeExpertNum_;
    uint32_t sharedExpertRankNum_;
    uint32_t moeExpertRankNum_;
    uint32_t localExpertNum_;
    uint32_t aivNum_;
    uint32_t bskNum_;
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_;

    uint32_t perTokenInSize_;
    uint32_t perTokenOutSize_;
    uint32_t perTokenMergeSize_;
    uint32_t perTokenCommSize_;
    uint32_t perRankDataSize_;
    uint32_t expertTokenNumsType_;
    uint32_t scaleOutBytes_;
    uint32_t shareRankRcvTokenCnt_;

    bool isShareExpertRank_;

    GlobalTensor<XType> xGT_;
    GlobalTensor<int32_t> expertIdsGT_;
    GlobalTensor<int32_t> expandIdxGT_;
    GlobalTensor<ExpandXOutType> expandXOutGT_;
    GlobalTensor<uint8_t> dynamicScaleGT_;
    GlobalTensor<int64_t> expertTokenNumsGT_;
    GlobalTensor<int32_t> sendCountsGT_;
    GlobalTensor<float> scalesGT_;
    GlobalTensor<bool> xActiveMaskGT_;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    AscendC::HcclHandle hcclHandleId_;
    AscendC::HcclDataType dataType_;

    GM_ADDR sendBufGM_;
    GM_ADDR sendSizeGM_;
    GM_ADDR sendOffsetGM_;
    GM_ADDR recvBufGM_;
    uint64_t recvOffset_;
    uint64_t dataCount_;

    LocalTensor<int32_t> expertIdsLT_;
    LocalTensor<uint8_t> expertIdsU8LT_;
    LocalTensor<float> expertIdsF32LT_;
    LocalTensor<uint16_t> expertFreqLT_;
    LocalTensor<uint16_t> expertCumSumLT_;
    LocalTensor<int32_t> indexLT_;
    LocalTensor<float> sortedOutF32LT_;
    LocalTensor<int32_t> sortedOutI32LT_;
    LocalTensor<uint32_t> sortedIndex1LT_;
    LocalTensor<uint32_t> sortedIndex2LT_;
    LocalTensor<int32_t> recvCntLT_;
    LocalTensor<int32_t> moeExpertTokenNumsLT_;
    LocalTensor<int32_t> moeExpertRankTokenCumSumLT_;
    LocalTensor<float> scalesLT_;
    LocalTensor<float> rowMaxLT_;
    LocalTensor<float> tokenF32LT_;

    TBuf<> tempBuf_;
    TBuf<> sortedBuf_;
    uint32_t sortNum_;
    uint32_t sortRepeat_;
    uint32_t scalesCount_;
    uint32_t scaleInBytes_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> tokenQue_;  // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, BUFFER_NUM> tokenInQue_;                         // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, BUFFER_NUM> tokenOutQue_;                       // 量化使用，量化后的输出
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> tokenGatherQue_;

    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> sumOutBuf_;

    uint32_t axisMaxBs_{0};
    uint32_t maxBsKNum_{0};
    uint32_t activeBs_{0};
    uint32_t activeBsKNum_{0};
    uint32_t alignedBs_{0};
    bool isTokenMaskFlag_ = false;
    uint32_t sharedExpertNum_{0};
    uint32_t rankNumPerSharedExpert_{0};
};

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::InitGlobalAttrs(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    localExpertNum_ = moeExpertNum_ / moeExpertRankNum_;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    if (sharedExpertRankNum_ != 0) {
        sharedUsedAivNum_ = aivNum_ / (axisK_ + 1);  // 均等分， 取整
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
        shareRankRcvTokenCnt_ = axisBS_ * (epWorldSize_ / sharedExpertRankNum_);
    }
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;

    if (sharedExpertNum_ != 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }

    bskNum_ = axisBS_ * axisK_;
    isShareExpertRank_ = epRankId_ < sharedExpertRankNum_;
    sortNum_ = Align<uint32_t, uint32_t>(bskNum_, ONE_REPEAT_SORT_NUM);
    sortRepeat_ = Ceil<uint32_t, uint32_t>(bskNum_, ONE_REPEAT_SORT_NUM);

    axisMaxBs_ = axisBS_;
    maxBsKNum_ = bskNum_;
    if (tilingData->moeDistributeDispatchV2Info.globalBs != 0) {
        axisMaxBs_ = tilingData->moeDistributeDispatchV2Info.globalBs / epWorldSize_;
        maxBsKNum_ = axisMaxBs_ * axisK_;
    }
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    activeBs_ = axisBS_;
    activeBsKNum_ = bskNum_;
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::InitBuf()
{
    TBuf<> workspaceBuf;
    uint32_t idsB32Size = bskNum_ * sizeof(int32_t);
    uint32_t idsB16Size = bskNum_ * sizeof(half);
    uint32_t idsB8Size = bskNum_ * sizeof(uint8_t);
    uint32_t histoSize = VEC_LEN * sizeof(uint16_t);

    pipe_->InitBuffer(workspaceBuf, idsB32Size);
    expertIdsLT_ = workspaceBuf.Get<int32_t>();
    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(float));
    expertIdsF32LT_ = workspaceBuf.Get<float>();
    pipe_->InitBuffer(workspaceBuf, idsB8Size);
    expertIdsU8LT_ = workspaceBuf.Get<uint8_t>();

    pipe_->InitBuffer(workspaceBuf, histoSize);
    expertFreqLT_ = workspaceBuf.Get<uint16_t>();
    pipe_->InitBuffer(workspaceBuf, histoSize);
    expertCumSumLT_ = workspaceBuf.Get<uint16_t>();

    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(int32_t));
    indexLT_ = workspaceBuf.Get<int32_t>();

    pipe_->InitBuffer(tempBuf_, sortNum_ * sizeof(int32_t) * DUAL_DATA);
    pipe_->InitBuffer(sortedBuf_, sortNum_ * sizeof(int32_t) * DUAL_DATA);

    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(float));
    sortedOutF32LT_ = workspaceBuf.Get<float>();
    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(int32_t));
    sortedOutI32LT_ = workspaceBuf.Get<int32_t>();
    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(uint32_t));
    sortedIndex1LT_ = workspaceBuf.Get<uint32_t>();

    pipe_->InitBuffer(dstExpBuf_, idsB8Size);
    pipe_->InitBuffer(subExpBuf_, idsB16Size);
    pipe_->InitBuffer(sumOutBuf_, idsB16Size);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::InitCommAndStatus(
    GM_ADDR workspaceGM, const MoeDistributeDispatchV2TilingData *tilingData)
{
    __gm__ HcclCombineOpParam *context = (__gm__ HcclCombineOpParam *)(GetHcclContext<0>());
    // 结构体切换后，V1版本初始化方法不可用（已废弃），改用V2版本
    hccl_.InitV2((GM_ADDR)context, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchV2TilingData, mc2CcTiling1));

    sendBufGM_ = workspaceGM;
    sendSizeGM_ = sendBufGM_ + perRankDataSize_ * epWorldSize_;
    sendOffsetGM_ = sendSizeGM_ + epWorldSize_ * sizeof(uint64_t) * DUAL_DATA;

    GlobalTensor<int32_t> statusGT;
    GM_ADDR cclBuf = (GM_ADDR)context->windowsOut[0];

    uint64_t statusSize = aivNum_ * STATUS_OFFSET;
    statusGT.SetGlobalBuffer((__gm__ int32_t *)(cclBuf + aivId_ * STATUS_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGT);
    if (statusGT(0) == 0) {
        recvBufGM_ = cclBuf + statusSize;
        recvOffset_ = statusSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 1;
    } else {
        uint64_t halfDataSize = (context->winSize - statusSize) / HALF_DATA_SIZE_DIV;
        recvBufGM_ = cclBuf + statusSize + halfDataSize;
        recvOffset_ = statusSize + halfDataSize + epRankId_ * perRankDataSize_;
        statusGT(0) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGT);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::QuantInit()
{
    if constexpr ((QuantMode == UNQUANT) && IsSmoothScaleExist) {
        perTokenMergeSize_ += scaleInBytes_;
        perTokenInSize_ += scaleInBytes_;
        scaleOutBytes_ = scaleInBytes_;
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        perTokenMergeSize_ += sizeof(float);
        scaleOutBytes_ = sizeof(float);
    }

    perTokenCommSize_ = Align512<uint32_t>(perTokenMergeSize_);
    perRankDataSize_ = COUNT_OFFSET + perTokenCommSize_ * axisMaxBs_ * localExpertNum_;

    if constexpr (QuantMode > UNQUANT) {
        pipe_->InitBuffer(tokenInQue_, BUFFER_NUM, perTokenInSize_);
        pipe_->InitBuffer(tokenOutQue_, BUFFER_NUM, perTokenMergeSize_);

        TBuf<> tmpBuf;
        uint32_t tokenB32Size = axisH_ * sizeof(float);
        pipe_->InitBuffer(tmpBuf, tokenB32Size);
        tokenF32LT_ = tmpBuf.Get<float>();
        pipe_->InitBuffer(tmpBuf, tokenB32Size);
        scalesLT_ = tmpBuf.Get<float>();
        if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
            pipe_->InitBuffer(tmpBuf, UB_ALIGN);
            rowMaxLT_ = tmpBuf.Get<float>();
        }
    } else {
        pipe_->InitBuffer(tokenQue_, BUFFER_NUM, perTokenInSize_);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    pipe_ = pipe;

    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    aivId_ = GetBlockIdx();

    InitGlobalAttrs(tilingData);

    xGT_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGT_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGT_.SetGlobalBuffer((__gm__ ExpandXOutType *)expandXOut);
    expandIdxGT_.SetGlobalBuffer((__gm__ int32_t *)expandIdxOut);
    dynamicScaleGT_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsGT_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    sendCountsGT_.SetGlobalBuffer((__gm__ int32_t *)sendCountsOut);
    if constexpr (IsSmoothScaleExist) {
        scalesGT_.SetGlobalBuffer((__gm__ float *)scales);
    }
    if (isTokenMaskFlag_) {
        xActiveMaskGT_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    }

    InitBuf();

    perTokenInSize_ = axisH_ * sizeof(XType);
    perTokenOutSize_ = axisH_ * sizeof(ExpandXOutType);
    perTokenMergeSize_ = Align32(axisH_) * sizeof(ExpandXOutType);
    scaleInBytes_ =
        tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scaleOutBytes_ = 0;
    QuantInit();

    InitCommAndStatus(workspaceGM, tilingData);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::QuantStatic(
    LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal, int32_t expertIndex)
{
    Cast(tokenF32LT_, inLocal, RoundMode::CAST_NONE, axisH_);
    DataCopyParams scalesInParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(float)), 0U, 0U};
    DataCopyPadParams scalesPadParams = {false, 0, 0, 0};
    if constexpr (Std::IsSame<ExpandXOutType, int8_t>::value) {
        if (scalesCount_ == 1) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGT_);
            float scaleVal = scalesGT_(0);
            SyncFunc<AscendC::HardEvent::S_V>();
            Muls(tokenF32LT_, tokenF32LT_, scaleVal, axisH_);
        } else if (scalesCount_ == axisH_) {
            DataCopyPad(scalesLT_, scalesGT_, scalesInParams, scalesPadParams);
        } else {
            DataCopyPad(scalesLT_, scalesGT_[expertIndex * axisH_], scalesInParams, scalesPadParams);
        }
        if (scalesCount_ != 1) {
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Mul(tokenF32LT_, tokenF32LT_, scalesLT_, axisH_);
        }

        LocalTensor<int32_t> tokenI32LT = tokenF32LT_.ReinterpretCast<int32_t>();
        Cast(tokenI32LT, tokenF32LT_, RoundMode::CAST_RINT, axisH_);
        LocalTensor<half> tokenF16LT = tokenF32LT_.ReinterpretCast<half>();
        SetDeqScale((half)1.000000e+00f);
        Cast(tokenF16LT, tokenI32LT, RoundMode::CAST_ROUND, axisH_);
        Cast(outLocal, tokenF16LT, RoundMode::CAST_TRUNC, axisH_);
    } else if constexpr (Std::IsSame<ExpandXOutType, hifloat8_t>::value) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGT_);
        float scaleVal = scalesGT_(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(tokenF32LT_, tokenF32LT_, scaleVal, axisH_);
        Cast(outLocal, tokenF32LT_, RoundMode::CAST_ROUND, axisH_);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::QuantDynamicPerToken(
    LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal, int32_t expertIndex)
{
    float dynamicScale = 0.0;
    float maxVal = 0.0f;
    if constexpr (Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    } else if constexpr (Std::IsSame<ExpandXOutType, int8_t>::value) {
        maxVal = INT8_MAX_VALUE;
    }
    Cast(tokenF32LT_, inLocal, RoundMode::CAST_NONE, axisH_);
    if constexpr (IsSmoothScaleExist) {
        DataCopyParams scalesInParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(float)), 0U, 0U};
        DataCopyPadParams scalesPadParams = {true, 0, 0, 0};
        DataCopyPad(scalesLT_, scalesGT_[expertIndex * axisH_], scalesInParams, scalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(tokenF32LT_, tokenF32LT_, scalesLT_, axisH_);
    }

    Abs(scalesLT_, tokenF32LT_, axisH_);
    ReduceMax(rowMaxLT_, scalesLT_, scalesLT_, axisH_, false);
    SyncFunc<AscendC::HardEvent::V_S>();
    dynamicScale = static_cast<float>(maxVal / rowMaxLT_(0));
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(tokenF32LT_, tokenF32LT_, dynamicScale, axisH_);

    if constexpr (Std::IsSame<ExpandXOutType, int8_t>::value) {
        LocalTensor<int32_t> tokenI32LT = tokenF32LT_.ReinterpretCast<int32_t>();
        Cast(tokenI32LT, tokenF32LT_, RoundMode::CAST_RINT, axisH_);
        LocalTensor<half> tokenF16LT = tokenF32LT_.ReinterpretCast<half>();
        SetDeqScale((half)1.000000e+00f);
        Cast(tokenF16LT, tokenI32LT, RoundMode::CAST_ROUND, axisH_);
        Cast(outLocal, tokenF16LT, RoundMode::CAST_TRUNC, axisH_);
    } else if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value ||
                         Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
        Cast(outLocal, tokenF32LT_, RoundMode::CAST_RINT, axisH_);
    }

    LocalTensor<float> tokenF32Tmp = outLocal.template ReinterpretCast<float>();
    tokenF32Tmp(Align32<uint32_t>(axisH_) / sizeof(float)) = static_cast<float>(1.0) / dynamicScale;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::QuantProcess(
    LocalTensor<ExpandXOutType> &outLocal, LocalTensor<XType> &inLocal, int32_t expertIndex)
{
    if constexpr (QuantMode == STATIC_QUANT) {
        QuantStatic(outLocal, inLocal, expertIndex);
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        QuantDynamicPerToken(outLocal, inLocal, expertIndex);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::CalcTokenActiveMask()
{
    LocalTensor<half> maskHalfTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    alignedBs_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskHalfTensor = subExpBuf_.Get<half>();
    sumOutTensor = sumOutBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    // xActiveMask GM to UB
    DataCopyPad(maskInputTensor, xActiveMaskGT_, maskParams, maskCopyPadParams);
    // Sync
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    // Cast mask to type half for Sum calculation
    Cast(maskHalfTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, alignedBs_, axisBS_};
    Sum(sumOutTensor, maskHalfTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeBs_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    activeBsKNum_ = activeBs_ * axisK_;
    sortNum_ = Align<uint32_t, uint32_t>(activeBsKNum_, ONE_REPEAT_SORT_NUM);
    sortRepeat_ = Ceil<uint32_t, uint32_t>(activeBsKNum_, ONE_REPEAT_SORT_NUM);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline uint32_t
MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::CalcToSharedRankId(uint32_t sendCnt)
{
    uint32_t toSharedExpertIdx = sendCnt / activeBs_;
    uint32_t sharedExpertRankIdInGroup = epRankId_ % rankNumPerSharedExpert_;
    uint32_t toSharedRankId = rankNumPerSharedExpert_ * toSharedExpertIdx + sharedExpertRankIdInGroup;
    return toSharedRankId;
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::ShareScatterCopyTokens()
{
    uint32_t startTokenId = 0;
    uint32_t endTokenId = 0;
    uint32_t sendTokenNum = 0;
    uint32_t actualSendCnt = activeBs_ * sharedExpertNum_;
    SplitToCore(actualSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);

    GlobalTensor<ExpandXOutType> outTokenGT;
    DataCopyParams tokenInParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams tokenOutParams = {1U, static_cast<uint16_t>(perTokenMergeSize_), 0U, 0U};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};

    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t toRankId = CalcToSharedRankId(tokenIndex);
        GM_ADDR dstBaseGM =
            sendBufGM_ + COUNT_OFFSET * (toRankId + 1) + toRankId * axisMaxBs_ * localExpertNum_ * perTokenCommSize_;
        GM_ADDR tokenGM = dstBaseGM + tokenIndex * perTokenCommSize_;
        outTokenGT.SetGlobalBuffer((__gm__ ExpandXOutType *)tokenGM);
        ProcessToken(outTokenGT, tokenIndex, tokenInParams, padParams, tokenOutParams, scaleInParams, 0);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::SetShareExpertSendSize()
{
    uint32_t startRank = 0;
    uint32_t endRank = 0;
    uint32_t rankPerAiv = 0;
    SplitToCore(sharedExpertRankNum_, sharedUsedAivNum_, startRank, endRank, rankPerAiv, false);
    if (rankPerAiv == 0) {
        return;
    }

    TBuf<> sizeBuf;
    TBuf<> sendOffsetBuf;
    TBuf<> countBuf;
    uint32_t eachSize = Align32<uint32_t>(rankPerAiv * sizeof(uint64_t));
    uint32_t secondOffset = eachSize / sizeof(uint64_t);
    pipe_->InitBuffer(sizeBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sizeLT = sizeBuf.Get<uint64_t>();
    pipe_->InitBuffer(sendOffsetBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sendOffsetLT = sendOffsetBuf.Get<uint64_t>();
    pipe_->InitBuffer(countBuf, rankPerAiv * Align32<uint32_t>(sizeof(uint32_t)));
    LocalTensor<uint32_t> cntLT = countBuf.Get<uint32_t>();
    for (uint32_t i = startRank; i < endRank; ++i) {
        uint32_t count = activeBs_;
        uint32_t idx = i - startRank;
        uint32_t sendSize = count * perTokenCommSize_ + COUNT_OFFSET;
        uint32_t halfSize = static_cast<uint32_t>(sendSize / HALF_DATA_SIZE_DIV);

        cntLT(idx) = activeBs_;

        sizeLT(idx) = halfSize;
        sizeLT(idx + secondOffset) = sendSize - halfSize;

        sendOffsetLT(idx) = i * perRankDataSize_;
        sendOffsetLT(idx + secondOffset) = i * perRankDataSize_ + halfSize;
    }

    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<uint64_t> sendGT;
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendSizeGM_ + startRank * sizeof(uint64_t)));
    uint32_t cpSize = rankPerAiv * sizeof(uint64_t);
    DataCopyExtParams params = {BLOCK_COUNT, cpSize, Ceil32<uint32_t>(eachSize) - Ceil32<uint32_t>(cpSize),
                                static_cast<uint32_t>(epWorldSize_ * sizeof(uint64_t) - cpSize), 0};
    DataCopyPad(sendGT, sizeLT, params);
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendOffsetGM_ + startRank * sizeof(uint64_t)));
    DataCopyPad(sendGT, sendOffsetLT, params);

    GlobalTensor<uint32_t> sendCntGT;
    sendCntGT.SetGlobalBuffer((__gm__ uint32_t *)(sendBufGM_ + startRank * perRankDataSize_));
    uint32_t cntSize = sizeof(uint32_t);
    DataCopyExtParams cntParams = {static_cast<uint16_t>(rankPerAiv), cntSize, 0, perRankDataSize_ - cntSize, 0};
    DataCopyPad(sendCntGT, cntLT, cntParams);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::ShareScatter()
{
    if ((sharedExpertRankNum_ == 0) || (aivId_ >= aivNum_)) {
        return;
    }

    ShareScatterCopyTokens();
    SetShareExpertSendSize();
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::ProcessToken(
    GlobalTensor<ExpandXOutType> &outTokenGT, uint32_t tokenIndex, DataCopyParams &tokenInParams,
    DataCopyPadParams &padParams, DataCopyParams &tokenOutParams, DataCopyParams &scaleInParams, uint32_t expertIndex)
{
    if constexpr (QuantMode > UNQUANT) {
        auto tok = tokenInQue_.AllocTensor<XType>();
        // Initialize local tensor with zeros for mx/pertile quantations
        LocalTensor<uint8_t> singleByteTok = tok.template ReinterpretCast<uint8_t>();
        if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT)) {
            Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
        }
        SyncFunc<HardEvent::V_MTE2>();
        DataCopyPad(tok, xGT_[tokenIndex * axisH_], tokenInParams, padParams);
        tokenInQue_.EnQue(tok);
        tok = tokenInQue_.DeQue<XType>();
        auto quantTok = tokenOutQue_.AllocTensor<ExpandXOutType>();
        QuantProcess(quantTok, tok, expertIndex);
        tokenOutQue_.EnQue(quantTok);
        tokenInQue_.FreeTensor<XType>(tok);
        quantTok = tokenOutQue_.DeQue<ExpandXOutType>();
        DataCopyPad(outTokenGT, quantTok, tokenOutParams);
        tokenOutQue_.FreeTensor<ExpandXOutType>(quantTok);
    } else {
        auto tok = tokenQue_.AllocTensor<XType>();
        DataCopyPad(tok, xGT_[tokenIndex * axisH_], tokenInParams, padParams);
        if constexpr (IsSmoothScaleExist) {
            auto tmp = scalesGT_.ReinterpretCast<uint8_t>();
            DataCopyPad(tok[axisH_].template ReinterpretCast<uint8_t>(), tmp[tokenIndex * scaleInBytes_], scaleInParams,
                        padParams);
        }
        tokenQue_.EnQue(tok);
        tok = tokenQue_.DeQue<XType>();
        DataCopyPad(outTokenGT, tok, tokenOutParams);
        tokenQue_.FreeTensor(tok);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::MoeScatterCopyTokens()
{
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(activeBsKNum_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= activeBsKNum_) {
        return;
    }

    GlobalTensor<ExpandXOutType> outTokenGT;
    DataCopyParams tokenInParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams tokenOutParams = {1U, static_cast<uint16_t>(perTokenMergeSize_), 0U, 0U};
    DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};

    SyncFunc<HardEvent::V_S>();
    for (uint32_t tokenId = startTokenId; tokenId < endTokenId; ++tokenId) {
        uint32_t expertId = sortedOutI32LT_(tokenId);
        uint32_t toExpertRankId = expertId / localExpertNum_;
        uint32_t toRankId = toExpertRankId + sharedExpertRankNum_;
        uint32_t rankStartExpertId = toExpertRankId * localExpertNum_;
        uint32_t preCount = tokenId;
        if (rankStartExpertId > 0) {
            preCount = tokenId - expertCumSumLT_(rankStartExpertId - 1);
        }

        GM_ADDR tokenGM = sendBufGM_ + COUNT_OFFSET * (toRankId + 1) +
                          (toRankId * axisMaxBs_ * localExpertNum_ + preCount) * perTokenCommSize_;
        uint32_t tokenIndex = sortedIndex1LT_(tokenId) / axisK_;
        outTokenGT.SetGlobalBuffer((__gm__ ExpandXOutType *)tokenGM);
        uint32_t expertIndex = sharedExpertRankNum_ > 0 ? (expertId + 1) : expertId;
        ProcessToken(outTokenGT, tokenIndex, tokenInParams, padParams, tokenOutParams, scaleInParams, expertIndex);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::MoeScatter()
{
    MoeScatterCopyTokens();
    uint32_t startRank, endRank, rankPerAiv;
    SplitToCore(moeExpertRankNum_, moeUsedAivNum_, startRank, endRank, rankPerAiv);
    if (rankPerAiv == 0) {
        return;
    }

    TBuf<> sizeBuf;
    TBuf<> sendSizeBuf;
    TBuf<> countBuf;
    uint32_t eachSize = Align32<uint32_t>(rankPerAiv * sizeof(uint64_t));
    uint32_t secondOffset = eachSize / sizeof(uint64_t);
    pipe_->InitBuffer(sizeBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sizeLT = sizeBuf.Get<uint64_t>();
    pipe_->InitBuffer(sendSizeBuf, eachSize * DUAL_DATA);
    LocalTensor<uint64_t> sendOffsetLT = sendSizeBuf.Get<uint64_t>();
    uint32_t rankCntSize = Align32<uint32_t>(localExpertNum_ * sizeof(uint32_t));
    uint32_t rankCnt = rankCntSize / sizeof(uint32_t);
    pipe_->InitBuffer(countBuf, rankPerAiv * rankCntSize);
    LocalTensor<uint32_t> cntLT = countBuf.Get<uint32_t>();
    for (uint32_t i = startRank; i < endRank; ++i) {
        uint32_t idx = i - startRank;
        uint32_t count = 0;
        for (uint32_t j = 0; j < localExpertNum_; ++j) {
            uint32_t cnt = expertFreqLT_(i * localExpertNum_ + j);
            count += cnt;
            cntLT(idx * rankCnt + j) = cnt;
        }

        uint32_t sendSize = count * perTokenCommSize_ + COUNT_OFFSET;
        uint32_t halfSize = static_cast<uint32_t>(sendSize / HALF_DATA_SIZE_DIV);
        sizeLT(idx) = halfSize;
        sizeLT(idx + secondOffset) = sendSize - halfSize;

        sendOffsetLT(idx) = (i + sharedExpertRankNum_) * perRankDataSize_;
        sendOffsetLT(idx + secondOffset) = (i + sharedExpertRankNum_) * perRankDataSize_ + halfSize;
    }

    SyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<uint64_t> sendGT;
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendSizeGM_ + (startRank + sharedExpertRankNum_) * sizeof(uint64_t)));
    uint32_t cpSize1 = rankPerAiv * sizeof(uint64_t);
    DataCopyExtParams params = {BLOCK_COUNT, cpSize1, Ceil32<uint32_t>(eachSize) - Ceil32<uint32_t>(cpSize1),
                                static_cast<uint32_t>(epWorldSize_ * sizeof(uint64_t) - cpSize1), 0};
    DataCopyPad(sendGT, sizeLT, params);
    sendGT.SetGlobalBuffer((__gm__ uint64_t *)(sendOffsetGM_ + (startRank + sharedExpertRankNum_) * sizeof(uint64_t)));
    DataCopyPad(sendGT, sendOffsetLT, params);

    GlobalTensor<uint32_t> sendCntGT;
    sendCntGT.SetGlobalBuffer((__gm__ uint32_t *)(sendBufGM_ + (startRank + sharedExpertRankNum_) * perRankDataSize_));
    uint32_t cpSize2 = sizeof(uint32_t) * localExpertNum_;
    DataCopyExtParams cntParams = {static_cast<uint16_t>(rankPerAiv), cpSize2, rankCntSize - cpSize2,
                                   perRankDataSize_ - cpSize2, 0};
    DataCopyPad(sendCntGT, cntLT, cntParams);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::TokenScatter()
{
    if (activeBs_ != 0) {
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(activeBsKNum_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams padParams = {true, 0, 0, 0};
        DataCopyPad(expertIdsLT_, expertIdsGT_, copyParams, padParams);
        Duplicate(expertIdsF32LT_, MAX_FP32, sortNum_);
        CreateVecIndex(indexLT_, 0, sortNum_);
        Duplicate<uint16_t>(expertFreqLT_, (uint16_t)0, VEC_LEN);
        Duplicate<uint16_t>(expertCumSumLT_, (uint16_t)0, VEC_LEN);
        SyncFunc<HardEvent::MTE2_V>();
        Cast(expertIdsF32LT_, expertIdsLT_, RoundMode::CAST_RINT, activeBsKNum_);
        Muls(expertIdsF32LT_, expertIdsF32LT_, (float)-1, sortNum_);
        LocalTensor<float> concatLocal;
        LocalTensor<float> tempTensor = tempBuf_.Get<float>(GetSortLen<float>(sortNum_));
        Concat(concatLocal, expertIdsF32LT_, tempTensor, sortRepeat_);
        LocalTensor<uint32_t> indexU32LT = indexLT_.ReinterpretCast<uint32_t>();
        LocalTensor<float> sortedLocal = sortedBuf_.Get<float>(GetSortLen<float>(sortNum_));
        Sort<float, true>(sortedLocal, concatLocal, indexU32LT, tempTensor, sortRepeat_);
        Extract(sortedOutF32LT_, sortedIndex1LT_, sortedLocal, sortRepeat_);
        Muls(sortedOutF32LT_, sortedOutF32LT_, (float)-1, sortNum_);
        Cast(sortedOutI32LT_, sortedOutF32LT_, RoundMode::CAST_RINT, activeBsKNum_);
        Cast(expertIdsU8LT_, sortedOutI32LT_, RoundMode::CAST_NONE, activeBsKNum_);
        GetExpertFreq(expertFreqLT_, expertIdsU8LT_, activeBsKNum_);
        GetExpertCumSum(expertCumSumLT_, expertIdsU8LT_, activeBsKNum_);
    }
    if (aivId_ >= moeUsedAivNum_) {
        ShareScatter();
    } else if (aivId_ < moeUsedAivNum_) {
        MoeScatter();
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::SaveExpandIdx()
{
    TBuf<> workspaceBuf;
    pipe_->InitBuffer(workspaceBuf, sortNum_ * sizeof(uint32_t));
    sortedIndex2LT_ = workspaceBuf.Get<uint32_t>();
    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuf_.Get<float>(GetSortLen<float>(sortNum_));
    LocalTensor<uint32_t> indexU32LT = indexLT_.ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuf_.Get<float>(GetSortLen<float>(sortNum_));
    LocalTensor<int32_t> sortedIndex1I32LT = sortedIndex1LT_.ReinterpretCast<int32_t>();
    Duplicate(expertIdsF32LT_, MAX_FP32, sortNum_);
    Cast(expertIdsF32LT_, sortedIndex1I32LT, RoundMode::CAST_RINT, activeBsKNum_);
    Muls(expertIdsF32LT_, expertIdsF32LT_, (float)-1, sortNum_);
    Concat(concatLocal, expertIdsF32LT_, tempTensor, sortRepeat_);
    sortedLocal = sortedBuf_.Get<float>(GetSortLen<float>(sortNum_));
    Sort<float, true>(sortedLocal, concatLocal, indexU32LT, tempTensor, sortRepeat_);
    Extract(sortedOutF32LT_, sortedIndex2LT_, sortedLocal, sortRepeat_);
    uint32_t startCnt = 0;
    uint32_t endCnt = 0;
    uint32_t cntPerAiv = 0;
    SplitToCore(activeBsKNum_, aivNum_, startCnt, endCnt, cntPerAiv);
    if (cntPerAiv == 0) {
        return;
    }

    TBuf<> tmpBuf;
    pipe_->InitBuffer(tmpBuf, cntPerAiv * sizeof(int32_t));
    LocalTensor<int32_t> idxLT = tmpBuf.Get<int32_t>();

    SyncFunc<HardEvent::V_S>();
    for (uint32_t i = startCnt; i < endCnt; ++i) {
        uint32_t newIdx = i - startCnt;
        int expertId = expertIdsLT_(i);
        if (expertId >= localExpertNum_) {
            int32_t preCount = expertCumSumLT_(expertId - 1);
            int32_t oriIdx = sortedIndex2LT_(i) - preCount;
            int32_t startExpert = (expertId / localExpertNum_) * localExpertNum_;
            idxLT(newIdx) = oriIdx + (startExpert > 0 ? preCount - expertCumSumLT_(startExpert - 1) : 0);
        } else {
            idxLT(newIdx) = sortedIndex2LT_(i);
        }
    }
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyExtParams params = {1U, static_cast<uint32_t>(cntPerAiv * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expandIdxGT_[startCnt], idxLT, params);
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void
MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::TokenGatherInit(uint32_t localExpertNum)
{
    TBuf<> recvCntBuf;
    uint32_t alignSize = Align32<uint32_t>(epWorldSize_ * sizeof(int32_t));
    pipe_->InitBuffer(recvCntBuf, localExpertNum * alignSize);
    recvCntLT_ = recvCntBuf.Get<int32_t>();
    TBuf<> moeExpertRankTokenCumSumBuf;
    pipe_->InitBuffer(moeExpertRankTokenCumSumBuf, localExpertNum * alignSize);
    moeExpertRankTokenCumSumLT_ = moeExpertRankTokenCumSumBuf.Get<int32_t>();
    moeExpertTokenNumsLT_ = tempBuf_.Get<int32_t>();
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::Communication()
{
    SyncAll<true>();
    if (aivId_ == 0) {
        uint64_t localDataSize = COUNT_OFFSET;
        if (!isShareExpertRank_) {
            int32_t startExpertId = (epRankId_ - sharedExpertRankNum_) * localExpertNum_;
            for (int i = 0; i < localExpertNum_; ++i) {
                int32_t expertId = startExpertId + i;
                localDataSize += expertFreqLT_(expertId) * perTokenCommSize_;
            }
        } else {
            localDataSize += activeBs_ * perTokenCommSize_;
        }

        hcclHandleId_ = hccl_.AlltoAllvWrite<true>(sendBufGM_, sendOffsetGM_, sendSizeGM_, recvOffset_, localDataSize);
    }

    SaveExpandIdx();
    // Communication
    uint32_t localExpertNum = localExpertNum_;
    if (isShareExpertRank_) {
        localExpertNum = 1;
    }
    TokenGatherInit(localExpertNum);

    if (aivId_ == 0) {
        hccl_.Wait(hcclHandleId_);
    }

    SyncAll<true>();
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::UpdateShareRankOutput()
{
    uint32_t startRank = 0;
    uint32_t rankCnt = 0;

    if (aivId_ == 0) {
        expertTokenNumsGT_(0) = shareRankRcvTokenCnt_;
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTokenNumsGT_[0]);
        if (epRankId_ > 0) {
            rankCnt = epRankId_;
            Duplicate(moeExpertRankTokenCumSumLT_, 0, rankCnt);
        }
    } else {
        uint32_t endAivId = epWorldSize_ / sharedExpertRankNum_;
        startRank = epRankId_ + (aivId_ - 1) * sharedExpertRankNum_;
        if (aivId_ > endAivId) {
            rankCnt = 0;
        } else {
            rankCnt = aivId_ < endAivId ? sharedExpertRankNum_
                                        : epWorldSize_ - (epRankId_ + (endAivId - 1) * sharedExpertRankNum_);
        }
        Duplicate(moeExpertRankTokenCumSumLT_, static_cast<int32_t>(aivId_ * axisBS_), rankCnt);
    }

    if (rankCnt > 0) {
        SyncFunc<HardEvent::V_MTE3>();
        DataCopyParams cntParams = {1U, static_cast<uint16_t>(rankCnt * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(sendCountsGT_[startRank], moeExpertRankTokenCumSumLT_, cntParams);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::CopyScalesToOut(
    uint32_t currentTokenIndex, LocalTensor<ExpandXOutType> &quantTok, int32_t expertIndex)
{
    DataCopyParams scaleOutParams = {1U, static_cast<uint16_t>(scaleOutBytes_), 0U, 0U};
    if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT || ((QuantMode == UNQUANT) && IsSmoothScaleExist)) {
        auto scaleLT = quantTok[Align32<uint32_t>(axisH_)].template ReinterpretCast<uint8_t>();
        DataCopyPad(dynamicScaleGT_[currentTokenIndex * scaleOutBytes_], scaleLT, scaleOutParams);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::MoeGatherPrepare(
    uint32_t localExpertNum, uint32_t endRankIdx)
{
    uint32_t alignSize = Align32<uint32_t>(epWorldSize_ * sizeof(int32_t));
    uint32_t alignCnt = alignSize / sizeof(int32_t);

    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint32_t>(sizeof(int32_t)),
                                      static_cast<uint32_t>(perRankDataSize_ - sizeof(int32_t)), 0, 0};
    LoopModeParams loopParams;
    loopParams.loop1Size = localExpertNum;
    loopParams.loop1SrcStride = sizeof(int32_t);
    loopParams.loop1DstStride = alignSize;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    GlobalTensor<int32_t> recvCntGT;
    recvCntGT.SetGlobalBuffer((__gm__ int32_t *)recvBufGM_);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<int32_t, PaddingMode::Compact>(recvCntLT_, recvCntGT, copyInParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    PipeBarrier<PIPE_ALL>();
    GetReduceSum(moeExpertTokenNumsLT_, recvCntLT_, static_cast<uint16_t>(localExpertNum), alignCnt, epWorldSize_, 1);
    GetReduceSum(moeExpertRankTokenCumSumLT_, recvCntLT_, static_cast<uint16_t>(localExpertNum), alignCnt, endRankIdx,
                 alignCnt);
    SyncFunc<HardEvent::V_S>();
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - (aivNum_ - curUseAivNum);
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::UpdateMoeRankOutput(
    int32_t expertIndex, uint32_t startRank, uint32_t rankPerAiv, uint32_t expertOffset, uint32_t expertTokenNum)
{
    SyncFunc<HardEvent::S_MTE3>();
    DataCopyParams cntParams = {1U, static_cast<uint16_t>(rankPerAiv * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(sendCountsGT_[epWorldSize_ * expertIndex + startRank], moeExpertRankTokenCumSumLT_[expertOffset],
                cntParams);
    if (aivId_ == 0) {
        int64_t curExpertTokenNum = (expertTokenNumsType_ == 0) && (expertIndex > 0)
                                        ? (expertTokenNum + expertTokenNumsGT_(expertIndex - 1))
                                        : expertTokenNum;
        expertTokenNumsGT_(expertIndex) = curExpertTokenNum;
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsGT_[expertIndex]);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::TokenGatherProcess(
    int32_t sharedExpertRankNum, int32_t localExpertNum)
{
    // Same as Moe gather
    uint32_t startRank, endRank, rankPerAiv;
    SplitToCore(epWorldSize_, aivNum_, startRank, endRank, rankPerAiv);
    if (rankPerAiv == 0) {
        return;
    }

    Duplicate(indexLT_, 0, rankPerAiv);
    MoeGatherPrepare(localExpertNum, startRank + 1);

    DataCopyParams tokenInParams = {1U, static_cast<uint16_t>(perTokenMergeSize_), 0U, 0U};
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams tokenOutParams = {1U, static_cast<uint16_t>(perTokenOutSize_), 0U, 0U};

    GlobalTensor<ExpandXOutType> tokenGT;
    tokenGT.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    uint32_t alignWorldSize = Align32<uint32_t>(epWorldSize_ * sizeof(int32_t)) / sizeof(int32_t);
    uint32_t outPreExpertCount = 0;
    for (int k = 0; k < localExpertNum; ++k) {
        uint32_t expertId = k + (epRankId_ - sharedExpertRankNum) * localExpertNum;
        uint32_t expertOffset = k * alignWorldSize;
        moeExpertRankTokenCumSumLT_(expertOffset) += outPreExpertCount;
        uint32_t curCumSum = moeExpertRankTokenCumSumLT_(expertOffset);
        uint32_t outPreCount = curCumSum - recvCntLT_(expertOffset + startRank);

        for (uint32_t i = 0; i < rankPerAiv; ++i) {
            uint32_t curRankId = startRank + i;
            uint32_t recvCount = recvCntLT_(expertOffset + curRankId);
            if (i > 0) {
                curCumSum += recvCount;
                moeExpertRankTokenCumSumLT_(expertOffset + i) = curCumSum;
            }
            uint32_t rankInPreCnt = indexLT_(i);

            GM_ADDR baseAddr =
                recvBufGM_ + curRankId * perRankDataSize_ + COUNT_OFFSET + rankInPreCnt * perTokenCommSize_;
            for (uint32_t j = 0; j < recvCount; ++j) {
                tokenGT.SetGlobalBuffer((__gm__ ExpandXOutType *)(baseAddr + j * perTokenCommSize_));
                auto tok = tokenGatherQue_.AllocTensor<ExpandXOutType>();
                DataCopyPad(tok, tokenGT, tokenInParams, padParams);
                tokenGatherQue_.EnQue(tok);
                tok = tokenGatherQue_.DeQue<ExpandXOutType>();
                DataCopyPad(expandXOutGT_[outPreCount * axisH_], tok, tokenOutParams);
                CopyScalesToOut(outPreCount, tok, sharedExpertRankNum > 0 ? (expertId + 1) : expertId);
                tokenGatherQue_.FreeTensor(tok);
                outPreCount += 1;
            }
            indexLT_(i) = rankInPreCnt + recvCount;
        }
        uint32_t expertTokenNum = moeExpertTokenNumsLT_(k);
        UpdateMoeRankOutput(k, startRank, rankPerAiv, expertOffset, expertTokenNum);
        outPreExpertCount += expertTokenNum;
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::TokenGather()
{
    pipe_->InitBuffer(tokenGatherQue_, BUFFER_NUM, perTokenCommSize_);
    if (isShareExpertRank_) {
        // Shared expert ranks can be regarded as a special case of moe expert ranks
        // where sharedExpertRankNum=0 and localExpertNum=1
        TokenGatherProcess(0, 1);
    } else {
        TokenGatherProcess(sharedExpertRankNum_, localExpertNum_);
    }
}

template <TemplateMoeDistributeDispatchA5TypeClass>
__aicore__ inline void MoeDistributeDispatchA5<TemplateMoeDistributeDispatchA5TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (isTokenMaskFlag_) {
            CalcTokenActiveMask();
        }

        TokenScatter();
        Communication();
        TokenGather();

        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA5CCUImpl
#endif
