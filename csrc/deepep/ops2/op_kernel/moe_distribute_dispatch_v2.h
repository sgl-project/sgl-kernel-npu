/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_a2.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_H
#define MOE_DISTRIBUTE_DISPATCH_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_base.h"

namespace MoeDistributeDispatchA2Impl {
constexpr static uint8_t BUFFER_NUM = 2;             // 多buf
constexpr static uint32_t DATA_OFFSET = 512;         // 数据空间起始偏移
constexpr static uint32_t STATE_SIZE = 1024 * 1024;  // 1M
constexpr static uint32_t STATUS_ENTRY_COUNT = 32;
constexpr static uint32_t STATUS_SIZE = STATUS_ENTRY_COUNT * sizeof(int32_t);
constexpr static uint32_t UB_ALIGN = 32;  // UB按32字节对齐
constexpr static uint32_t BITS32_PER_BLOCK = UB_ALIGN / 4;
constexpr static uint32_t STATUS_BLOCK_COUNT = STATUS_ENTRY_COUNT / BITS32_PER_BLOCK;
constexpr static uint32_t FLAG_OFFSET = 24;
constexpr static uint32_t BW_ITEM_SIZE = 32;  // = sizeof(BatchWriteItem)
constexpr static uint32_t U64_PER_ITEM = BW_ITEM_SIZE / sizeof(uint64_t);
constexpr static uint32_t U32_PER_ITEM = BW_ITEM_SIZE / sizeof(uint32_t);
constexpr static uint32_t SKIP_OFFSET = 512;
constexpr static int32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint64_t MB_SIZE = 1024 * 1024;
template <typename T>
__aicore__ inline T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}
template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeA2Class \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2Func XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeDispatchV2
{
private:
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint64_t ALIGNED_LEN_256 = 256UL;
    constexpr static int32_t BITS_PER_BYTE = 8;
    constexpr static uint32_t REPEAT_BYTES = 256;
    constexpr static uint32_t BITS16_PER_BLOCK = UB_ALIGN / sizeof(int16_t);

public:
    __aicore__ inline MoeDistributeDispatchV2(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void IndexSort();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void GetStatusCumSum();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void ConstructBatchWriteInfo();
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void ReorderTokensPipeSet();
    __aicore__ inline void ReorderTokensPipeReset();
    __aicore__ inline void QuantProcess(uint32_t expertIndex, TEventID eventId);
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalVaildExpIdx(LocalTensor<int8_t> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<XType> windowInTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<ExpandXOutType> sendTokensTensor_;
    GlobalTensor<uint32_t> batchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<int8_t> xActiveMaskGMTensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<ExpandXOutType> xOutPingTensor_;
    LocalTensor<ExpandXOutType> xOutPongTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<int32_t> expertTokenNumsW64Tensor_;
    LocalTensor<uint32_t> expertCumsumTensor_;
    LocalTensor<int32_t> vaildExpIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    TBuf<> dynamicScalesBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_;  // gather mask输出buf
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> tBuf_;

    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_;  // 这个输出没有使用
    GM_ADDR epRecvCountsGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR batchWriteInfo_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0};         // aiv id
    uint32_t moeExpertNum_{0};  // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t bufferSizePerRank_{0};
    uint32_t hSize_{0};
    uint32_t hQuantSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleParamPad_{0};
    uint32_t axisHCommu_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t localMoeExpertNumAlign_{0};
    uint32_t dataSizePerRank_{0};
    uint32_t dataSize_{0};
    uint32_t bufferChosen_{0};
    uint32_t totalSize_{0};
    uint64_t activeMaskBsCnt_{0};
    uint32_t expertTokenNumsType_{0};
    int32_t zeroComputeExpertNum_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint32_t leftUbSize_{TBUF_SIZE};
    uint32_t baseBuffOffset_{0};
    uint32_t xActiveMaskSize_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool isQuant_ = false;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    GM_ADDR tilingGM)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    auto tiling = (__gm__ MoeDistributeDispatchV2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    winContext_ = (__gm__ HcclOpResParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchV2Info.epRankId;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData.moeDistributeDispatchV2Info.h;
    axisK_ = tilingData.moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData.moeDistributeDispatchV2Info.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchV2Info.expertTokenNumsType;
    isTokenMaskFlag_ = tilingData.moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    zeroComputeExpertNum_ = tilingData.moeDistributeDispatchV2Info.zeroComputeExpertNum;
    totalSize_ = winContext_->winSize / 2;  // 2G / 2 = 1G
    dataSize_ = totalSize_ - STATE_SIZE;    // 1G - 1M
    dataSizePerRank_ = dataSize_ / worldSize_;
    moeExpertNum_ = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    localMoeExpertNumAlign_ = (localMoeExpertNum_ + BITS32_PER_BLOCK - 1) / BITS32_PER_BLOCK * BITS32_PER_BLOCK;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSize_));
    bufferChosen_ = bufferChosenGlobal_(0);  // 魔数

    windowInGM_ = windowInGM_ + totalSize_ * bufferChosen_;
    windowOutGM_ = windowOutGM_ + totalSize_ * bufferChosen_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    windowInTensor_.SetGlobalBuffer((__gm__ XType *)(windowInGM_));
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_));
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowInGM_));
    sendTokensTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_));

    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ int8_t *)xActiveMask);
    expandIdxOutGM_ = expandIdxOut;
    expertTokenNumsOutGM_ = expertTokenNumsOut;
    epRecvCountsGM_ = epRecvCountsOut;

    isQuant_ = StaticQuant | DynamicQuant;
    hSize_ = axisH_ * sizeof(XType);
    hQuantSize_ = axisH_ * sizeof(ExpandXOutType);  // 如有量化，需要量化后通信
    scaleParamPad_ = (isQuant_ ? 32 : 0);           // 预留32B给量化参数，实际只使用了4B(fp32)
    hCommuSize_ = hQuantSize_ + scaleParamPad_;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    bufferSizePerRank_ = 32 * hSize_;

    batchWriteInfo_ = workspaceGM;
    batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t *)(batchWriteInfo_), worldSize_ * U32_PER_ITEM);

    tpipe_->InitBuffer(statusBuf_, worldSize_ * STATUS_ENTRY_COUNT * sizeof(int32_t));  // worldsize * 32B
    leftUbSize_ -= worldSize_ * STATUS_ENTRY_COUNT * sizeof(int32_t);
    statusTensor_ = statusBuf_.Get<int32_t>();  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT);  // 8 = UB_ALIGN / sizeof(int32_t)

    uint64_t mask[2] = {0x0100000001000000, 0};
    Duplicate<int32_t>(statusTensor_, FLAG_VALUE, mask, worldSize_ * STATUS_ENTRY_COUNT / 64, 1, 8);
    if (isQuant_) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    }

    tpipe_->InitBuffer(batchWriteInfoBuf_, worldSize_ * BW_ITEM_SIZE);
    leftUbSize_ -= worldSize_ * BW_ITEM_SIZE;
    // Ensure not less than REPEAT_BYTES for TokenActiveMaskCal
    uint32_t expertIdsBufSize = Std::max(
        static_cast<uint32_t>((expertIdsCnt_ * sizeof(int32_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN), REPEAT_BYTES);
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize);
    leftUbSize_ -= expertIdsBufSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    tpipe_->InitBuffer(expertCountBuf_, expertIdsBufSize);
    leftUbSize_ -= expertIdsBufSize;
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();

    tpipe_->InitBuffer(gatherMaskOutBuf_, (localMoeExpertNumAlign_ * worldSize_ + moeExpertNum_) * sizeof(float));
    leftUbSize_ -= (localMoeExpertNumAlign_ * worldSize_ + moeExpertNum_) * sizeof(float);
    tpipe_->InitBuffer(tBuf_, leftUbSize_);

    uint64_t stateSizeMaxSize =
        2 * STATE_SIZE;  // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) + STATE_SIZE，近似计算使用2 * STATE_SIZE
    uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_ > axisK_ ? axisK_ : localMoeExpertNum_) * axisH_ *
                               sizeof(uint16_t) +
                           stateSizeMaxSize) *
                          BUFFER_NUM;  // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小

    assert(winContext_->winSize >= winSizeMin,
           "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
        epWorldSize:%u, epRankId:%u, moeExpertNum:%u, quantMode:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
        isQuant:%d, totalUbSize:%llu, expertTokenNumsType:%u\n",
           winContext_->winSize / MB_SIZE, winSizeMin / MB_SIZE, tilingData.moeDistributeDispatchV2Info.epWorldSize,
           tilingData.moeDistributeDispatchV2Info.epRankId, tilingData.moeDistributeDispatchV2Info.moeExpertNum,
           tilingData.moeDistributeDispatchV2Info.quantMode, tilingData.moeDistributeDispatchV2Info.globalBs,
           tilingData.moeDistributeDispatchV2Info.bs, tilingData.moeDistributeDispatchV2Info.k,
           tilingData.moeDistributeDispatchV2Info.h, tilingData.moeDistributeDispatchV2Info.aivNum,
           tilingData.moeDistributeDispatchV2Info.isQuant, tilingData.moeDistributeDispatchV2Info.totalUbSize,
           tilingData.moeDistributeDispatchV2Info.expertTokenNumsType);
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (tilingData.moeDistributeDispatchV2Info.isTokenMask) {
        TokenActiveMaskCal();
    }
    vaildExpIndexTensor_ = tBuf_.GetWithOffset<int32_t>(RoundUp(expertIdsCnt_, BITS32_PER_BLOCK), baseBuffOffset_);
    CreateVecIndex(vaildExpIndexTensor_, 0, RoundUp(expertIdsCnt_, BITS32_PER_BLOCK));
    baseBuffOffset_ += RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(int32_t);
    xActiveMaskSize_ = Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE;
    LocalTensor<uint8_t> gatherMaskTensorInt8 = tBuf_.GetWithOffset<uint8_t>(xActiveMaskSize_, baseBuffOffset_);
    baseBuffOffset_ += xActiveMaskSize_;
    gatherMaskTensor_ = gatherMaskTensorInt8.template ReinterpretCast<uint32_t>();

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) {
        baseBuffOffset_ = RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(uint32_t);
        return;
    }

    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }
    baseBuffOffset_ = RoundUp(expertIdsCnt_, BITS32_PER_BLOCK) * sizeof(uint32_t);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::QuantProcess(uint32_t expertIndex,
                                                                                    TEventID eventId)
{
    float dynamicScale = 0.0;
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<float> floatLocalTemp = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);
    LocalTensor<float> smoothScalesTensor = tBuf_.GetWithOffset<float>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    /*
        <xType> xInTensor_ --> <float> floatLocalTemp --> <int32_t> int32LocalTemp --> <half>halfLocalTemp
        fp32先转int32再转fp16 -- 对标A3实现
    */
    SyncFunc<AscendC::HardEvent::MTE3_V>();  // QuantProcess没开ping-pong XOut的ping-pong未生效
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        DataCopy(smoothScalesTensor, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesTensor;  // 复用
        rowMaxTensor_ = tBuf_.GetWithOffset<float>(BITS32_PER_BLOCK, baseBuffOffset);
        baseBuffOffset += UB_ALIGN;
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMax(rowMaxTensor_, floatLocalAbsTemp, floatLocalAbsTemp, axisH_, false);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / rowMaxTensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    SetFlag<HardEvent::V_MTE3>(eventId);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    dynamicScale = 1 / dynamicScale;
    floatLocalTemp.SetValue(axisH_ / sizeof(float), dynamicScale);  // int8->float32
    SetFlag<HardEvent::S_MTE3>(eventId);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::IndexSort()
{
    uint32_t activeExpertIds = activeMaskBsCnt_ * axisK_;
    DataCopyExtParams copyExpertIdsParams{1, static_cast<uint32_t>(activeExpertIds * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, copyExpertIdsParams, padParams);
    Duplicate(expertCountTensor_, 0, RoundUp(activeExpertIds, BITS32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    // 24个核
    for (uint32_t tokenIndex = 0; tokenIndex < sendToMoeExpTokenCnt_; ++tokenIndex) {
        int32_t expertIdx = vaildExpIndexTensor_(tokenIndex);
        int32_t expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t expertOffsetInRank = expertId % localMoeExpertNum_;
        expertCountTensor_(expertIdx) = statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank);
        statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank)++;
    }
    uint32_t baseBuffOffset = baseBuffOffset_;
    expertCumsumTensor_ = gatherMaskOutBuf_.Get<uint32_t>();
    expertCumsumTensor_.SetValue(0, 0);
    for (uint32_t expertId = 1; expertId < moeExpertNum_; expertId++) {
        int32_t rankId = (expertId - 1) / localMoeExpertNum_;
        int32_t expertOffsetInRank = (expertId - 1) % localMoeExpertNum_;
        uint32_t count = statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffsetInRank);
        uint32_t preSum = expertCumsumTensor_(expertId - 1);
        expertCumsumTensor_(expertId) = count + preSum;
    }

    expertCumsumTensor_(moeExpertNum_) = sendToMoeExpTokenCnt_;

    if (aivId_ == aivNum_ - 1) {  // 最后一个核
        SyncFunc<AscendC::HardEvent::S_MTE3>();

        GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutGM_);
        DataCopyPad(expandIdxGMTensor, expertCountTensor_, copyExpertIdsParams);

        DataCopy(windowInstatusTensor_[rankId_ * dataSizePerRank_ / sizeof(int32_t)],
                 statusTensor_[rankId_ * STATUS_ENTRY_COUNT], STATUS_ENTRY_COUNT);

        LocalTensor<int32_t> flagTmpLocal = tBuf_.GetWithOffset<int32_t>(BITS32_PER_BLOCK, baseBuffOffset);
        Duplicate<int32_t>(flagTmpLocal, FLAG_VALUE, UB_ALIGN / sizeof(int32_t));

        for (uint32_t rankId = 0; rankId < worldSize_; rankId++) {
            uint64_t rankOffset = rankId * dataSizePerRank_ / sizeof(int32_t);
            DataCopy(sendStatusTensor_[rankOffset], statusTensor_[rankId * STATUS_ENTRY_COUNT], STATUS_ENTRY_COUNT);

            uint32_t startExpertId = rankId * localMoeExpertNum_;
            uint32_t tokenCount =
                expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
            uint64_t dataFlagOffset =
                rankOffset + (DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sendStatusTensor_[dataFlagOffset], flagTmpLocal, copyFlagParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ReorderTokensPipeSet()
{
    if constexpr (StaticQuant || DynamicQuant) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ReorderTokensPipeReset()
{
    if constexpr (StaticQuant || DynamicQuant) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    } else {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ReorderTokens()
{
    uint32_t sendTokenNum = sendToMoeExpTokenCnt_ / aivNum_;
    uint32_t remainderTokenNum = sendToMoeExpTokenCnt_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    GlobalTensor<ExpandXOutType> sendTokensGlobal;
    ReorderTokensPipeSet();
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<XType> xInPingTensor;
    LocalTensor<XType> xInPongTensor;
    if (isQuant_) {
        xInPingTensor = tBuf_.GetWithOffset<XType>(axisH_, baseBuffOffset);
        baseBuffOffset += axisH_ * sizeof(XType);
        xInPongTensor = tBuf_.GetWithOffset<XType>(axisH_, baseBuffOffset);
        baseBuffOffset += axisH_ * sizeof(XType);
    }
    LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    baseBuffOffset_ = baseBuffOffset;

    int32_t expertId = 0;
    int32_t expertIdx = 0;
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        TEventID eventId = (tokenIndex & 1) ? EVENT_ID0 : EVENT_ID1;
        expertIdx = vaildExpIndexTensor_(tokenIndex);
        expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t startExpertId = rankId * localMoeExpertNum_;
        uint32_t expertOffset = expertCumsumTensor_(expertId) - expertCumsumTensor_(startExpertId);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        int32_t tokenOffset = expertCountTensor_(expertIdx);
        sendTokensGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId * dataSizePerRank_ + DATA_OFFSET));
        if constexpr (StaticQuant || DynamicQuant) {
            xInTensor_ = (eventId & 1) ? xInPingTensor : xInPongTensor;
            xOutTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopy(xInTensor_, xGMTensor_[expertIdx / axisK_ * axisH_], axisH_);  // 约束对齐
            SetFlag<HardEvent::MTE2_V>(eventId);

            QuantProcess(expertId, eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::S_MTE3>(eventId);
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xOutTensor_, axisHCommu_);
        } else {
            xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xTmpTensor_, xGMTensor_[expertIdx / axisK_ * axisH_],
                     axisH_);  // 约束对齐 tokenIndex / axisK_ * axisH_
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xTmpTensor_, axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
    }
    baseBuffOffset_ = 0;  // 释放零和专家相关ub
    ReorderTokensPipeReset();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    batchWriteU32Tensor_ = batchWriteInfoBuf_.Get<uint32_t>();
    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();

    uint32_t batchWriteDataType = static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8);

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t rankIndex = startRankId; rankIndex < endRankId; ++rankIndex) {
        uint32_t startExpertId = rankIndex * localMoeExpertNum_;
        uint32_t currentIndex = rankIndex - startRankId;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GM_ADDR rankGM = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(rankIndex) + totalSize_ * bufferChosen_ +
                                            (dataSizePerRank_ * rankId_));
        GM_ADDR localBuf = (__gm__ uint8_t *)(windowOutGM_ + dataSizePerRank_ * rankIndex);
        uint64_t batchWriteDataSize = DATA_OFFSET + tokenCount * hCommuSize_ + sizeof(int32_t) + SKIP_OFFSET;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM) = (uint64_t)localBuf;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 1) = (uint64_t)rankGM;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 2) = batchWriteDataSize;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 6) = batchWriteDataType;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 7) = rankIndex;
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(batchWriteInfoTensor_[startRankId * U32_PER_ITEM], batchWriteU32Tensor_, batchWriteItemNum * U32_PER_ITEM);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
{
    LocalTensor<int8_t> xActiveMaskInt8Tensor;
    LocalTensor<half> xActiveMaskHalfTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<uint8_t> tempTensor;
    uint32_t axisBsAlignSize = (axisBS_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t baseBuffOffset = baseBuffOffset_;
    xActiveMaskInt8Tensor = tBuf_.GetWithOffset<int8_t>(axisBsAlignSize, baseBuffOffset);
    baseBuffOffset += axisBsAlignSize * sizeof(int8_t);
    xActiveMaskHalfTensor = tBuf_.GetWithOffset<half>(axisBsAlignSize, baseBuffOffset);
    baseBuffOffset += axisBsAlignSize * sizeof(half);
    sumOutTensor = tBuf_.GetWithOffset<half>(UB_ALIGN, baseBuffOffset);
    baseBuffOffset += UB_ALIGN * sizeof(half);
    tempTensor = expertCountBuf_.Get<uint8_t>();
    DataCopyExtParams xActiveMaskParams = {1U, axisBS_, 0U, 0U, 0U};
    DataCopyPadExtParams<int8_t> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskInt8Tensor, xActiveMaskGMTensor_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(xActiveMaskHalfTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, xActiveMaskHalfTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void
MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::CalVaildExpIdx(LocalTensor<int8_t> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<half> tempTensor = tBuf_.GetWithOffset<half>(calCnt, baseBuffOffset);
    baseBuffOffset += calCnt * sizeof(half);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.template ReinterpretCast<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor =
        tBuf_.GetWithOffset<int32_t>(RoundUp(curMaskCnt, BITS32_PER_BLOCK), baseBuffOffset);

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_, ALIGNED_LEN_256) * ALIGNED_LEN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, RoundUp(curMaskCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    GatherMask(vaildExpIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ExpertActiveMaskCal()
{
    // 计算vaildExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    uint32_t tempSize = ((expertIdsCnt_ * sizeof(int8_t) + 1) / UB_ALIGN + 1) * UB_ALIGN / sizeof(int8_t);
    LocalTensor<int8_t> maskInputTensor = tBuf_.GetWithOffset<int8_t>(tempSize, baseBuffOffset_);
    baseBuffOffset_ += tempSize;
    DataCopyPadExtParams<int8_t> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(int8_t)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalVaildExpIdx(maskInputTensor);
    baseBuffOffset_ -= tempSize;
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::MaskZeroComputeExpert(uint32_t maskCnt)
{
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
    uint32_t tmpTokenCnt = static_cast<uint32_t>(sendToMoeExpTokenCnt_);
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<int32_t> expertsIndexTensor =
        tBuf_.GetWithOffset<int32_t>(RoundUp(tmpTokenCnt, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += RoundUp(tmpTokenCnt, BITS32_PER_BLOCK) * sizeof(int32_t);
    int32_t maskTensorInt16Cnt = Ceil(tmpTokenCnt, UB_ALIGN / 2);
    LocalTensor<uint32_t> maskTensorInt32 =
        tBuf_.GetWithOffset<uint32_t>(RoundUp(tmpTokenCnt, UB_ALIGN), baseBuffOffset);  // expertCountBuf_
    LocalTensor<uint8_t> maskTensorInt8 = maskTensorInt32.template ReinterpretCast<uint8_t>();
    baseBuffOffset += RoundUp(tmpTokenCnt, UB_ALIGN) * sizeof(uint32_t);
    LocalTensor<half> expertIdsTensorCast =
        tBuf_.GetWithOffset<half>(RoundUp(tmpTokenCnt, BITS16_PER_BLOCK), baseBuffOffset);  // expertCountBuf_
    baseBuffOffset += RoundUp(tmpTokenCnt, BITS16_PER_BLOCK) * sizeof(half);
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {
        1U, static_cast<uint32_t>(RoundUp(tmpTokenCnt, BITS32_PER_BLOCK) * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, expertIdsTensor_, RoundMode::CAST_NONE, RoundUp(tmpTokenCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(tmpTokenCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    // CompareScalar需要保证元素所占空间256字节对齐。
    uint32_t calcCnt = Ceil(sendToMoeExpTokenCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    // 逐元素比较一个tensor中的元素和另一个Scalar的大小，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT,
                  calcCnt);
    PipeBarrier<PIPE_V>();
    // ?
    LocalTensor<uint16_t> maskTensorInt16 = maskTensorInt32.template ReinterpretCast<uint16_t>();  // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTensor_.template ReinterpretCast<uint16_t>();  // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16，
     * gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, RoundUp(tmpTokenCnt, BITS32_PER_BLOCK));
    PipeBarrier<PIPE_V>();
    GatherMask(vaildExpIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::GenerateGatherMaskTensor(uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) {  // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) {  // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
    isExpertMaskFlag_ = true;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::SendToMoeExpert()
{
    ConstructBatchWriteInfo();
    SyncAll<true>();

    if (aivId_ == 0) {
        HcclHandle batchWriteResult = hccl_.BatchWrite<true>(batchWriteInfo_, worldSize_);
        bufferChosenGlobal_(0) = bufferChosen_ ^ 1;
    }
    if (aivId_ == aivNum_ - 1) {
        uint32_t baseBuffOffset = baseBuffOffset_;
        LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
        baseBuffOffset += hCommuSize_;
        LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
        baseBuffOffset += hCommuSize_;

        uint32_t startExpertId = rankId_ * localMoeExpertNum_;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GlobalTensor<ExpandXOutType> currRankWindowInGlobal;
        GlobalTensor<ExpandXOutType> currRankWindowOutGlobal;
        currRankWindowInGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowInGM_ + rankId_ * dataSizePerRank_ + DATA_OFFSET));
        currRankWindowOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId_ * dataSizePerRank_ + DATA_OFFSET));
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t currTokenIdx = 0; currTokenIdx < tokenCount; currTokenIdx++) {
            TEventID eventId = (currTokenIdx & 1) ? EVENT_ID0 : EVENT_ID1;
            xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xTmpTensor_, currRankWindowOutGlobal[currTokenIdx * axisHCommu_], axisHCommu_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(currRankWindowInGlobal[currTokenIdx * axisHCommu_], xTmpTensor_, axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        uint64_t dataFlagOffset =
            (rankId_ * dataSizePerRank_ + DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        windowInstatusTensor_(dataFlagOffset) = FLAG_VALUE;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowInstatusTensor_[dataFlagOffset]);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    uint32_t batchWriteItemNum = worldSize_ / aivNum_;
    uint32_t remainderItemNum = worldSize_ % aivNum_;
    uint32_t startRankId = batchWriteItemNum * aivId_;
    if (aivId_ < remainderItemNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        batchWriteItemNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderItemNum;
    }
    uint32_t endRankId = startRankId + batchWriteItemNum;

    if (batchWriteItemNum == 0) {
        SyncAll<true>();
        return;
    }

    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    LocalTensor<int32_t> dataFlagLocal = tBuf_.GetWithOffset<int32_t>(BITS32_PER_BLOCK, baseBuffOffset_);
    SyncFunc<AscendC::HardEvent::S_MTE2>();

    for (uint32_t rankId = startRankId; rankId < endRankId; rankId++) {
        int32_t statusFlag = 0;
        int32_t dataFlag = 0;
        while (statusFlag != FLAG_VALUE) {
            DataCopy(statusTensor_[rankId * STATUS_ENTRY_COUNT],
                     windowInstatusTensor_[rankId * dataSizePerRank_ / sizeof(int32_t)], STATUS_ENTRY_COUNT);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            statusFlag = statusTensor_(rankId * STATUS_ENTRY_COUNT + FLAG_OFFSET);
            PipeBarrier<PIPE_MTE2>();
        }

        uint32_t tokenCount = 0;
        for (int32_t expertOffset = 0; expertOffset < localMoeExpertNum_; expertOffset++) {
            tokenCount += statusTensor_(rankId * STATUS_ENTRY_COUNT + expertOffset);
        }

        uint64_t dataFlagOffset =
            (rankId * dataSizePerRank_ + DATA_OFFSET + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        while (dataFlag != FLAG_VALUE) {
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            dataFlag = dataFlagLocal(0);
            PipeBarrier<PIPE_MTE2>();
        }
        windowInstatusTensor_(dataFlagOffset) = 0;
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::GetStatusCumSum()
{
    uint32_t baseBuffOffset = baseBuffOffset_;
    uint32_t srcStrideU32 = dataSizePerRank_ - STATUS_SIZE;
    DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, srcStrideU32, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(statusTensor_, windowInstatusTensor_, copyStatusParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int32_t> epRecvCountsTempLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(localMoeExpertNumAlign_ * worldSize_, 0);
    LocalTensor<int32_t> epRecvCountsOutLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_, localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));
    uint16_t srcStrideU16 = (STATUS_ENTRY_COUNT - localMoeExpertNumAlign_) / BITS32_PER_BLOCK;
    uint16_t worldSizeU16 = (uint16_t)worldSize_;
    DataCopyParams copyParamsMultiple{worldSizeU16, static_cast<uint16_t>(localMoeExpertNumAlign_ / BITS32_PER_BLOCK),
                                      srcStrideU16, 0};
    DataCopy(epRecvCountsTempLocal, statusTensor_, copyParamsMultiple);
    uint64_t mask4Adds = localMoeExpertNum_;
    PipeBarrier<PIPE_V>();
    for (uint32_t rankIndex = 1; rankIndex < worldSize_; ++rankIndex) {
        uint32_t statusOffset = rankIndex * localMoeExpertNumAlign_;
        Add(epRecvCountsTempLocal[statusOffset], epRecvCountsTempLocal[statusOffset - localMoeExpertNumAlign_],
            epRecvCountsTempLocal[statusOffset], mask4Adds, 1, {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint32_t> patternLocal = tBuf_.GetWithOffset<uint32_t>(localMoeExpertNumAlign_, baseBuffOffset);
    baseBuffOffset += localMoeExpertNumAlign_ * sizeof(uint32_t);
    Duplicate<uint32_t>(patternLocal, 0, localMoeExpertNumAlign_);
    SyncFunc<AscendC::HardEvent::V_S>();
    patternLocal(0) = 1;
    srcStrideU16 = localMoeExpertNumAlign_ * sizeof(int32_t) / UB_ALIGN;
    int32_t previousSum = 0;
    uint64_t rsvdCnt = 0;
    mask4Adds = worldSize_;
    uint32_t mask4Gather = localMoeExpertNumAlign_;
    for (uint32_t expertIndex = 0; expertIndex < localMoeExpertNum_; expertIndex++) {
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(epRecvCountsOutLocal[expertIndex * worldSize_], epRecvCountsTempLocal, patternLocal, true,
                   mask4Gather, {1, worldSizeU16, srcStrideU16, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        Adds(epRecvCountsOutLocal[expertIndex * worldSize_], epRecvCountsOutLocal[expertIndex * worldSize_],
             previousSum, worldSize_);
        SyncFunc<AscendC::HardEvent::V_S>();
        previousSum = epRecvCountsOutLocal(expertIndex * worldSize_ + worldSize_ - 1);
        patternLocal(0) = patternLocal(0) << 1;
    }
    if (aivId_ == aivNum_ - 1) {
        LocalTensor<int32_t> expertTokenNumsW64Tensor =
            tBuf_.GetWithOffset<int32_t>(localMoeExpertNum_ * 2, baseBuffOffset);
        if (expertTokenNumsType_ == 0) {
            mask4Gather = worldSize_;
            patternLocal(0) = 0;
            patternLocal((worldSize_ - 1) / 32) = 1 << ((worldSize_ - 1) % 32);
            srcStrideU16 = worldSize_ * sizeof(int32_t) / UB_ALIGN;
            SyncFunc<AscendC::HardEvent::S_V>();
            GatherMask(epRecvCountsTempLocal, epRecvCountsOutLocal, patternLocal, true, mask4Gather,
                       {1, static_cast<uint16_t>(localMoeExpertNum_), srcStrideU16, 0}, rsvdCnt);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor(i * 2) = epRecvCountsTempLocal(i);
                expertTokenNumsW64Tensor(i * 2 + 1) = 0;
            }
        } else {
            uint32_t tokenCountOffset = (worldSize_ - 1) * localMoeExpertNumAlign_;
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Tensor(i * 2) = epRecvCountsTempLocal(tokenCountOffset + i);
                expertTokenNumsW64Tensor(i * 2 + 1) = 0;
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t *)(expertTokenNumsOutGM_));
        DataCopyExtParams copyPadParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokenNumsGlobal, expertTokenNumsW64Tensor, copyPadParams);

        GlobalTensor<int32_t> epRecvCountsGlobal;
        epRecvCountsGlobal.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(epRecvCountsGlobal, epRecvCountsOutLocal, moeExpertNum_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    uint32_t dynamicScalesLocalIdx = 0;
    GetStatusCumSum();
    LocalTensor<int32_t> epRecvCountsOutLocal =
        gatherMaskOutBuf_.GetWithOffset<int32_t>(moeExpertNum_, localMoeExpertNumAlign_ * worldSize_ * sizeof(int32_t));
    uint32_t dealRankNum = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    uint32_t startRankId = dealRankNum * aivId_;
    if (aivId_ < remainderRankNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        dealRankNum += 1;
        startRankId += aivId_;
    } else {
        startRankId += remainderRankNum;
    }
    uint32_t endRankId = startRankId + dealRankNum;

    GlobalTensor<ExpandXOutType> currRankWindowGlobal;
    uint32_t baseBuffOffset = baseBuffOffset_;
    LocalTensor<float> dynamicScalesTensor =
        tBuf_.GetWithOffset<float>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += RoundUp(axisBS_, BITS32_PER_BLOCK) * sizeof(float);
    LocalTensor<ExpandXOutType> xOutPingTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;
    LocalTensor<ExpandXOutType> xOutPongTensor = tBuf_.GetWithOffset<ExpandXOutType>(axisHCommu_, baseBuffOffset);
    baseBuffOffset += hCommuSize_;

    for (uint32_t index = startRankId; index < endRankId; index++) {
        GM_ADDR wAddr =
            (__gm__ uint8_t *)(windowInGM_) + index * dataSizePerRank_ + DATA_OFFSET;  // * bufferSizePerRank_;
        currRankWindowGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
        uint32_t currRankDataOffset = 0;
        uint32_t currRankStatusOffset = index * STATUS_ENTRY_COUNT;

        for (uint32_t j = 0; j < localMoeExpertNum_; j++) {
            // 将数据从Window拷贝到UB
            uint32_t currTokensCount = statusTensor_(currRankStatusOffset + j);
            uint32_t currTokensOffset = epRecvCountsOutLocal(j * worldSize_ + index) - currTokensCount;
            dynamicScalesLocalIdx = 0;
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t k = 0; k < currTokensCount; k++) {
                TEventID eventId = (k & 1) ? EVENT_ID0 : EVENT_ID1;
                xTmpTensor_ = (eventId & 1) ? xOutPingTensor : xOutPongTensor;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopy(xTmpTensor_, currRankWindowGlobal[(currRankDataOffset + k) * axisHCommu_], axisHCommu_);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                if constexpr (DynamicQuant) {
                    PipeBarrier<PIPE_ALL>();
                    xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                    dynamicScalesTensor.SetValue(dynamicScalesLocalIdx++,
                                                 xOutFp32Tensor_.GetValue(axisH_ / sizeof(float)));  // int8->float32
                    PipeBarrier<PIPE_ALL>();
                }
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                DataCopy(expandXOutGMTensor_[(currTokensOffset + k) * axisH_], xTmpTensor_, axisH_);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            currRankDataOffset += currTokensCount;
            PipeBarrier<PIPE_ALL>();
            if constexpr (DynamicQuant) {
                DataCopyExtParams scalesCopyParams{1U, static_cast<uint32_t>(dynamicScalesLocalIdx * sizeof(float)), 0U,
                                                   0U, 0U};
                DataCopyPad(dynamicScalesOutGMTensor_[currTokensOffset], dynamicScalesTensor, scalesCopyParams);
            }
        }
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchV2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        IndexSort();
        ReorderTokens();
        SendToMoeExpert();
        WaitDispatch();
        LocalWindowCopy();
        SyncAll<true>();
        if (aivId_ == 0) {
            Duplicate<int32_t>(statusTensor_, 0, worldSize_ * STATUS_ENTRY_COUNT);  // 8 = UB_ALIGN / sizeof(int32_t)
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            uint32_t dstStrideU32 = dataSizePerRank_ - STATUS_SIZE;
            DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), STATUS_SIZE, 0, dstStrideU32, 0};
            DataCopyPad(windowInstatusTensor_, statusTensor_, copyStatusParams);
        }
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_V2_H
