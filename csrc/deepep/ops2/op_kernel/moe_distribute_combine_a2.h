/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_a2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_H
#define MOE_DISTRIBUTE_COMBINE_A2_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_a2_tiling.h"
#include "moe_distribute_base.h"
namespace {
constexpr uint8_t BUFFER_NUM = 2;                       // 多buf
constexpr uint32_t STATE_OFFSET = 512;                  // 状态空间偏移地址
constexpr uint32_t STATE_SPACE_SIZE = 1024 * 1024;      // 1M
constexpr uint32_t UB_ALIGN = 32;                       // UB按32字节对齐
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;      // 本卡状态空间偏移地址
constexpr uint32_t BATCH_WRITE_ITEM_OFFSET = 8 * 1024;  // batchWriteInfo结构体地址相对于windowOut最后1M的偏移
constexpr uint32_t BATCH_WRITE_ITEM_SIZE = 32;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t B32_PER_BLOCK = 8;
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint32_t SKIP_OFFSET = 32;
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint64_t MB_SIZE = 1024 * 1024;
template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

struct TaskInfo {
    uint32_t startTaskId;
    uint32_t endTaskId;
    uint32_t taskNum;

    __aicore__ inline TaskInfo() {}
    __aicore__ inline void SplitCore(uint32_t taskNumTotal, uint32_t aivNum, uint32_t aivId)
    {
        if (aivNum == 0) {
            startTaskId = 0;
            endTaskId = 0;
            taskNum = 0;
            return;
        }

        uint32_t formerNum = taskNumTotal / aivNum;
        uint32_t tailNum = taskNumTotal % aivNum;
        startTaskId = formerNum * aivId;
        if (aivId < tailNum) {
            formerNum++;
            startTaskId += aivId;
        } else {
            startTaskId += tailNum;
        }
        taskNum = formerNum;
        endTaskId = startTaskId + taskNum;
    }
};

}  // namespace
namespace MoeDistributeCombineA2Impl {
#define TemplateMC2TypeA2Class typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA2Func ExpandXType, ExpandIdxType
using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeCombineA2
{
public:
    __aicore__ inline MoeDistributeCombineA2(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineA2TilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void Preload();
    __aicore__ inline void WaitDispatch();
    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_;  // 用于存对端window的变量
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> windowInstatusTensor_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // win区状态位置拷入相关参数
    GlobalTensor<uint64_t> workspaceGlobal_;    // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> workspaceGlobal32_;  // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> flagGlobal_;
    LocalTensor<uint64_t> batchWriteItemLocalB64;
    LocalTensor<uint32_t> batchWriteItemLocalB32;
    LocalTensor<uint32_t> recvCountLocal_;
    LocalTensor<uint32_t> expertWindowOffsetLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatLocal_;
    LocalTensor<ExpandIdxType> expertIdsLocal_;
    LocalTensor<float> expandScalesLocal_;
    LocalTensor<ExpandIdxType> indexCountsLocal_;
    LocalTensor<ExpandXType> tmpUb_;
    LocalTensor<uint32_t> statusTensor_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR XOutGM_;
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    uint32_t moeExpertNum_{0};         // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};    // 每张卡的专家数
    uint32_t expandXRows_;
    uint64_t rankSizeOnWin_{0};
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t bsKAlign_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    uint32_t tokenIndex_{0};
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moeQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> moeSumQueue_;
    TBuf<> expertIdsBuf_;
    TBuf<> expandScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> sendCountBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> tokenBuf_;
    TBuf<> statusBuf_;
    TBuf<> batchWriteItemBuf_;
    TBuf<> recvCountBuf_;
    TBuf<> expertWindowOffsetBuf_;

    TaskInfo taskInfo_;

    GlobalTensor<uint32_t> expertRecvCountGlobal_;
    GlobalTensor<uint32_t> expertWindowOffsetGlobal_;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR XOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData, __gm__ void *mc2InitTiling,
    __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    expandXGM_ = expandX;
    expertIdsGM_ = expertIds;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    XOutGM_ = XOut;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;
    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)contextGM;
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);
    coreIdx_ = GetBlockIdx();
    PRINTF("[Init] combine_a2, coreId:%d \n", coreIdx_);

    /*
    halfWinSize_ = winContext_->winSize / 2;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_));
    bufferId_ = bufferIdGlobal_.GetValue(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;

    windowInstatusTensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_));
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));
    workspaceGlobal32_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));

    expertRecvCountGlobal_.SetGlobalBuffer((__gm__ uint32_t *)workspaceGM);
    expertWindowOffsetGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(workspaceGM + moeExpertNum_ * sizeof(uint32_t)));

    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    expandXRows_ = localMoeExpertNum_ * axisBS_ * worldSize_;
    rankSizeOnWin_ = dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = dataSpaceSize_ + rankId_ * STATE_OFFSET;
    axisHFloatSize_ = axisH_ * sizeof(float);
    axisHExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    bsKAlign_ = RoundUp(axisBS_ * axisK_, (uint32_t)8);

    uint64_t stateSizeMaxSize = 2 * STATE_SPACE_SIZE; // 2: 实际上是(DATA_OFFSET+SKIP_OFFSET+sizeof(uint32)) +
    STATE_SPACE_SIZE，近似计算使用2 * STATE_SPACE_SIZE uint64_t winSizeMin = (axisBS_ * worldSize_ * (localMoeExpertNum_
    > axisK_ ? axisK_ : localMoeExpertNum_) * axisH_ * sizeof(uint16_t) + stateSizeMaxSize) * BUFFER_NUM; //
    考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
    assert(winContext_->winSize >= winSizeMin, "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
        epWorldSize:%u, epRankId:%u, moeExpertNum:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
        totalUbSize:%llu\n",
        winContext_->winSize / MB_SIZE, winSizeMin / MB_SIZE,
        tilingData->moeDistributeCombineInfo.epWorldSize, tilingData->moeDistributeCombineInfo.epRankId,
    tilingData->moeDistributeCombineInfo.moeExpertNum, tilingData->moeDistributeCombineInfo.globalBs,
    tilingData->moeDistributeCombineInfo.bs, tilingData->moeDistributeCombineInfo.k,
        tilingData->moeDistributeCombineInfo.h, tilingData->moeDistributeCombineInfo.aivNum,
    tilingData->moeDistributeCombineInfo.totalUbSize
    );

    BuffInit();
    SplitCoreCal();
    */
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::BuffInit()
{
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_);  // 7168 * 2 * 2 = 28672
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(expertIdsBuf_, axisBS_ * axisK_ * sizeof(int32_t));   // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(expandScalesBuf_, axisBS_ * axisK_ * sizeof(float));  // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(tokenBuf_, axisHExpandXTypeSize_);                    // 7168 * 2 = 14336
    tpipe_->InitBuffer(rowTmpFloatBuf_, axisHFloatSize_);                    // 7168 * 4 = 28672
    tpipe_->InitBuffer(mulBuf_, axisHFloatSize_);                            // 7168 * 4 = 28672
    tpipe_->InitBuffer(sumFloatBuf_, axisHFloatSize_);                       // 7168 * 4 = 28672
    tpipe_->InitBuffer(sendCountBuf_, RoundUp(moeExpertNum_, B32_PER_BLOCK) * sizeof(int32_t));
    tpipe_->InitBuffer(indexCountsBuf_, axisBS_ * axisK_ * sizeof(int32_t));  // 32 * 8 * 4 = 1024
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
    tpipe_->InitBuffer(batchWriteItemBuf_, BATCH_WRITE_ITEM_SIZE * worldSize_);
    batchWriteItemLocalB64 = batchWriteItemBuf_.Get<uint64_t>();
    batchWriteItemLocalB32 = batchWriteItemLocalB64.template ReinterpretCast<uint32_t>();
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    sendRankNum_ = worldSize_ / aivNum_;
    uint32_t remainderRankNum = worldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AlltoAllDispatch()
{
    if (sendRankNum_ == 0) {
        SyncAll<true>();
        return;
    }
    LocalTensor<ExpandIdxType> sendCountLocal = sendCountBuf_.Get<int32_t>();
    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_));
        uint32_t rankTokenNum = 0;
        for (uint32_t expertId = 0; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0;
            if (expertId != 0 || dstRankId != 0) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            uint32_t startTokenAddr = preCount * axisH_;
            uint32_t tokenNum = sendCountLocal(expertId * worldSize_ + dstRankId) - preCount;
            for (uint32_t tokenId = 0; tokenId < tokenNum; ++tokenId) {
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                DataCopy(InUb, expandXGlobal_[startTokenAddr], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(localOutWindow_[rankTokenNum * axisH_], OutUb, axisH_);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
                startTokenAddr += axisH_;
                rankTokenNum++;
            }
        }
        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)(localOutWindow_.GetPhyAddr(rankTokenNum * axisH_) + SKIP_OFFSET / sizeof(ExpandXType)));
        flagGlobal_(0) = FLAG_VALUE;
        uint32_t rankIdOffset = dstRankId - startRankId_;
        batchWriteItemLocalB64(rankIdOffset * 4) = (uint64_t)(localOutWindow_.GetPhyAddr());
        batchWriteItemLocalB64(rankIdOffset * 4 + 1) =
            (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + dataOffsetOnWin_);
        batchWriteItemLocalB64(rankIdOffset * 4 + 2) = rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType) + 2;
        batchWriteItemLocalB32(rankIdOffset * 8 + 6) = HcclDataType::HCCL_DATA_TYPE_FP16;
        batchWriteItemLocalB32(rankIdOffset * 8 + 7) = dstRankId;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            flagGlobal_);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(workspaceGlobal_[startRankId_ * 4], batchWriteItemLocalB64, sendRankNum_ * 4);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();
    if ASCEND_IS_AIV {
        if (coreIdx_ == 0) {
            HcclHandle handleId = hccl_.BatchWrite<true>((GM_ADDR)(workspaceGlobal_.GetPhyAddr()), worldSize_);
            bufferIdGlobal_(0) = bufferId_ ^ 1;
        }
        if (rankId_ >= startRankId_ && rankId_ < endRankId_) {
            localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dataOffsetOnWin_));
            localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowInGM_ + dataOffsetOnWin_));
            uint32_t rankIdOffset = rankId_ - startRankId_;
            uint64_t rankTokenNum =
                (batchWriteItemLocalB64(rankIdOffset * 4 + 2) - SKIP_OFFSET / sizeof(ExpandXType) - 2) / axisH_;
            for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                DataCopy(InUb, localOutWindow_[tokenId * axisH_], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(localInWindow_[tokenId * axisH_], OutUb, axisH_);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
            }
            flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)localInWindow_.GetPhyAddr(
                rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
            flagGlobal_(0) = FLAG_VALUE;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                flagGlobal_);
        }
    }
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Preload()
{
    tpipe_->InitBuffer(recvCountBuf_, sizeof(uint32_t) * moeExpertNum_);
    tpipe_->InitBuffer(expertWindowOffsetBuf_, sizeof(uint32_t) * moeExpertNum_);
    recvCountLocal_ = recvCountBuf_.Get<uint32_t>();
    expertWindowOffsetLocal_ = expertWindowOffsetBuf_.Get<uint32_t>();
    expertIdsLocal_ = expertIdsBuf_.Get<ExpandIdxType>();
    DataCopy(expertIdsLocal_, expertIdsGlobal_, bsKAlign_);
    Duplicate(recvCountLocal_, (uint32_t)0, moeExpertNum_);
    Duplicate(expertWindowOffsetLocal_, (uint32_t)0, moeExpertNum_);

    SyncFunc<AscendC::HardEvent::V_MTE3>();

    if (coreIdx_ == aivNum_ - 1) {
        DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
                    {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    }

    SyncAll<true>();

    taskInfo_.SplitCore(axisBS_ * axisK_, aivNum_, coreIdx_);
    for (uint32_t i = taskInfo_.startTaskId; i < taskInfo_.endTaskId; ++i) {
        uint32_t expId = expertIdsLocal_.GetValue(i);
        recvCountLocal_(expId) += 1;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    SetAtomicAdd<int32_t>();
    DataCopyPad(expertRecvCountGlobal_, recvCountLocal_,
                {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
    SetAtomicNone();

    SyncAll<true>();

    DataCopyPad(recvCountLocal_, expertRecvCountGlobal_,
                {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    taskInfo_.SplitCore(moeExpertNum_ / localMoeExpertNum_, aivNum_, coreIdx_);
    for (uint32_t groupIdx = taskInfo_.startTaskId; groupIdx < taskInfo_.endTaskId; ++groupIdx) {
        uint32_t start = groupIdx * localMoeExpertNum_;
        uint32_t end = start + localMoeExpertNum_;
        uint32_t prefixSum = 0;
        for (uint32_t i = start; i < end; ++i) {
            expertWindowOffsetLocal_(i - start) = prefixSum;
            prefixSum += recvCountLocal_.GetValue(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertWindowOffsetGlobal_[start], expertWindowOffsetLocal_,
                    {1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(uint32_t)), 0, 0, 0});
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();

    DataCopyPad(expertWindowOffsetLocal_, expertWindowOffsetGlobal_,
                {1, static_cast<uint32_t>(moeExpertNum_ * sizeof(uint32_t)), 0, 0, 0}, {false, 0, 0, 0});

    tokenNumPerCore_ = axisBS_ / aivNum_;
    uint32_t undoTokenNum = axisBS_ % aivNum_;
    tokenIndex_ = 0;
    if (coreIdx_ < undoTokenNum) {
        tokenNumPerCore_ = tokenNumPerCore_ + 1;
        tokenIndex_ = coreIdx_ * tokenNumPerCore_;
    } else {
        tokenIndex_ = (undoTokenNum + coreIdx_ * tokenNumPerCore_);
    }
    if (tokenNumPerCore_ == 0) {
        return;
    }
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    mulBufLocal_ = mulBuf_.Get<float>();
    sumFloatLocal_ = sumFloatBuf_.Get<float>();
    expandScalesLocal_ = expandScalesBuf_.Get<float>();
    indexCountsLocal_ = indexCountsBuf_.Get<ExpandIdxType>();
    DataCopy(indexCountsLocal_, expandIdxGlobal_, bsKAlign_);
    DataCopy(expandScalesLocal_, expandScalesGlobal_, bsKAlign_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    if (startRankId_ >= worldSize_) {
        SyncAll<true>();
        return;
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t waitFlagNum = 0; waitFlagNum < sendRankNum_;) {
        waitFlagNum = 0;
        for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
            uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
            GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
                            (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
            flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                flagGlobal_);
            uint32_t flag = flagGlobal_(0);
            if (flag == FLAG_VALUE) {
                waitFlagNum++;
            }
        }
    }
    for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
        uint32_t tokenIdx = (rankId + 1) * localMoeExpertNum_ - 1;
        GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET +
                        (recvCountLocal_(tokenIdx) + expertWindowOffsetLocal_(tokenIdx)) * axisHExpandXTypeSize_;
        flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
        flagGlobal_(0) = 0;
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()
{
    PRINTF("[Process] combine_a2, coreId:%d \n", coreIdx_);
    SyncAll<true>();
    hccl_.Finalize();

    /*
    if ASCEND_IS_AIV {
        AlltoAllDispatch();
        Preload();
        WaitDispatch();
        LocalWindowCopy();
        hccl_.Finalize();
    }
    */
}
template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    if (tokenNumPerCore_ == 0) {
        return;
    }
    // step 4 & step 5
    GM_ADDR wAddr;
    int32_t expId = 0;
    float scaleVal = 0.0;
    for (uint32_t i = 0; i < tokenNumPerCore_; i++) {
        uint32_t index = (tokenIndex_ + i) * axisK_;
        Duplicate(sumFloatLocal_, 0.0f, axisH_);
        for (uint32_t j = 0; j < axisK_; j++) {
            expId = expertIdsLocal_.GetValue(index);
            scaleVal = expandScalesLocal_.GetValue(index);
            uint32_t rank = expId / localMoeExpertNum_;
            wAddr = (__gm__ uint8_t *)(windowInGM_) + rankSizeOnWin_ * rank +
                    expertWindowOffsetLocal_.GetValue(expId) * axisHExpandXTypeSize_ +
                    indexCountsLocal_.GetValue(index) * axisHExpandXTypeSize_;
            // copy experts from window
            rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
            tmpUb_ = moeSumQueue_.AllocTensor<ExpandXType>();
            DataCopy(tmpUb_, rankWindow_, axisH_);
            moeSumQueue_.EnQue(tmpUb_);
            LocalTensor<ExpandXType> tmpOtherUb_ = moeSumQueue_.DeQue<ExpandXType>();
            // cast before muls
            Cast(rowTmpFloatLocal_, tmpOtherUb_, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            // muls expert and scaleVal
            AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
            PipeBarrier<PIPE_V>();
            // add mulBufLocal to sumFloatBufLocal
            AscendC::Add(sumFloatLocal_, sumFloatLocal_, mulBufLocal_, axisH_);
            index++;
            moeSumQueue_.FreeTensor<ExpandXType>(tmpOtherUb_);
        }
        // 结果搬出
        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXType> sumBufLocal_ = tokenBuf_.Get<ExpandXType>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(sumBufLocal_, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[(tokenIndex_ + i) * axisH_], sumBufLocal_, axisH_);
        PipeBarrier<PIPE_V>();
    }
}
}  // namespace MoeDistributeCombineA2Impl
#endif  // MOE_DISTRIBUTE_COMBINE_A2_H
