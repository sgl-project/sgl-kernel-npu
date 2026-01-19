#ifndef SHMEM_MOE_DISPATCH_NORMAL_H
#define SHMEM_MOE_DISPATCH_NORMAL_H

#include "shmem_api.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem_moe_dispatch_normal_tiling.h"
#include "shmem_comm_args.h"

namespace ShmemMoeDispatchNormalImpl {
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t UB_ALIGN = 32U;

constexpr uint64_t DISPATCH_STATUS_OFFSET = 20UL * 1024UL;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define CamTypeClass \
    typename XType, typename ExpandXOutType, bool DynamicQuant, bool IsSmoothScaleExist, bool IsShareExpertRank

#define CamTypeFunc XType, ExpandXOutType, DynamicQuant, IsSmoothScaleExist, IsShareExpertRank

using namespace AscendC;
using namespace ShmemMoe;
template <CamTypeClass>
class ShmemMoeDispatchNormal
{
public:
    __aicore__ inline ShmemMoeDispatchNormal(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_tokenIdx, GM_ADDR put_offset,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                GM_ADDR waitRecvCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const ShmemMoeDispatchNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx);
    __aicore__ inline void QuantInit();
    __aicore__ inline void ResetMetaState();
    __aicore__ inline void PutShareAddr();
    __aicore__ inline void SetSyncFlag(int metaType);
    __aicore__ inline void WaitSyncFlag(int metaType);
    __aicore__ inline void GetShareAddr();
    __aicore__ inline void InputToDstOutput();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess();

    __aicore__ inline GM_ADDR GetMetaAddrByRankId(const int32_t rankId, const int metaType)
    {
        auto ptr = shmem_ptr(gva_gm, rankId);

        switch (metaType) {
            case STATE:  // 存放通信结束的state
                return (GM_ADDR)(ptr);
            case ADDR:  // 存放交换的共享地址
                return (GM_ADDR)(ptr) + DISPATCH_STATUS_OFFSET;
            case FLAG:  // 存放第一次清理state空间后的同步flag
                return (GM_ADDR)(ptr) + META_FLAG_OFFSET;
            default:
                return (GM_ADDR)(ptr);
        }
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGT;
    GlobalTensor<int32_t> expertIdsGT;
    GlobalTensor<int32_t> putOffsetGT;
    GlobalTensor<int32_t> sendTokenIdxGT;
    GlobalTensor<float> dynamicScalesOutGT;

    GlobalTensor<ExpandXOutType> dstGT;
    GlobalTensor<float> dstScaleOutGT;

    GlobalTensor<int32_t> dstStatusGT;
    GlobalTensor<int32_t> waitRecvCostStatsGT;

    LocalTensor<XType> xInTensor;
    LocalTensor<ExpandXOutType> xOutTensor;
    LocalTensor<ExpandXOutType> xTmpTensor;
    LocalTensor<int32_t> expertIdsTensor;
    LocalTensor<int32_t> putOffsetTensor;  // 全局recv_count前缀和
    LocalTensor<int32_t> sendTokenIdxTensor;
    LocalTensor<int32_t> statusTensor;

    TBuf<> expertIdsBuf;
    TBuf<> putOffsetBuf;
    TBuf<> sendTokenIdxBuf;
    TBuf<> addrBuf;
    TBuf<> statusBuf;
    TBuf<> waitStatusBuf;
    TBuf<> gatherMaskOutBuf;
    TBuf<> statusSumBuf;
    TBuf<> scalarBuf;
    TBuf<> tokenCastFloatBuf;
    TBuf<> tokenAbsFloatBuf;

    GM_ADDR expandXOut_;
    GM_ADDR dynamicScalesOut_;

    uint32_t batchSize{0};
    uint32_t globalBatchSize{0};
    uint32_t h{0};
    uint32_t topK{0};
    uint32_t blockNum{0};
    uint32_t blockIdx{0};
    uint32_t epRankSize{0};
    uint32_t epRankId{0};
    uint32_t tpRankSize{0};
    uint32_t tpRankId{0};
    uint32_t moeExpertNum{0};
    uint32_t moeExpertNumPerRank{0};
    bool isEnableDiagnose{false};

    uint32_t rankNumPerBlock;
    uint32_t curBlockStartRankId;
    uint32_t curBlockEndRankId;

    uint32_t hUBAlignSize{0};
    uint32_t hOutGMAlignSize{0};
    uint32_t hOutUBAlignSize{0};
    uint32_t hGMAlignCnt{0};
    uint32_t putOffsetAlignSize{0};
    uint32_t expandIdxStartIdx{0};
    uint32_t expertIdsCnt{0};
    uint32_t stateOffset{0};
    uint32_t dataState{0};
    uint32_t winDataSizeOffset{0};
    uint32_t waitRecvCostStatsBufSize{0};
    uint32_t addrUint64AlignLen_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue;
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> xOutQueue;
    TQue<QuePosition::VECOUT, 1> waitRecvCostStatsOutQueue;

    GM_ADDR gva_gm;
    uint64_t shareExpandOutAddrs[CAM_MAX_RANK_SIZE];  // List of shmem asymmetric output addresses (expandXOut_)
    uint64_t
        shareDynamicScaleAddrs[CAM_MAX_RANK_SIZE];  // List of shmem asymmetric output addresses (dynamicScalesOut_)
    uint32_t shareAddrNum{2};
};

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_tokenIdx,
                                                                 GM_ADDR put_offset, GM_ADDR expandXOut,
                                                                 GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                                                 GM_ADDR waitRecvCostStatsOut, GM_ADDR workspaceGM,
                                                                 TPipe *pipe,
                                                                 const ShmemMoeDispatchNormalTilingData *tilingData)
{
    tpipe_ = pipe;
    blockIdx = GetBlockIdx();

    gva_gm = (GM_ADDR)(tilingData->shmemPtr);

    batchSize = tilingData->moeDispatchNormalInfo.bs;
    globalBatchSize = tilingData->moeDispatchNormalInfo.globalBs;
    h = tilingData->moeDispatchNormalInfo.h;
    topK = tilingData->moeDispatchNormalInfo.k;
    blockNum = tilingData->moeDispatchNormalInfo.aivNum;
    epRankSize = tilingData->moeDispatchNormalInfo.epWorldSize;
    epRankId = tilingData->moeDispatchNormalInfo.epRankId;
    moeExpertNum = tilingData->moeDispatchNormalInfo.moeExpertNum;
    moeExpertNumPerRank = moeExpertNum / epRankSize;
    isEnableDiagnose = tilingData->moeDispatchNormalInfo.isEnableDiagnose;

    xGT.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGT.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    putOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(put_offset));
    sendTokenIdxGT.SetGlobalBuffer((__gm__ int32_t *)(send_tokenIdx));
    dynamicScalesOutGT.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    if (isEnableDiagnose) {
        waitRecvCostStatsGT.SetGlobalBuffer((__gm__ int32_t *)waitRecvCostStatsOut);
    }
    expandXOut_ = expandXOut;
    dynamicScalesOut_ = dynamicScalesOut;
    expertIdsCnt = batchSize * topK;

    hUBAlignSize = Ceil(h * sizeof(ExpandXOutType), UB_ALIGN) * UB_ALIGN;
    uint32_t hScaleSizeAlign = hUBAlignSize + UB_ALIGN;

    hOutUBAlignSize = Ceil(hScaleSizeAlign, UB_ALIGN) * UB_ALIGN;  // h_align_32b + scale(32b)
    if constexpr (DynamicQuant) {
        QuantInit();
    } else {
        tpipe_->InitBuffer(xQueue, BUFFER_NUM, hOutUBAlignSize);  // 2 * 14K = 28K
    }

    putOffsetAlignSize = Ceil(epRankSize * moeExpertNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;  // 4 * ranks * moeNum
    tpipe_->InitBuffer(putOffsetBuf, putOffsetAlignSize);
    putOffsetTensor = putOffsetBuf.Get<int32_t>();

    addrUint64AlignLen_ = Ceil(shareAddrNum * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(addrBuf, addrUint64AlignLen_);

    // rank分核
    SplitCoreCal(epRankSize, rankNumPerBlock, curBlockStartRankId, curBlockEndRankId);
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::QuantInit()
{
    uint32_t hAlignSize = Ceil(h * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xInQueue, BUFFER_NUM, hAlignSize);        // 14K * 2
    tpipe_->InitBuffer(xOutQueue, BUFFER_NUM, hOutUBAlignSize);  // 7K * 2

    tpipe_->InitBuffer(tokenCastFloatBuf, h * sizeof(float));  // 28K
    tpipe_->InitBuffer(tokenAbsFloatBuf, h * sizeof(float));   // 28K
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum,
                                                                         uint32_t &startIdx, uint32_t &endIdx)
{
    perCoreNum = totalNum / blockNum;
    uint32_t remainderRankNum = totalNum % blockNum;

    startIdx = perCoreNum * blockIdx;
    if (blockIdx < remainderRankNum) {
        perCoreNum++;
        startIdx += blockIdx;
    } else {
        startIdx += remainderRankNum;
    }
    endIdx = startIdx + perCoreNum;
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::ResetMetaState()
{
    if (rankNumPerBlock == 0U) {
        return;
    }

    uint32_t waitStatusBufSize = (((rankNumPerBlock * UB_ALIGN) > 256) ? (rankNumPerBlock * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf, waitStatusBufSize);  // ranks/48 * 32B = 1 * 32B

    GlobalTensor<float> statusFp32TensorGT;
    auto ptr = GetMetaAddrByRankId(epRankId, STATE);
    statusFp32TensorGT.SetGlobalBuffer((__gm__ float *)(ptr));

    DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusFp32TensorGT[curBlockStartRankId * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::PutShareAddr()
{
    // 一个核将地址写入本rank的meta
    if (blockIdx != 0) {
        return;
    }

    LocalTensor<uint64_t> addrTensor_ = addrBuf.Get<uint64_t>();
    uint64_t expandXOutAddr = reinterpret_cast<__gm__ uint64_t>(expandXOut_);
    uint64_t dynamicScalesOutAddr = reinterpret_cast<__gm__ uint64_t>(dynamicScalesOut_);
    addrTensor_(0) = expandXOutAddr;
    addrTensor_(1) = dynamicScalesOutAddr;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

    AscendC::GlobalTensor<uint64_t> metaDataGt;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(shareAddrNum * sizeof(uint64_t)), 0, 0, 0};
    GM_ADDR remote_meta = GetMetaAddrByRankId(epRankId, ADDR);
    metaDataGt.SetGlobalBuffer((__gm__ uint64_t *)(remote_meta));
    DataCopyPad(metaDataGt, addrTensor_, copyParams);
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::GetShareAddr()
{
    LocalTensor<uint64_t> addrTensor_ = addrBuf.Get<uint64_t>();
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(addrUint64AlignLen_), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> copyExtParams{false, 0U, 0U, 0U};

    // 从远端获取共享地址
    for (uint32_t i = 0; i < epRankSize; i++) {
        GM_ADDR remote_meta = GetMetaAddrByRankId(i, ADDR);
        AscendC::GlobalTensor<uint64_t> shareAddrGt;
        shareAddrGt.SetGlobalBuffer((__gm__ uint64_t *)(remote_meta));

        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopyPad(addrTensor_, shareAddrGt, copyParams, copyExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        shareExpandOutAddrs[i] = addrTensor_(0);
        shareDynamicScaleAddrs[i] = addrTensor_(1);
    }
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::SetSyncFlag(int metaType)
{
    if (rankNumPerBlock == 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t statusCntAlign = Ceil(rankNumPerBlock, 8) * 8;
    tpipe_->InitBuffer(statusBuf, statusCntAlign * UB_ALIGN);
    LocalTensor statusTensor = statusBuf.Get<int32_t>();
    Duplicate<int32_t>(statusTensor, 0, rankNumPerBlock * 8);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor, 0x3F800000, mask, statusCntAlign / 8, 1, 8);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    AscendC::GlobalTensor<int32_t> gmRemoteStatusGt;
    for (uint32_t i = curBlockStartRankId; i < curBlockEndRankId; i++) {
        auto ptr = GetMetaAddrByRankId(i, metaType) + epRankId * STATE_OFFSET;
        gmRemoteStatusGt.SetGlobalBuffer((__gm__ int32_t *)(ptr));
        DataCopy<int32_t>(gmRemoteStatusGt, statusTensor[(i - curBlockStartRankId) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::WaitSyncFlag(int metaType)
{
    if (rankNumPerBlock == 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t waitStatusBufSize = (((rankNumPerBlock * UB_ALIGN) > 256) ? (rankNumPerBlock * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf, waitStatusBufSize);  // ranks/48 * 32B = 1 * 32B
    uint32_t maskAlign = Ceil(epRankSize * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf, maskAlign);  // rankSize * 4B
    tpipe_->InitBuffer(statusSumBuf, UB_ALIGN);       // 32B

    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf.Get<float>();
    LocalTensor<float> statusSumOutTensor = statusSumBuf.Get<float>(UB_ALIGN);
    LocalTensor<float> statusFp32Tensor = waitStatusBuf.Get<float>();
    GlobalTensor<float> statusFp32TensorGT;
    auto ptr = GetMetaAddrByRankId(epRankId, metaType);
    statusFp32TensorGT.SetGlobalBuffer((__gm__ float *)(ptr));
    uint32_t mask = 1;
    float compareTarget = static_cast<float>(1.0) * rankNumPerBlock;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(rankNumPerBlock), 1, 0, 0};

    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor, statusFp32TensorGT[curBlockStartRankId * STATE_OFFSET / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor, gatherMaskOutTensor, mask, rankNumPerBlock, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    // 清标记位
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusFp32TensorGT[curBlockStartRankId * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>();
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::ReduceMaxInplace(const LocalTensor<float> &srcLocal,
                                                                             uint32_t count)
{
    uint64_t repsFp32 = count >> 6;        // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6;  // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;      // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL;  // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::QuantProcess()
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = tokenCastFloatBuf.Get<float>();

    Cast(floatLocalTemp, xInTensor, RoundMode::CAST_NONE, h);
    xInQueue.FreeTensor<XType>(xInTensor);
    PipeBarrier<PIPE_V>();

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = tokenAbsFloatBuf.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, h);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, h);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / (floatLocalAbsTemp.GetValue(0) + 1e-12f);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, h);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, h);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, h);

    PipeBarrier<PIPE_V>();
    Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, h);

    floatLocalTemp = xOutTensor.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hUBAlignSize / sizeof(float), float(1.0) / dynamicScale);  // int8->float32
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::InputToDstOutput()
{
    uint32_t startTokenId, endTokenId, sendTokenNum, remainTokenNum;
    sendTokenNum = expertIdsCnt / blockNum;
    remainTokenNum = expertIdsCnt % blockNum;
    startTokenId = sendTokenNum * blockIdx;
    if (blockIdx < remainTokenNum) {
        sendTokenNum += 1;
        startTokenId += blockIdx;
    } else {
        startTokenId += remainTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;

    if (startTokenId >= expertIdsCnt) {
        return;  // 按照bs*k的token数进行分核
    }

    DataCopyExtParams putOffsetParams = {1U, static_cast<uint32_t>(epRankSize * moeExpertNum * sizeof(uint32_t)), 0U,
                                         0U, 0U};
    DataCopyPadExtParams<int32_t> putOffsetCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(putOffsetTensor, putOffsetGT, putOffsetParams, putOffsetCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    tpipe_->InitBuffer(expertIdsBuf, sendTokenNum * sizeof(int32_t));     // 4 * bs * k / 48
    tpipe_->InitBuffer(sendTokenIdxBuf, sendTokenNum * sizeof(int32_t));  // 4 * bs * k / 48
    expertIdsTensor = expertIdsBuf.Get<int32_t>();
    sendTokenIdxTensor = sendTokenIdxBuf.Get<int32_t>();
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyExtParams sendTokenIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor, expertIdsGT[startTokenId], expertIdsCntParams, copyPadExtParams);
    DataCopyPad(sendTokenIdxTensor, sendTokenIdxGT[startTokenId], sendTokenIdxParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(h * sizeof(XType)), 0U, 0U, 0U};
    DataCopyPadExtParams<XType> tokenCopyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams xOutCopyParams = {1U, static_cast<uint32_t>(hUBAlignSize), 0U, 0U, 0U};  // 只拷贝hidden_size
    DataCopyExtParams scaleCopyParams = {1U, sizeof(float), 0U, 0U, 0U};                       // 拷贝dynamicScales

    for (int32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor(tokenIndex - startTokenId);
        if (dstExpertId < 0 || dstExpertId >= moeExpertNum) {
            continue;
        }
        uint32_t dstRankId = dstExpertId / moeExpertNumPerRank;
        // 对端output的小偏移，专家内不同rank来源内的，本卡发送给该专家的token序号
        int32_t curExpertIdx = sendTokenIdxTensor(tokenIndex - startTokenId);
        // 对端output的大偏移，不同专家及不同rank来源间的，本卡需要放置给该rank的token大偏移，定位到专家和来源rank
        int32_t dstExpertOffset = putOffsetTensor(dstExpertId * epRankSize + epRankId);

        auto ptr = shareExpandOutAddrs[dstRankId];
        dstGT.SetGlobalBuffer((__gm__ ExpandXOutType *)(ptr + hUBAlignSize * (dstExpertOffset + curExpertIdx)));

        if constexpr (DynamicQuant) {
            auto dsPtr = shareDynamicScaleAddrs[dstRankId];
            dstScaleOutGT.SetGlobalBuffer((__gm__ float *)(dsPtr) + (dstExpertOffset + curExpertIdx));

            xInTensor = xInQueue.AllocTensor<XType>();
            DataCopyPad(xInTensor, xGT[tokenIndex / topK * h], xCopyParams, tokenCopyPadExtParams);
            xInQueue.EnQue(xInTensor);
            xInTensor = xInQueue.DeQue<XType>();
            xOutTensor = xOutQueue.AllocTensor<ExpandXOutType>();
            QuantProcess();
            xOutQueue.EnQue(xOutTensor);
            xOutTensor = xOutQueue.DeQue<ExpandXOutType>();
            DataCopyPad(dstGT, xOutTensor, xOutCopyParams);  // 拷贝token

            LocalTensor<float> xOutFp32Tensor = xOutTensor.template ReinterpretCast<float>();
            DataCopyPad(dstScaleOutGT, xOutFp32Tensor[hUBAlignSize / sizeof(float)], scaleCopyParams);

            xOutQueue.FreeTensor(xOutTensor);
        } else {
            xTmpTensor = xQueue.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor, xGT[tokenIndex / topK * h], xCopyParams, tokenCopyPadExtParams);
            xQueue.EnQue(xTmpTensor);
            xTmpTensor = xQueue.DeQue<ExpandXOutType>();
            DataCopyPad(dstGT, xTmpTensor, xOutCopyParams);
            xQueue.FreeTensor<ExpandXOutType>(xTmpTensor);
        }
    }
}

template <CamTypeClass>
__aicore__ inline void ShmemMoeDispatchNormal<CamTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        ResetMetaState();
        PutShareAddr();
        SetSyncFlag(FLAG);  // 全卡同步，确保对称地址都放到了meta空间
        WaitSyncFlag(FLAG);

        GetShareAddr();
        InputToDstOutput();
        SetSyncFlag(STATE);  // 全卡同步，确保数据已经获取完
        WaitSyncFlag(STATE);
    }
}

}  // namespace ShmemMoeDispatchNormalImpl
#endif
