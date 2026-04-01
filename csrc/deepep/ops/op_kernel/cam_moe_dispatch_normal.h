#ifndef CAM_MOE_DISPATCH_NORMAL_H
#define CAM_MOE_DISPATCH_NORMAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "cam_moe_dispatch_normal_tiling.h"
#include "comm_args.h"

namespace CamMoeDispatchNormalImpl {
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint8_t COMM_NUM = 2;
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;

constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t EXPAND_IDX_INFO = 3U;
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 4UL * 1024UL * 1024UL;
constexpr int64_t CYCLE_TO_TIME = 50;  // cycle num is converted into a fixed base unit of time, set at 50
constexpr uint64_t ROUND_STATE_OFFSET = Moe::BASE_ROUND_STATE_OFFSET;
constexpr uint32_t FLOAT_NUM_PER_ALIGN = 8U;

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
template <CamTypeClass>
class CamMoeDispatchNormal
{
public:
    __aicore__ inline CamMoeDispatchNormal(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset, GM_ADDR send_tokenIdx,
                                GM_ADDR recv_offset, GM_ADDR recv_count, GM_ADDR expert_global_offset,
                                GM_ADDR srcrank_in_expert_offset, GM_ADDR r_in_srcrank_offset, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR waitRecvCostStatsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const CamMoeDispatchNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InputToShare();
    __aicore__ inline void SetStatus();
    __aicore__ inline void SetRoundStatus();
    __aicore__ inline void WaitStatus();
    __aicore__ inline void WaitRoundStatus();
    __aicore__ inline void ShareToOutputLongSeq();
    __aicore__ inline void ShareToOutput();
    __aicore__ inline void UpdateOutput();
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void QuantInit();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankId : tpRankId);
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + winDataSizeOffset + COMBINE_STATE_WIN_OFFSET +
                   Moe::NOTIFY_DISPATCH_BUFF_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset + COMBINE_STATE_WIN_OFFSET + Moe::NOTIFY_DISPATCH_BUFF_OFFSET;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ctxIdx == COMM_EP_IDX ? epRankId : tpRankId;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_[ctxIdx]->localWindowsExp) + dataState * WIN_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))
                             ->windowsExp) +
               dataState * WIN_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetRoundStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ctxIdx == COMM_EP_IDX ? epRankId : tpRankId;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_[ctxIdx]->localWindowsExp) + dataState * Moe::ROUND_STATE_MAX_SIZE +
                   ROUND_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))
                             ->windowsExp) +
               dataState * Moe::ROUND_STATE_MAX_SIZE + ROUND_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGT;
    GlobalTensor<int32_t> expertIdsGT;
    GlobalTensor<int32_t> sendOffsetGT;
    GlobalTensor<int32_t> sendTokenIdxGT;
    GlobalTensor<int32_t> recvOffsetGT;
    GlobalTensor<int32_t> recvCountGT;
    GlobalTensor<int32_t> expertGlobalOffsetGT;
    GlobalTensor<int32_t> srcrankInExpertOffsetGT;
    GlobalTensor<int32_t> rInSrcrankOffsetGT;
    GlobalTensor<float> dynamicScalesOutGT;
    GlobalTensor<int32_t> expandIdxOutGT;
    GlobalTensor<ExpandXOutType> dstGT;
    GlobalTensor<int32_t> dstStatusGT;
    GlobalTensor<int32_t> waitRecvCostStatsGT;
    GlobalTensor<float> dstRoundStatusGT;
    LocalTensor<XType> xInTensor;
    LocalTensor<ExpandXOutType> xOutTensor;
    LocalTensor<ExpandXOutType> xTmpTensor;
    LocalTensor<int32_t> expertIdsTensor;
    LocalTensor<int32_t> sendOffsetTensor;
    LocalTensor<int32_t> sendTokenIdxTensor;
    LocalTensor<int32_t> recvOffsetTensor;
    LocalTensor<int32_t> recvCountTensor;
    LocalTensor<int32_t> statusTensor;
    LocalTensor<int32_t> waitRecvCostStatsTensor;
    LocalTensor<float> recvStatusTensor1;
    LocalTensor<float> recvStatusTensor2;
    LocalTensor<int32_t> expertGlobalOffsetTensor;
    LocalTensor<int32_t> srcrankInExpertOffsetTensor;
    LocalTensor<int32_t> rInSrcrankOffsetTensor;

    TBuf<> expertIdsBuf;
    TBuf<> sendOffsetBuf;
    TBuf<> sendTokenIdxBuf;
    TBuf<> recvOffsetBuf;
    TBuf<> recvCountBuf;
    TBuf<> statusBuf;
    TBuf<> waitStatusBuf;
    TBuf<> gatherMaskOutBuf;
    TBuf<> scalarBuf;
    TBuf<> tokenCastFloatBuf;
    TBuf<> tokenAbsFloatBuf;
    TBuf<> recvStatusBuf;
    TBuf<> roundStatusBuf;
    TBuf<> tempRoundStatusBuf;
    TBuf<> expertGlobalOffsetBuf;
    TBuf<> srcrankInExpertOffsetBuf;
    TBuf<> rInSrcrankOffsetBuf;

    GM_ADDR expandXOutGM;
    GM_ADDR shareGM;

    uint32_t batchSize{0};
    uint32_t realMaxBatchSize{0};
    uint32_t globalBatchSize{0};
    uint32_t round{4};
    uint32_t perRoundTokens{1024};
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

    uint32_t hUBAlignSize{0};
    uint32_t hOutGMAlignSize{0};
    uint32_t hOutUBAlignSize{0};
    uint32_t hGMAlignCnt{0};
    uint32_t expandIdxStartIdx{0};
    uint32_t expertIdsCnt{0};
    uint32_t stateOffset{0};
    uint32_t dataState{0};
    uint32_t winDataSizeOffset{0};
    uint32_t waitRecvCostStatsBufSize{0};
    uint32_t srcRankOffset{0};

    uint32_t startStatusId;
    uint32_t endStatusId;
    uint32_t statusNumPerCore;
    uint32_t remainStatus;
    uint32_t roundIndex;
    uint32_t hScaleIdxSize;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue;
    TQue<QuePosition::VECIN, 1> xInQueue;
    TQue<QuePosition::VECOUT, 1> xOutQueue;
    TQue<QuePosition::VECOUT, 1> waitRecvCostStatsOutQueue;

    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};

    DataCopyExtParams hCommuCopyOutParams;
};

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset, GM_ADDR send_tokenIdx, GM_ADDR recv_offset, GM_ADDR recv_count,
    GM_ADDR expert_global_offset, GM_ADDR srcrank_in_expert_offset, GM_ADDR r_in_srcrank_offset, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR waitRecvCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const CamMoeDispatchNormalTilingData *tilingData)
{
    tpipe_ = pipe;
    blockIdx = GetBlockIdx();

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_[COMM_TP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<1>();

    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET + blockIdx * WIN_ADDR_ALIGN));

    batchSize = tilingData->camMoeDispatchNormalInfo.bs;
    realMaxBatchSize = tilingData->camMoeDispatchNormalInfo.realMaxBs;
    globalBatchSize = tilingData->camMoeDispatchNormalInfo.globalBs;
    round = tilingData->camMoeDispatchNormalInfo.round;
    perRoundTokens = tilingData->camMoeDispatchNormalInfo.perRoundTokens;
    h = tilingData->camMoeDispatchNormalInfo.h;
    topK = tilingData->camMoeDispatchNormalInfo.k;
    blockNum = tilingData->camMoeDispatchNormalInfo.aivNum;
    epRankSize = tilingData->camMoeDispatchNormalInfo.epWorldSize;
    epRankId = tilingData->camMoeDispatchNormalInfo.epRankId;
    moeExpertNum = tilingData->camMoeDispatchNormalInfo.moeExpertNum;
    moeExpertNumPerRank = moeExpertNum / epRankSize;
    isEnableDiagnose = tilingData->camMoeDispatchNormalInfo.isEnableDiagnose;

    xGT.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGT.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    sendOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(send_offset));
    sendTokenIdxGT.SetGlobalBuffer((__gm__ int32_t *)(send_tokenIdx));
    recvOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(recv_offset));
    recvCountGT.SetGlobalBuffer((__gm__ int32_t *)(recv_count));
    expertGlobalOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(expert_global_offset));
    srcrankInExpertOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(srcrank_in_expert_offset));
    rInSrcrankOffsetGT.SetGlobalBuffer((__gm__ int32_t *)(r_in_srcrank_offset));
    dynamicScalesOutGT.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expandIdxOutGT.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    if (isEnableDiagnose) {
        waitRecvCostStatsGT.SetGlobalBuffer((__gm__ int32_t *)waitRecvCostStatsOut);
    }

    expandXOutGM = expandXOut;

    hUBAlignSize = Ceil(h * sizeof(ExpandXOutType), UB_ALIGN) * UB_ALIGN;
    uint32_t hScaleSizeAlign = hUBAlignSize + UB_ALIGN;
    expandIdxStartIdx = hScaleSizeAlign / sizeof(int32_t);

    hScaleIdxSize = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);
    hOutGMAlignSize = Ceil(hScaleIdxSize, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hGMAlignCnt = hOutGMAlignSize / sizeof(ExpandXOutType);

    expertIdsCnt = batchSize * topK;
    roundIndex = 0;
    statusNumPerCore = moeExpertNum / blockNum;
    remainStatus = moeExpertNum % blockNum;
    startStatusId = statusNumPerCore * blockIdx;
    if (blockIdx < remainStatus) {
        statusNumPerCore += 1;
        startStatusId += blockIdx;
    } else {
        startStatusId += remainStatus;
    }
    endStatusId = startStatusId + statusNumPerCore;
    stateOffset = STATE_OFFSET;
    srcRankOffset = startStatusId / moeExpertNumPerRank;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState = selfDataStatusTensor(0);
    if (dataState == 0) {
        selfDataStatusTensor(0) = 1;
    } else {
        selfDataStatusTensor(0) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    PipeBarrier<PIPE_ALL>();

    uint64_t hSizeAlignCombine = Ceil(h * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hSizeAlignCombine = round > 1 ? hSizeAlignCombine * 2 : hSizeAlignCombine;
    winDataSizeOffset = dataState * (tilingData->camMoeDispatchNormalInfo.totalWinSize / 2) +
                        min(realMaxBatchSize, perRoundTokens) * topK * hSizeAlignCombine;  // *2 是因为double buffer
    shareGM = GetWindAddrByRankId(COMM_EP_IDX, epRankId);

    hCommuCopyOutParams = {1U, static_cast<uint32_t>(hScaleIdxSize), 0U, 0U, 0U};
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::QuantInit()
{
    uint32_t hAlignSize = Ceil(h * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xInQueue, BUFFER_NUM, hAlignSize);        // 14K * 2
    tpipe_->InitBuffer(xOutQueue, BUFFER_NUM, hOutUBAlignSize);  // 7K * 2

    tpipe_->InitBuffer(tokenCastFloatBuf, h * sizeof(float));  // 28K
    tpipe_->InitBuffer(tokenAbsFloatBuf, h * sizeof(float));   // 28K
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::ReduceMaxInplace(const LocalTensor<float> &srcLocal,
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
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::QuantProcess()
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
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::FillTriple(LocalTensor<ExpandXOutType> &xOutTensor,
                                                                     uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(expandIdxStartIdx) = epRankId;
    xOutTint32(expandIdxStartIdx + 1) = tokenIndex;
    xOutTint32(expandIdxStartIdx + 2) = k;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::InputToShare()
{
    tpipe_->Reset();
    hOutUBAlignSize = Ceil(hScaleIdxSize, UB_ALIGN) * UB_ALIGN;
    if constexpr (DynamicQuant) {
        QuantInit();
    } else {
        tpipe_->InitBuffer(xQueue, BUFFER_NUM, hOutUBAlignSize);  // 2 * 14K = 28K
    }
    tpipe_->InitBuffer(sendOffsetBuf, moeExpertNum * sizeof(int32_t));  // 4 * moeNum
    sendOffsetTensor = sendOffsetBuf.Get<int32_t>();

    DataCopyExtParams sendOffsetParams = {1U, static_cast<uint32_t>(moeExpertNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> sendOffsetCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(sendOffsetTensor, sendOffsetGT[roundIndex * moeExpertNum], sendOffsetParams, sendOffsetCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint32_t startTokenId, endTokenId, sendTokenNum, remainTokenNum;

    uint32_t realRound = (realMaxBatchSize + perRoundTokens - 1) / perRoundTokens;
    uint32_t localRound = (batchSize + perRoundTokens - 1) / perRoundTokens;

    if (roundIndex >= localRound) {
        expertIdsCnt = 0;
    } else if (roundIndex < localRound - 1) {
        expertIdsCnt = perRoundTokens * topK;
    } else {
        uint32_t processedTokens = perRoundTokens * roundIndex;
        uint32_t remaining = (batchSize > processedTokens) ? (batchSize - processedTokens) : 0;
        expertIdsCnt = remaining * topK;
    }
    if (expertIdsCnt == 0) {
        return;
    }
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

    if (startTokenId >= expertIdsCnt || sendTokenNum == 0) {
        return;
    }
    tpipe_->InitBuffer(expertIdsBuf, sendTokenNum * sizeof(int32_t));     // 4 * bs * k / 48
    tpipe_->InitBuffer(sendTokenIdxBuf, sendTokenNum * sizeof(int32_t));  // 4 * bs * k / 48
    expertIdsTensor = expertIdsBuf.Get<int32_t>();
    sendTokenIdxTensor = sendTokenIdxBuf.Get<int32_t>();
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyExtParams sendTokenIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPadExtParams<XType> tokenCopyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor, expertIdsGT[roundIndex * perRoundTokens * topK + startTokenId], expertIdsCntParams,
                copyPadExtParams);
    DataCopyPad(sendTokenIdxTensor, sendTokenIdxGT[roundIndex * perRoundTokens * topK + startTokenId],
                sendTokenIdxParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(h * sizeof(XType)), 0U, 0U, 0U};
    for (int32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor(tokenIndex - startTokenId);
        if (dstExpertId < 0 || dstExpertId >= moeExpertNum) {
            continue;
        }
        int32_t curExpertCnt = sendTokenIdxTensor(tokenIndex - startTokenId);
        int32_t dstExpertOffset = sendOffsetTensor(dstExpertId);
        GM_ADDR rankGM = (__gm__ uint8_t *)(shareGM + hOutGMAlignSize * (dstExpertOffset + curExpertCnt));
        dstGT.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);

        if constexpr (DynamicQuant) {
            xInTensor = xInQueue.AllocTensor<XType>();
            DataCopyPad(xInTensor, xGT[(roundIndex * perRoundTokens + tokenIndex / topK) * h], xCopyParams,
                        tokenCopyPadExtParams);
            xInQueue.EnQue(xInTensor);
            xInTensor = xInQueue.DeQue<XType>();
            xOutTensor = xOutQueue.AllocTensor<ExpandXOutType>();
            QuantProcess();
            xOutQueue.EnQue(xOutTensor);
            xOutTensor = xOutQueue.DeQue<ExpandXOutType>();
            FillTriple(xOutTensor, (roundIndex * perRoundTokens + tokenIndex / topK), tokenIndex % topK);
            DataCopyPad(dstGT, xOutTensor, hCommuCopyOutParams);
            xOutQueue.FreeTensor(xOutTensor);
        } else {
            xTmpTensor = xQueue.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor, xGT[(roundIndex * perRoundTokens + tokenIndex / topK) * h], xCopyParams,
                        tokenCopyPadExtParams);
            xQueue.EnQue(xTmpTensor);
            xTmpTensor = xQueue.DeQue<ExpandXOutType>();
            FillTriple(xTmpTensor, (roundIndex * perRoundTokens + tokenIndex / topK), tokenIndex % topK);
            DataCopyPad(dstGT, xTmpTensor, hCommuCopyOutParams);
            xQueue.FreeTensor<ExpandXOutType>(xTmpTensor);
        }
    }
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::SetStatus()
{
    uint32_t startExpId, endExpId, expNumPerCore;
    expNumPerCore = statusNumPerCore;
    startExpId = startStatusId;
    endExpId = endStatusId;
    if (startExpId > moeExpertNum) {
        SyncAll<true>();
        return;
    }
    uint32_t statusCntAlign = Ceil(expNumPerCore, 8) * 8;
    tpipe_->InitBuffer(statusBuf, statusCntAlign * UB_ALIGN);  // moeNum / 48 * 32
    statusTensor = statusBuf.Get<int32_t>();
    Duplicate<int32_t>(statusTensor, 0, expNumPerCore * 8);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor, 0x3F800000, mask, statusCntAlign / 8, 1, 8);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    for (uint32_t i = startExpId; i < endExpId; ++i) {
        uint32_t targetRankId = i / moeExpertNumPerRank;
        uint32_t offset = stateOffset * (epRankId + i % moeExpertNumPerRank * epRankSize);
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_EP_IDX, targetRankId) + offset);
        dstStatusGT.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(dstStatusGT, statusTensor[(i - startExpId) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::SetRoundStatus()
{
    if (blockIdx >= 1) {
        return;
    }
    tpipe_->InitBuffer(roundStatusBuf, epRankSize * UB_ALIGN);
    LocalTensor<float> roundStatusTensor = roundStatusBuf.AllocTensor<float>();
    Duplicate<float>(roundStatusTensor, 1.0, FLOAT_NUM_PER_ALIGN);
    for (uint32_t i = 0; i < epRankSize; ++i) {
        uint32_t targetRankId = i;
        uint32_t offset = stateOffset * epRankId;
        GM_ADDR rankGM = GetRoundStateAddrByRankId(COMM_EP_IDX, targetRankId) + offset;
        dstRoundStatusGT.SetGlobalBuffer((__gm__ float *)rankGM);
        DataCopy<float>(dstRoundStatusGT, roundStatusTensor, FLOAT_NUM_PER_ALIGN);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    roundStatusBuf.FreeTensor(roundStatusTensor);
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::WaitStatus()
{
    tpipe_->Reset();
    uint32_t waitStatusBufSize = (((statusNumPerCore * UB_ALIGN) > 256) ? (statusNumPerCore * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf, waitStatusBufSize);                // moeNum /48 * 32B = 43 * 32B
    tpipe_->InitBuffer(gatherMaskOutBuf, moeExpertNum * sizeof(float));  // moeNum * 4B
    tpipe_->InitBuffer(scalarBuf, UB_ALIGN * 3);                         // 96B
    tpipe_->InitBuffer(xQueue, BUFFER_NUM, hOutUBAlignSize);             // 28K
    tpipe_->InitBuffer(recvOffsetBuf, moeExpertNum * sizeof(int32_t));   // moeNum * 4B
    tpipe_->InitBuffer(recvCountBuf, moeExpertNum * sizeof(int32_t));    // moeNum * 4B

    if (isEnableDiagnose) {
        waitRecvCostStatsBufSize = Ceil(statusNumPerCore * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(waitRecvCostStatsOutQueue, BUFFER_NUM, waitRecvCostStatsBufSize);
        tpipe_->InitBuffer(recvStatusBuf, waitRecvCostStatsBufSize * 2);

        waitRecvCostStatsTensor = waitRecvCostStatsOutQueue.AllocTensor<int32_t>();
        recvStatusTensor1 = recvStatusBuf.GetWithOffset<float>(waitRecvCostStatsBufSize, 0);
        recvStatusTensor2 = recvStatusBuf.GetWithOffset<float>(waitRecvCostStatsBufSize, waitRecvCostStatsBufSize);

        Duplicate<int32_t>(waitRecvCostStatsTensor, 0, waitRecvCostStatsBufSize / sizeof(int32_t));
        Duplicate<float>(recvStatusTensor1, 0, waitRecvCostStatsBufSize / sizeof(float));
        Duplicate<float>(recvStatusTensor2, 0, waitRecvCostStatsBufSize / sizeof(float));
    }

    recvOffsetTensor = recvOffsetBuf.Get<int32_t>();
    recvCountTensor = recvCountBuf.Get<int32_t>();
    DataCopyExtParams recvOffsetParams = {1U, static_cast<uint32_t>(moeExpertNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyExtParams recvCountParams = {1U, static_cast<uint32_t>(moeExpertNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};

    DataCopyPad(recvOffsetTensor, recvOffsetGT[roundIndex * moeExpertNum], recvOffsetParams, copyPadExtParams);

    DataCopyPad(recvCountTensor, recvCountGT[roundIndex * moeExpertNum], recvCountParams, copyPadExtParams);

    if (startStatusId >= moeExpertNum) {
        SyncAll<true>();
        return;
    }

    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf.Get<float>();
    LocalTensor<float> statusSumOutTensor = scalarBuf.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    LocalTensor<float> statusFp32Tensor = waitStatusBuf.Get<float>();
    GlobalTensor<float> windowInstatusFp32Tensor;
    windowInstatusFp32Tensor.SetGlobalBuffer((__gm__ float *)(GetWindStateAddrByRankId(COMM_EP_IDX, epRankId)));
    uint32_t mask = 1;
    float compareTarget = static_cast<float>(1.0) * statusNumPerCore;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(statusNumPerCore), 1, 0, 0};

    int64_t systemCycleStart = 0;
    if (isEnableDiagnose) {
        systemCycleStart = GetSystemCycle();
    }

    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor, windowInstatusFp32Tensor[startStatusId * stateOffset / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor, gatherMaskOutTensor, mask, statusNumPerCore, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);

        if (isEnableDiagnose) {
            int32_t durationTime = static_cast<int32_t>((GetSystemCycle() - systemCycleStart) / CYCLE_TO_TIME);  // us
            SyncFunc<AscendC::HardEvent::S_V>();
            int32_t repeatTimes = Ceil(statusNumPerCore, 8);  // 8 is the num of blocks within one iteration
            int mask2 = (statusNumPerCore > 8 ? 8 : statusNumPerCore) * 8;  // num of elements within one iteration
            AscendC::BlockReduceSum<float>(recvStatusTensor1, statusFp32Tensor, repeatTimes, mask2, 1, 1, 8);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (uint32_t i = 0; i < statusNumPerCore; ++i) {
                if (recvStatusTensor1.GetValue(i) != recvStatusTensor2.GetValue(i)) {
                    int32_t srcRank = (i + startStatusId) / moeExpertNumPerRank - srcRankOffset;
                    int32_t preTime = waitRecvCostStatsTensor.GetValue(srcRank);
                    waitRecvCostStatsTensor.SetValue(srcRank, preTime + durationTime);
                    float preStatus = recvStatusTensor1.GetValue(i);
                    recvStatusTensor2.SetValue(i, preStatus);
                }
            }
        }
    }

    if (isEnableDiagnose) {
        // copy waitRecvCostStats from UB to GM
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::SetAtomicAdd<int32_t>();
        DataCopyExtParams statsCopyOutParams = {1U, waitRecvCostStatsBufSize, 0U, 0U, 0U};
        DataCopyPad<int32_t>(waitRecvCostStatsGT[srcRankOffset], waitRecvCostStatsTensor, statsCopyOutParams);
        AscendC::SetAtomicNone();
        waitRecvCostStatsOutQueue.FreeTensor<int32_t>(waitRecvCostStatsTensor);
    }

    // 清状态
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(statusNumPerCore), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(statusNumPerCore, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor[startStatusId * stateOffset / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::WaitRoundStatus()
{
    tpipe_->Reset();
    if (blockIdx >= 1) {
        return;
    }
    tpipe_->InitBuffer(roundStatusBuf, epRankSize * sizeof(float));
    tpipe_->InitBuffer(tempRoundStatusBuf, epRankSize * sizeof(float));
    uint32_t count = epRankSize * FLOAT_NUM_PER_ALIGN;
    uint32_t inner = (count * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
    GM_ADDR roundStateGM = GetRoundStateAddrByRankId(COMM_EP_IDX, epRankId);
    GlobalTensor<float> roundStatusGMTensor;

    roundStatusGMTensor.SetGlobalBuffer((__gm__ float *)roundStateGM);
    float current = (float)0.0;
    float target = (float)(1.0) * epRankSize * FLOAT_NUM_PER_ALIGN;
    SumParams sumPerRankParams{1, inner, count};
    LocalTensor<float> stateTensorLocal = roundStatusBuf.Get<float>();
    LocalTensor<float> tempRoundStateTensorLocal = tempRoundStatusBuf.Get<float>();

    int64_t systemCycleBefore = AscendC::GetSystemCycle();
    while (current != target) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensorLocal, roundStatusGMTensor, count);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(tempRoundStateTensorLocal, stateTensorLocal, sumPerRankParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        current = tempRoundStateTensorLocal.GetValue(0);
        int64_t systemCycleAfter = AscendC::GetSystemCycle();
    }

    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(tempRoundStateTensorLocal, (float)0.0, count);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(roundStatusGMTensor, tempRoundStateTensorLocal, count);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::ShareToOutputLongSeq()
{
    if (startStatusId >= moeExpertNum) {
        return;
    }

    tpipe_->InitBuffer(expertGlobalOffsetBuf, moeExpertNumPerRank * sizeof(int32_t));
    expertGlobalOffsetTensor = expertGlobalOffsetBuf.Get<int32_t>();
    DataCopyExtParams expertGlobalOffsetParams{1U, static_cast<uint32_t>(sizeof(int32_t) * moeExpertNumPerRank), 0U, 0U,
                                               0U};
    DataCopyPadExtParams<int32_t> expertGlobalOffsetCopyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(expertGlobalOffsetTensor, expertGlobalOffsetGT, expertGlobalOffsetParams,
                expertGlobalOffsetCopyPadExtParams);

    tpipe_->InitBuffer(srcrankInExpertOffsetBuf, moeExpertNum * sizeof(int32_t));
    srcrankInExpertOffsetTensor = srcrankInExpertOffsetBuf.Get<int32_t>();
    DataCopyExtParams srcrankInExpertOffsetParams{1U, static_cast<uint32_t>(sizeof(int32_t) * moeExpertNum), 0U, 0U,
                                                  0U};
    DataCopyPadExtParams<int32_t> srcrankInExpertOffsetCopyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(srcrankInExpertOffsetTensor, srcrankInExpertOffsetGT, srcrankInExpertOffsetParams,
                srcrankInExpertOffsetCopyPadExtParams);

    tpipe_->InitBuffer(rInSrcrankOffsetBuf, round * moeExpertNum * sizeof(int32_t));
    rInSrcrankOffsetTensor = rInSrcrankOffsetBuf.Get<int32_t>();
    DataCopyExtParams CParams{1U, static_cast<uint32_t>(sizeof(int32_t) * moeExpertNum * round), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> CCopyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(rInSrcrankOffsetTensor, rInSrcrankOffsetGT, CParams, CCopyPadExtParams);

    uint32_t fromRank, count, preCount, recvOffset, targetOffset, local_e;
    DataCopyPadExtParams<ExpandXOutType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyExandIdxParams{1U, sizeof(int32_t) * EXPAND_IDX_INFO, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(statusNumPerCore * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams = {1U, static_cast<uint32_t>(h * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    LocalTensor<int32_t> xTmpTensorInt;
    AscendC::TQueSync<PIPE_MTE2, PIPE_S> recvCountLocalSync;
    recvCountLocalSync.SetFlag(0);
    recvCountLocalSync.WaitFlag(0);

    for (uint32_t i = startStatusId; i < endStatusId; ++i) {
        preCount = 0;
        if (likely(i != 0)) {
            preCount = recvCountTensor(i - 1);
        }

        fromRank = i % epRankSize;
        local_e = i / epRankSize;
        count = recvCountTensor(i) - preCount;
        recvOffset = recvOffsetTensor(i);

        // 目标地址 = 专家全局起始 + B[es_idx]（源rank在专家内偏移） + r_in_srcrank_offset[c_idx]（轮次在源rank内偏移）
        int32_t rInSrcrankIndex = local_e * epRankSize * round + fromRank * round + roundIndex;
        int32_t expertGlobalOffset = expertGlobalOffsetTensor(local_e);
        int32_t srcrankInExpertOffset = srcrankInExpertOffsetTensor(i);
        int32_t rInSrcrankOffset = rInSrcrankOffsetTensor(rInSrcrankIndex);
        int32_t writeOffset = expertGlobalOffset + srcrankInExpertOffset + rInSrcrankOffset;

        GM_ADDR recvStart =
            (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, fromRank)) + recvOffset * hOutGMAlignSize;
        GlobalTensor<ExpandXOutType> srcTokenGT, dstTokenGT;
        for (uint32_t j = 0; j < count; ++j) {
            srcTokenGT.SetGlobalBuffer((__gm__ ExpandXOutType *)(recvStart + j * hOutGMAlignSize));
            xTmpTensor = xQueue.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor, srcTokenGT, hCommuCopyOutParams, copyPadExtParams);

            xQueue.EnQue(xTmpTensor);
            xTmpTensor = xQueue.DeQue<ExpandXOutType>();
            xTmpTensorInt = xTmpTensor.template ReinterpretCast<int32_t>();
            DataCopyPad(expandIdxOutGT[(writeOffset + j) * EXPAND_IDX_INFO], xTmpTensorInt[expandIdxStartIdx],
                        dataCopyExandIdxParams);

            if constexpr (DynamicQuant) {
                DataCopyExtParams floatDataCopyParams = {1U, sizeof(float), 0U, 0U, 0U};
                LocalTensor<float> xOutFp32Tensor = xTmpTensor.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGT[writeOffset + j], xOutFp32Tensor[hUBAlignSize / sizeof(float)],
                            floatDataCopyParams);
            }

            dstTokenGT.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM) + (writeOffset + j) * h, h);
            DataCopyPad(dstTokenGT, xTmpTensor, expandXCopyParams);

            xQueue.FreeTensor(xTmpTensor);
        }
    }
}

template <CamTypeClass>
__aicore__ inline void CamMoeDispatchNormal<CamTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        uint32_t realRound = (realMaxBatchSize + perRoundTokens - 1) / perRoundTokens;
        while (roundIndex < realRound) {
            InputToShare();
            SetStatus();
            WaitStatus();
            ShareToOutputLongSeq();
            if (realRound > 1) {
                SyncAll<true>();
                SetRoundStatus();
                WaitRoundStatus();
                SyncAll<true>();
            }
            roundIndex += 1;
        }
    }
}

}  // namespace CamMoeDispatchNormalImpl
#endif
