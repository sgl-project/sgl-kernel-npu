#ifndef CAM_MOE_COMBINE_NORMAL_H
#define CAM_MOE_COMBINE_NORMAL_H

#include "shmem_api.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "cam_moe_combine_normal_tiling.h"
#include "comm_args.h"

namespace CamMoeCombineNormalImpl {
constexpr uint32_t RANK_ID_OFFSET_IN_SRC_INFO = 0U;
constexpr uint32_t TOKEN_IDX_OFFSET_IN_SRC_INFO = 1U;
constexpr uint32_t TOPK_IDX_OFFSET_IN_SRC_INFO = 2U;
// constexpr uint64_t COMBINE_STATE_WIN_OFFSET = Moe::NOTIFY_DISPATCH_BUFF_OFFSET;
constexpr uint64_t MAGIC_WIN_OFFSET = 975UL * 1024UL;
constexpr uint32_t TOKEN_SRC_INFO_LEN = 3U;
constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t MUL_256_ALIGN = 256U;
constexpr uint64_t WIN_512_ALIGN = 512UL;
constexpr uint32_t FLOAT_NUM_PER_ALIGN = 8U;
constexpr uint8_t DOUBLE_BUFFER = 2;
constexpr int64_t CYCLE_TO_TIME = 50;  // cycle num is converted into a fixed base unit of time, set at 50

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename RecvXType, typename XType, typename SrcInfoType
#define TemplateMC2TypeFunc RecvXType, XType, SrcInfoType

using namespace AscendC;
template <TemplateMC2TypeClass>
class CamMoeCombineNormal
{
public:
    __aicore__ inline CamMoeCombineNormal(){};
    __aicore__ inline void Init(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                GM_ADDR topkIdx, GM_ADDR sendTokenIdx, GM_ADDR tpRecvCount, GM_ADDR XOut,
                                GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const CamMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMagic();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                            GM_ADDR topkWeights, GM_ADDR sendTokenIdx, GM_ADDR XOut,
                                            GM_ADDR sendCostStatsOut);
    __aicore__ inline void InitTilingData(const CamMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void InitBuffLen();
    __aicore__ inline void CopyBufferToShareAndSetStatus();
    __aicore__ inline void CopyBufferToShare(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId,
                                             uint32_t tkIndex);
    __aicore__ inline void ReadBufferFromRemote();
    __aicore__ inline void WaitBuffCopy(uint32_t tokenIndex);
    __aicore__ inline void SetStatusBySrcInfo(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId);
    __aicore__ inline void ReadBufferAndWeightedSum(uint32_t tokenIndex, uint32_t startTokenIndex);
    __aicore__ inline void AllGatherRecvCount();

    // __aicore__ GM_ADDR GetStateAddrByRankId(const int32_t rankId)
    // {
    //     GM_ADDR bufferAddr;
    //     if (epRankId_ == rankId) {
    //         bufferAddr = (GM_ADDR)epWinContext_->localWindowsIn;
    //     } else {
    //         bufferAddr = (GM_ADDR)((HcclRankRelationResV2
    //         *)epWinContext_->remoteRes[rankId].nextDevicePtr)->windowsIn;
    //     }
    //     return (GM_ADDR)(bufferAddr + stateWinOffset_ + Moe::NOTIFY_DISPATCH_BUFF_OFFSET);
    // }
    /* shmem meta data buffer
      |notify_magic|dispatch_combine_magic|notify_dispatch_flag_0|dispatch_flag_0|combine_flag_0|notify_dispatch_flag_1|dispatch_flag_1|combine_flag_1|
      |50KB|50KB|1M|2M|2M|1M|2M|2M|
    */
    // __aicore__ GM_ADDR GetCombineMagicAddr()
    // {
    //     return (GM_ADDR)(metaDataGvaGM_ + Moe::COMBINE_MAGIC_OFFSET);
    // }

    // __aicore__ GM_ADDR GetCombineFlagAddr()
    // {
    //     return (GM_ADDR)(metaDataGvaGM_ + Moe::NOTIFY_DISPATCH_BUFF_OFFSET);
    // }

    // __aicore__ GM_ADDR GetBufferAddrByRankId(const int32_t rankId)
    // {
    //     return GetStateAddrByRankId(rankId) + COMBINE_STATE_WIN_OFFSET;
    // }

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx)
    {
        perCoreNum = totalNum / aivNum_;
        uint32_t remainderRankNum = totalNum % aivNum_;

        startIdx = perCoreNum * coreIdx_;
        if (coreIdx_ < remainderRankNum) {
            perCoreNum++;
            startIdx += coreIdx_;
        } else {
            startIdx += remainderRankNum;
        }
        endIdx = startIdx + perCoreNum;
    }

    __gm__ HcclOpResParam *epWinContext_{nullptr};
    __gm__ HcclOpResParam *tpWinContext_{nullptr};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint64_t magic_{0};
    uint64_t stateWinOffset_{0};
    uint64_t metaStateWinOffset_{0};
    uint64_t dataStateWinOffset_{0};
    uint32_t selfSendCnt_{0};
    uint32_t hRecvXTypeLen_{0};
    uint32_t h32AlignFloatLen_{0};
    uint32_t h256AlignFloatLen_{0};
    uint32_t h32AlignRecvXLen_{0};
    uint32_t h512AlignRecvXLen_{0};
    uint32_t sendCostStatsBufSize_{0};

    bool isEnableDiagnose_{false};

    TPipe *tpipe_{nullptr};
    TQue<QuePosition::VECIN, 1> weightedSumQueue_;
    TQue<QuePosition::VECOUT, 1> sendCostStatsOutQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> localCopyQueue_;
    TBuf<> stateBuf_;
    TBuf<> topkWeightsBuf_;
    TBuf<> sendTokenIdxBuf_;
    TBuf<> tokenFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> weightedMulBuf_;
    TBuf<> srcInfoBuf_;
    TBuf<> xOutBuf_;
    TBuf<> tempStateBuf_;
    TBuf<> baseAddrFlagBuf_;
    TBuf<> allRecvCountBuf_;
    TBuf<> topkIdxBuf_;

    LocalTensor<uint64_t> baseAddrLT_;

    GlobalTensor<RecvXType> recvXGM_;
    GlobalTensor<SrcInfoType> tokenSrcInfoGM_;
    GlobalTensor<SrcInfoType> epRecvCountGT_;
    GlobalTensor<float> topkWeightsGT_;
    GlobalTensor<int32_t> sendTokenIdxGT_;
    GlobalTensor<int32_t> topkIdxGT_;
    GlobalTensor<XType> xOutGlobal_;
    GlobalTensor<int32_t> sendCostStatsGT_;
    GlobalTensor<uint64_t> xOutAddrGT_;
    GlobalTensor<SrcInfoType> allRecvCountGT_;
    GM_ADDR localRankGM_;
    GM_ADDR XOutGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR metaDataGvaGM_;
    GM_ADDR metaStateGvaGM_;
    GM_ADDR dataStateGvaGM_;
    __gm__ int32_t *allRecvCountGM_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::InitMagic()
{
    // auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    // epWinContext_ = (__gm__ HcclOpResParam *)contextGM0;
    // metaDataGvaGM_ 需要在适配层或者框架侧做0初始化
    GM_ADDR magicAddr = (GM_ADDR)(metaDataGvaGM_ + Moe::COMBINE_MAGIC_OFFSET);
    GlobalTensor<uint64_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer((__gm__ uint64_t *)(magicAddr + coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    printf("[RANK %d AIC %d] magicAddr %p selfMagicAddr %p magic_ %d\n", epRankId_, coreIdx_, magicAddr,
           selfMagicTensor.GetPhyAddr(), magic_);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::InitGlobalBuffer(GM_ADDR recvX, GM_ADDR tokenSrcInfo,
                                                                                  GM_ADDR epRecvCount,
                                                                                  GM_ADDR topkWeights, GM_ADDR topkIdx,
                                                                                  GM_ADDR sendTokenIdx, GM_ADDR XOut,
                                                                                  GM_ADDR sendCostStatsOut)
{
    recvXGM_.SetGlobalBuffer((__gm__ RecvXType *)recvX);
    tokenSrcInfoGM_.SetGlobalBuffer((__gm__ SrcInfoType *)tokenSrcInfo);
    epRecvCountGT_.SetGlobalBuffer((__gm__ int32_t *)epRecvCount);
    topkWeightsGT_.SetGlobalBuffer((__gm__ float *)topkWeights);
    topkIdxGT_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    sendTokenIdxGT_.SetGlobalBuffer((__gm__ int32_t *)sendTokenIdx);
    xOutGlobal_.SetGlobalBuffer((__gm__ XType *)XOut);
    if (isEnableDiagnose_) {
        sendCostStatsGT_.SetGlobalBuffer((__gm__ int32_t *)sendCostStatsOut);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void
CamMoeCombineNormal<TemplateMC2TypeFunc>::InitTilingData(const CamMoeCombineNormalTilingData *tilingData)
{
    axisBS_ = tilingData->camMoeCombineNormalInfo.bs;
    axisH_ = tilingData->camMoeCombineNormalInfo.h;
    axisK_ = tilingData->camMoeCombineNormalInfo.k;
    aivNum_ = tilingData->camMoeCombineNormalInfo.aivNum;
    moeExpertNum_ = tilingData->camMoeCombineNormalInfo.moeExpertNum;
    moeExpertPerRankNum_ = tilingData->camMoeCombineNormalInfo.moeExpertPerRankNum;
    epWorldSize_ = tilingData->camMoeCombineNormalInfo.epWorldSize;
    epRankId_ = tilingData->camMoeCombineNormalInfo.epRankId;
    isEnableDiagnose_ = tilingData->camMoeCombineNormalInfo.isEnableDiagnose;
    metaDataGvaGM_ = (GM_ADDR)tilingData->camMoeCombineNormalInfo.metaDataPtr;
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::InitBuffLen()
{
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    h32AlignFloatLen_ = Ceil(hFloatSize, UB_32_ALIGN) * UB_32_ALIGN;
    h256AlignFloatLen_ = Ceil(hFloatSize, MUL_256_ALIGN) * MUL_256_ALIGN;
    hRecvXTypeLen_ = axisH_ * sizeof(RecvXType);
    h32AlignRecvXLen_ = Ceil(hRecvXTypeLen_, UB_32_ALIGN) * UB_32_ALIGN;
    h512AlignRecvXLen_ = Ceil(hRecvXTypeLen_, WIN_512_ALIGN) * WIN_512_ALIGN;
    if (isEnableDiagnose_) {
        sendCostStatsBufSize_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_32_ALIGN) * UB_32_ALIGN;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::Init(
    GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR topkIdx,
    GM_ADDR sendTokenIdx, GM_ADDR tpRecvCount, GM_ADDR XOut, GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const CamMoeCombineNormalTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    XOutGM_ = XOut;
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();

    InitMagic();
    InitTilingData(tilingData);
    InitGlobalBuffer(recvX, tokenSrcInfo, epRecvCount, topkWeights, topkIdx, sendTokenIdx, XOut, sendCostStatsOut);
    InitBuffLen();

    PipeBarrier<PIPE_ALL>();
    stateWinOffset_ = magic_ * Moe::FLAG_WIN_SIZE + Moe::COMBINE_FLAG_OFFSET;
    metaStateWinOffset_ = stateWinOffset_;
    dataStateWinOffset_ = stateWinOffset_ + Moe::COMBINE_META_FLAG_SIZE;
    metaStateGvaGM_ = (GM_ADDR)(metaDataGvaGM_ + metaStateWinOffset_);
    dataStateGvaGM_ = (GM_ADDR)(metaDataGvaGM_ + dataStateWinOffset_);
    printf(
        "[RANK %d AIC %d] stateWinOffset_ %d dataStateWinOffset_ %d magic_ %d FLAG_WIN_SIZE %d COMBINE_FLAG_OFFSET "
        "%d\n",
        epRankId_, coreIdx_, stateWinOffset_, dataStateWinOffset_, magic_, Moe::FLAG_WIN_SIZE,
        Moe::COMBINE_FLAG_OFFSET);
    // localRankGM_ = GetBufferAddrByRankId(epRankId_);
    DataCacheCleanAndInvalid<SrcInfoType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epRecvCountGT_[moeExpertNum_ - 1]);
    selfSendCnt_ = epRecvCountGT_(moeExpertNum_ - 1);

    allRecvCountGM_ = (__gm__ int32_t *)workspaceGM_;
    allRecvCountGT_.SetGlobalBuffer(allRecvCountGM_);
}

// allgather每个rank的recvCount，采用分核策略，
// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::AllGatherRecvCount()
// {
//     uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
//     SplitCoreCal(epWorldSize_, rankNumPerBlock, startRankId, endRankId);
//     if (rankNumPerBlock == 0U) {
//         SyncAll<true>();
//     }
//     for (uint32_t targetRankId = startRankId; targetRankId < endRankId; targetRankId++) {
//         shmem_get_int32_mem_nbi(allRecvCountGT_[targetRankId * moeExpertNum_], epRecvCountGT_, moeExpertNum_,
//         targetRankId); PipeBarrier<PIPE_ALL>();
//     }
//     SyncAll<true>();
//     printf("[RANK %d AIC %d] selfSendCnt_ %d\n", epRankId_, coreIdx_, selfSendCnt_);
// }

// 使用shmem对称内存（metaDataPtr）alltoall XOut地址
// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::AlltoAallMetaData()
// {
//     uint32_t startStatusId;
//     uint32_t endStatusId;
//     uint32_t statusNumPerCore;
//     uint32_t remainStatus;
//     tpipe_->Reset();
//     tpipe_->InitBuffer(stateBuf_, epWorldSize_ * UB_32_ALIGN);  //
//     每个rank将自己的xOut地址写到其他rank后，会写一个flag，每个flag占32B

//     LocalTensor<float> statusTensor = stateBuf_.AllocTensor<float>();
//     Duplicate<float>(statusTensor, (float)1.0, FLOAT_NUM_PER_ALIGN);

//     GlobalTensor<float> baseAddrFlagFP32GT;
//     GM_ADDR baseAddrFlagGvaGM = metaDataGvaGM_ + metaStateWinOffset_;
//     baseAddrFlagFP32GT.SetGlobalBuffer((__gm__ float *)(baseAddrFlagGvaGM));
//     uint32_t mask = 1;
//     statusNumPerCore = epWorldSize_ / aivNum_;
//     remainStatus = epWorldSize_ % blockNum;
//     startStatusId = statusNumPerCore * coreIdx_;
//     if (coreIdx_ < remainStatus) {
//         statusNumPerCore += 1;
//         startStatusId += coreIdx_;
//     } else {
//         startStatusId += remainStatus;
//     }
//     endStatusId = startStatusId + statusNumPerCore;

//     for (uint32_t statusId = startStatusId; statusId < endStatusId; statusId++) {
//         shmem_mte_put_mem_nbi<float>(baseAddrFlagGvaGM + statusId * FLOAT_NUM_PER_ALIGN, statusTensor.GetPhyAddr(),
//         FLOAT_NUM_PER_ALIGN, statusId, EVENT_ID0); SyncFunc<AscendC::HardEvent::MTE3_V>();
//     }
//     float compareTarget = static_cast<float>(1.0) * statusNumPerCore;
//     float sumOfFlag = static_cast<float>(-1.0);
//     DataCopyExtParams statsCopyOutParams = {1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};

//     SyncFunc<AscendC::HardEvent::S_V>();
//     while (sumOfFlag != compareTarget) {
//         DataCopy(statusFp32Tensor, baseAddrFlagFP32GT[startStatusId * stateOffset / sizeof(float)], intriParams);
//         SyncFunc<AscendC::HardEvent::MTE2_V>();
//         ReduceSum(statusSumOutTensor, statusFp32Tensor, gatherMaskOutTensor, mask, statusNumPerCore, 1);
//         SyncFunc<AscendC::HardEvent::V_S>();
//         sumOfFlag = statusSumOutTensor.GetValue(0);

//         if (isEnableDiagnose) {
//             int32_t durationTime = static_cast<int32_t>((GetSystemCycle() - systemCycleStart) / CYCLE_TO_TIME);  //
//             us SyncFunc<AscendC::HardEvent::S_V>(); int32_t repeatTimes = Ceil(statusNumPerCore, 8);  // 8 is the num
//             of blocks within one iteration int mask2 = (statusNumPerCore > 8 ? 8 : statusNumPerCore) * 8;  // num of
//             elements within one iteration AscendC::BlockReduceSum<float>(recvStatusTensor1, statusFp32Tensor,
//             repeatTimes, mask2, 1, 1, 8); SyncFunc<AscendC::HardEvent::V_S>(); for (uint32_t i = 0; i <
//             statusNumPerCore; ++i) {
//                 if (recvStatusTensor1.GetValue(i) != recvStatusTensor2.GetValue(i)) {
//                     int32_t srcRank = (i + startStatusId) / moeExpertNumPerRank - srcRankOffset;
//                     int32_t preTime = waitRecvCostStatsTensor.GetValue(srcRank);
//                     waitRecvCostStatsTensor.SetValue(srcRank, preTime + durationTime);
//                     float preStatus = recvStatusTensor1.GetValue(i);
//                     recvStatusTensor2.SetValue(i, preStatus);
//                 }
//             }
//         }
//     }

//     for (uint32_t targetRankId = 0; targetRankId < epWorldSize_; targetRankId++) {
//         shmem
//     }
// }

// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::CopyBufferToShareAndSetStatus()
// {
//     PipeBarrier<PIPE_ALL>();
//     uint32_t perBlockSendNum = 0, startTokenId = 0, endTokenId = 0;
//     SplitCoreCal(selfSendCnt_, perBlockSendNum, startTokenId, endTokenId);
//     if (perBlockSendNum == 0U) {
//         return;
//     }

//     uint32_t blockLen = static_cast<uint32_t>(perBlockSendNum * TOKEN_SRC_INFO_LEN * sizeof(uint32_t));
//     tpipe_->Reset();
//     tpipe_->InitBuffer(stateBuf_, UB_32_ALIGN);
//     tpipe_->InitBuffer(localCopyQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);
//     tpipe_->InitBuffer(srcInfoBuf_, blockLen);
//     LocalTensor<uint32_t> statusTensor = stateBuf_.AllocTensor<uint32_t>();
//     Duplicate<uint32_t>(statusTensor, 0x3F800000, FLOAT_NUM_PER_ALIGN);

//     LocalTensor<SrcInfoType> srcInfoLocal = srcInfoBuf_.Get<SrcInfoType>();
//     const DataCopyExtParams dataCopyParams{1U, blockLen, 0U, 0U, 0U};
//     const DataCopyPadExtParams<SrcInfoType> padParams{false, 0U, 0U, 0U};
//     DataCopyPad(srcInfoLocal, tokenSrcInfoGM_[startTokenId * TOKEN_SRC_INFO_LEN], dataCopyParams, padParams);
//     SyncFunc<AscendC::HardEvent::MTE2_S>();

//     LocalTensor<int32_t> sendCostStatsTensor;
//     if (isEnableDiagnose_) {
//         tpipe_->InitBuffer(sendCostStatsOutQueue_, DOUBLE_BUFFER, sendCostStatsBufSize_);
//         sendCostStatsTensor = sendCostStatsOutQueue_.AllocTensor<int32_t>();
//         Duplicate<int32_t>(sendCostStatsTensor, 0, sendCostStatsBufSize_ / sizeof(int32_t));
//     }

//     for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
//         uint32_t index = (tokenIndex - startTokenId) * TOKEN_SRC_INFO_LEN;
//         uint32_t srcRankId = static_cast<uint32_t>(srcInfoLocal(index + RANK_ID_OFFSET_IN_SRC_INFO));
//         uint32_t srcTokenId = static_cast<uint32_t>(srcInfoLocal(index + TOKEN_IDX_OFFSET_IN_SRC_INFO));
//         uint32_t srcTopkId = static_cast<uint32_t>(srcInfoLocal(index + TOPK_IDX_OFFSET_IN_SRC_INFO));
//         int64_t sendStartCycle = GetSystemCycle();

//         CopyBufferToShare(srcRankId, srcTokenId, srcTopkId, tokenIndex);
//         PipeBarrier<PIPE_ALL>();
//         SetStatusBySrcInfo(srcRankId, srcTokenId, srcTopkId);

//         if (isEnableDiagnose_) {
//             SyncFunc<AscendC::HardEvent::MTE3_S>();
//             int32_t durationTime = static_cast<int32_t>((GetSystemCycle() - sendStartCycle) / CYCLE_TO_TIME);  // us
//             int32_t preTime = sendCostStatsTensor.GetValue(srcRankId);
//             sendCostStatsTensor.SetValue(srcRankId, preTime + durationTime);
//         }
//     }

//     if (isEnableDiagnose_) {
//         SyncFunc<AscendC::HardEvent::S_MTE3>();
//         AscendC::SetAtomicAdd<int32_t>();
//         DataCopyExtParams statsCopyOutParams = {1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U,
//         0U}; DataCopyPad<int32_t>(sendCostStatsGT_, sendCostStatsTensor, statsCopyOutParams);
//         AscendC::SetAtomicNone();
//         sendCostStatsOutQueue_.FreeTensor<int32_t>(sendCostStatsTensor);
//     }

//     SyncFunc<AscendC::HardEvent::MTE3_S>();
// }

// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::CopyBufferToShare(uint32_t srcRankId,
//                                                                                    uint32_t srcTokenId,
//                                                                                    uint32_t srcTopkId, uint32_t
//                                                                                    tkIndex)
// {
//     uint32_t tokenOffset = tkIndex * axisH_;
//     // GM_ADDR dstGM = GetBufferAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * h512AlignRecvXLen_;
//     // dstWindow.SetGlobalBuffer((__gm__ XType *)dstGM);
//     GM_ADDR dstGM = XOutGM_ + (srcTokenId * axisK_ + srcTopkId) * hRecvXTypeLen_; //
//     由于是直接put到XOut并返回上层使用，所以这里不做512字节对齐 GlobalTensor<XType> dstWindow;
//     dstWindow.SetGlobalBuffer((__gm__ XType *)dstGM);
//     DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
//     DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};

//     LocalTensor<RecvXType> localCopyTensor;
//     localCopyTensor = localCopyQueue_.AllocTensor<RecvXType>();
//     DataCopyPad(localCopyTensor, recvXGM_[tokenOffset], xOutCopyParams, copyPadExtParams);
//     localCopyQueue_.EnQue(localCopyTensor);
//     localCopyTensor = localCopyQueue_.DeQue<RecvXType>();
//     // DataCopyPad(dstWindow, localCopyTensor, xOutCopyParams);
//     shmem_mte_put_mem_nbi<XType>(dstWindow, localCopyTensor, axisH_, srcRankId, EVENT_ID0);

//     localCopyQueue_.FreeTensor<RecvXType>(localCopyTensor);
// }

// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::SetStatusBySrcInfo(uint32_t srcRankId,
//                                                                                     uint32_t srcTokenId,
//                                                                                     uint32_t srcTopkId)
// {
//     LocalTensor<uint32_t> statusTensor = stateBuf_.AllocTensor<uint32_t>();
//     // GM_ADDR stateGM = GetStateAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * UB_32_ALIGN;
//     GM_ADDR stateGM = dataStateGvaGM_ + (srcTokenId * axisK_ + srcTopkId) * UB_32_ALIGN;
//     GlobalTensor<uint32_t> stateGMTensor;
//     stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
//     // DataCopy<uint32_t>(stateGMTensor, statusTensor, FLOAT_NUM_PER_ALIGN);
//     shmem_mte_put_mem_nbi<uint32_t>(stateGMTensor, statusTensor, 1, srcRankId, EVENT_ID0);
// }

// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::WaitBuffCopy(uint32_t tokenIndex)
// {
//     uint32_t calCount = axisK_ * FLOAT_NUM_PER_ALIGN;
//     // GM_ADDR stateGM = GetStateAddrByRankId(epRankId_) + tokenIndex * axisK_ * UB_32_ALIGN;  // 计算地址偏移
//     GM_ADDR stateGM = dataStateGvaGM_ + tokenIndex * axisK_ * UB_32_ALIGN;  // 计算地址偏移
//     GlobalTensor<float> stateGMTensor;
//     stateGMTensor.SetGlobalBuffer((__gm__ float *)stateGM);
//     float current = (float)0.0;
//     float target = (float)1.0 * axisK_ * FLOAT_NUM_PER_ALIGN;
//     SumParams sumPerKParams{1, calCount, calCount};
//     LocalTensor<float> stateTensorLocal = stateBuf_.Get<float>();
//     LocalTensor<float> tempStateTensorLocal = tempStateBuf_.Get<float>();
//     while (current != target) {
//         SyncFunc<AscendC::HardEvent::S_MTE2>();
//         DataCopy<float>(stateTensorLocal, stateGMTensor, calCount);
//         SyncFunc<AscendC::HardEvent::MTE2_V>();
//         Sum(tempStateTensorLocal, stateTensorLocal, sumPerKParams);
//         SyncFunc<AscendC::HardEvent::V_S>();
//         current = tempStateTensorLocal(0);
//     }
//     SyncFunc<AscendC::HardEvent::S_V>();
//     Duplicate<float>(tempStateTensorLocal, (float)0.0, calCount);
//     SyncFunc<AscendC::HardEvent::V_MTE3>();
//     DataCopy<float>(stateGMTensor, tempStateTensorLocal, calCount);
// }

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferAndWeightedSum(uint32_t tokenIndex,
                                                                                          uint32_t startTokenIndex)
{
    LocalTensor<float> tokenFloatLocal = tokenFloatBuf_.Get<float>();
    LocalTensor<float> weightedMulBufLocal = weightedMulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal = sumFloatBuf_.Get<float>();
    LocalTensor<float> topkWeightsLocal = topkWeightsBuf_.Get<float>();
    LocalTensor<int32_t> sendTokenIdxLocal = sendTokenIdxBuf_.Get<int32_t>();
    LocalTensor<uint32_t> stateTensorLocal = stateBuf_.Get<uint32_t>();
    LocalTensor<int32_t> allRecvCountLocal = allRecvCountBuf_.Get<int32_t>();
    LocalTensor<int32_t> topkIdxLocal = topkIdxBuf_.Get<int32_t>();
    Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);
    const DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};

    for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
        int32_t localTokenIdx = (tokenIndex - startTokenIndex) * axisK_ + topkId;
        float scale = topkWeightsLocal.GetValue(localTokenIdx);
        int32_t expertId = topkIdxLocal.GetValue(localTokenIdx);
        int32_t remoteReadOffset = sendTokenIdxLocal(localTokenIdx);
        int32_t remoteReadBase = allRecvCountLocal(expertId * epWorldSize_ + epRankId_);
        uint64_t remoteReadAddr = static_cast<uint64_t>(remoteReadBase + remoteReadOffset) * hRecvXTypeLen_;

        // GM_ADDR localTokenAddr = localRankGM_ + (tokenIndex * axisK_ + topkId) * h512AlignRecvXLen_;
        // GM_ADDR localTokenAddr = XOutGM_ + (tokenIndex * axisK_ + topkId) * hRecvXTypeLen_;
        // GlobalTensor<XType> localTokenTensor = xOutGlobal_[(tokenIndex * axisK_ + topkId) *axisH_];
        // localTokenTensor.SetGlobalBuffer((__gm__ XType *)localTokenAddr);

        LocalTensor<XType> tmpToken = weightedSumQueue_.AllocTensor<XType>();
        const DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};
        shmem_get_half_mem_nbi(tmpToken, recvX[remoteReadAddr / sizeof(XType)], axisH_,
                               expertId / moeExpertPerRankNum_);
        weightedSumQueue_.EnQue(tmpToken);
        tmpToken = weightedSumQueue_.DeQue<XType>();
        Cast(tokenFloatLocal, tmpToken, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(weightedMulBufLocal, tokenFloatLocal, scale, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, weightedMulBufLocal, axisH_);
        weightedSumQueue_.FreeTensor<XType>(tmpToken);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> xOutLocal = xOutBuf_.Get<XType>();
    Cast(xOutLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(xOutGlobal_[tokenIndex * axisH_], xOutLocal, xOutCopyParams);
}

// template <TemplateMC2TypeClass>
// __aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferFromRemote()
// {
//     if (axisBS_ == 0U) {
//         return;
//     }
//     uint32_t tokenPerBlock = 0U, startTokenIndex = 0U, endTokenIndex = 0U;
//     SplitCoreCal(axisBS_, tokenPerBlock, startTokenIndex, endTokenIndex);

//     if (tokenPerBlock == 0U) {
//         return;
//     }

//     tpipe_->Reset();
//     tpipe_->InitBuffer(xOutBuf_, h32AlignRecvXLen_);
//     tpipe_->InitBuffer(tokenFloatBuf_, h32AlignFloatLen_);
//     tpipe_->InitBuffer(weightedMulBuf_, h256AlignFloatLen_);
//     tpipe_->InitBuffer(sumFloatBuf_, h32AlignFloatLen_);
//     tpipe_->InitBuffer(weightedSumQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);
//     tpipe_->InitBuffer(stateBuf_, (axisK_)*UB_32_ALIGN);
//     tpipe_->InitBuffer(tempStateBuf_, (axisK_)*UB_32_ALIGN);
//     tpipe_->InitBuffer(topkWeightsBuf_, tokenPerBlock * axisK_ * sizeof(float));

//     LocalTensor<float> topkWeightsLocal = topkWeightsBuf_.Get<float>();
//     const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(tokenPerBlock * axisK_ * sizeof(float)), 0U, 0U, 0U};
//     const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
//     DataCopyPad(topkWeightsLocal, topkWeightsGT_[startTokenIndex * axisK_], bskParams, copyPadFloatParams);
//     SyncFunc<AscendC::HardEvent::MTE2_S>();

//     for (uint32_t tokenIndex = startTokenIndex; tokenIndex < endTokenIndex; tokenIndex++) {
//         WaitBuffCopy(tokenIndex);
//         SyncFunc<AscendC::HardEvent::MTE3_V>();  // 与结果搬出datacopy同tensor
//         ReadBufferAndWeightedSum(tokenIndex, startTokenIndex);
//     }
// }

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferFromRemote()
{
    if (axisBS_ == 0U) {
        return;
    }
    uint32_t tokenPerBlock = 0U, startTokenIndex = 0U, endTokenIndex = 0U;
    SplitCoreCal(axisBS_, tokenPerBlock, startTokenIndex, endTokenIndex);

    if (tokenPerBlock == 0U) {
        return;
    }

    tpipe_->Reset();
    tpipe_->InitBuffer(xOutBuf_, h32AlignRecvXLen_);                          // 14KB
    tpipe_->InitBuffer(tokenFloatBuf_, h32AlignFloatLen_);                    // 14KB
    tpipe_->InitBuffer(weightedMulBuf_, h256AlignFloatLen_);                  // 14KB
    tpipe_->InitBuffer(sumFloatBuf_, h32AlignFloatLen_);                      // 14KB
    tpipe_->InitBuffer(weightedSumQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);  // 2 * 14KB = 28KB
    tpipe_->InitBuffer(stateBuf_, (axisK_)*UB_32_ALIGN);                      // 8 * 32B = 256B
    tpipe_->InitBuffer(tempStateBuf_, (axisK_)*UB_32_ALIGN);                  // 8 * 32B = 256B
    tpipe_->InitBuffer(topkWeightsBuf_, tokenPerBlock * axisK_ *
                                            sizeof(float));  // bs最大为128K，tensor大小为128K/48*8*sizeof(float)=85K
    tpipe_->InitBuffer(sendTokenIdxBuf_, tokenPerBlock * axisK_ *
                                             sizeof(int32_t));  // bs最大为128K，tensor大小为128K/48*8*sizeof(int)=85K。
    tpipe_->InitBuffer(topkIdxBuf_, tokenPerBlock * axisK_ * sizeof(int32_t));
    tpipe_->InitBuffer(
        allRecvCountBuf_,
        epWorldSize_ * moeExpertPerRankNum_ *
            sizeof(int32_t));  // moeExpertNum最大为512，tensor大小为512*sizeof(int)=2K。 85K+85K+2K=172K < 192K

    LocalTensor<float> topkWeightsLocal = topkWeightsBuf_.Get<float>();
    LocalTensor<int32_t> sendTokenIdxLocal = sendTokenIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> allRecvCountLocal = allRecvCountBuf_.Get<int32_t>();
    LocalTensor<int32_t> topkIdxLocal = topkIdxBuf_.Get<int32_t>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(tokenPerBlock * axisK_ * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> copyPadint32Params{false, 0U, 0U, 0U};
    DataCopyPad(topkWeightsLocal, topkWeightsGT_[startTokenIndex * axisK_], bskParams, copyPadint32Params);
    DataCopyPad(sendTokenIdxLocal, sendTokenIdxGT_[startTokenIndex * axisK_], bskParams, copyPadint32Params);
    DataCopyPad(topkIdxLocal, topkIdxGT_[startTokenIndex * axisK_], bskParams, copyPadint32Params);
    DataCopy(allRecvCountLocal, allRecvCountGT_, epWorldSize_ * moeExpertPerRankNum_ * sizeof(int32_t));

    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = startTokenIndex; tokenIndex < endTokenIndex; tokenIndex++) {
        // WaitBuffCopy(tokenIndex);
        // SyncFunc<AscendC::HardEvent::MTE3_V>();  // 与结果搬出datacopy同tensor
        ReadBufferAndWeightedSum(tokenIndex, startTokenIndex);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormal<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        // AlltoAallMetaData();
        // AllGatherRecvCount();
        // CopyBufferToShareAndSetStatus();
        ReadBufferFromRemote();
    }
}

}  // namespace CamMoeCombineNormalImpl
#endif  // MOE_COMBINE_IMPL_H
