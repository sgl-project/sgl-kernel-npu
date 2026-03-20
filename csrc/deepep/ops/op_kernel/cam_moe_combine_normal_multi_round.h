#ifndef CAM_MOE_COMBINE_NORMAL_MULTI_ROUND_H
#define CAM_MOE_COMBINE_NORMAL_MULTI_ROUND_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "cam_moe_combine_normal_tiling.h"
#include "comm_args.h"

namespace CamMoeCombineNormalMultiRoundImpl {
constexpr uint32_t RANK_ID_OFFSET_IN_SRC_INFO = 0U;
constexpr uint32_t TOKEN_IDX_OFFSET_IN_SRC_INFO = 1U;
constexpr uint32_t TOPK_IDX_OFFSET_IN_SRC_INFO = 2U;
constexpr uint64_t STATE_WIN_SIZE = 4UL * 1024UL * 1024UL;
constexpr uint64_t STATE_WIN_SIZE_HALF = STATE_WIN_SIZE / 2;
constexpr uint64_t MAGIC_WIN_OFFSET = 975UL * 1024UL;
constexpr uint64_t ROUND_STATE_OFFSET = Moe::BASE_ROUND_STATE_OFFSET + Moe::ROUND_STATE_MAX_SIZE * 2UL;  // 458*1024
constexpr uint32_t TOKEN_SRC_INFO_LEN = 3U;
constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t MUL_256_ALIGN = 256U;
constexpr uint64_t WIN_512_ALIGN = 512UL;
constexpr uint32_t FLOAT_NUM_PER_ALIGN = 8U;
constexpr uint8_t DOUBLE_BUFFER = 2;
constexpr uint32_t WAIT_ROUND_INDEX = 2U;
constexpr int64_t CYCLE_TO_TIME = 50;  // cycle num is converted into a fixed base unit of time, set at 50
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t BATCH_SRC_INFO_CNT = 128U;

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
class CamMoeCombineNormalMultiRound
{
public:
    __aicore__ inline CamMoeCombineNormalMultiRound(){};
    __aicore__ inline void Init(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                GM_ADDR tokenIdx, GM_ADDR tpRecvCount, GM_ADDR XOut, GM_ADDR sendCostStatsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const CamMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMagic();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                            GM_ADDR topkWeights, GM_ADDR tokenIdx, GM_ADDR XOut,
                                            GM_ADDR sendCostStatsOut);
    __aicore__ inline void InitTilingData(const CamMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void InitBuffLen();
    __aicore__ inline void CopyBufferToShareAndSetStatus();
    __aicore__ inline void CopyBufferToShare(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId,
                                             uint32_t tkIndex);
    __aicore__ inline void ReadBufferFromRemote();
    __aicore__ inline void WaitBuffCopy(uint32_t recvXTokenIdx, uint32_t topkWeightTokenIdx);
    __aicore__ inline void SetStatusBySrcInfo(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId);
    __aicore__ inline void ReadBufferAndWeightedSum(uint32_t recvXTokenIdx, uint32_t topkWeightTokenIdx);
    __aicore__ inline void InitRoundSendData();
    __aicore__ inline void SetRoundStatus();
    __aicore__ inline void WaitRoundStatus();
    __aicore__ inline void InitRoundRecvData();

    __aicore__ GM_ADDR GetStateAddrByRankId(const int32_t rankId)
    {
        GM_ADDR bufferAddr;
        if (epRankId_ == rankId) {
            bufferAddr = (GM_ADDR)epWinContext_->localWindowsIn;
        } else {
            bufferAddr = (GM_ADDR)((HcclRankRelationResV2 *)epWinContext_->remoteRes[rankId].nextDevicePtr)->windowsIn;
        }
        return (GM_ADDR)(bufferAddr + winDataSizeOffset_ + Moe::NOTIFY_DISPATCH_BUFF_OFFSET);
    }

    __aicore__ GM_ADDR GetBufferAddrByRankId(const int32_t rankId)
    {
        return GetStateAddrByRankId(rankId) + STATE_WIN_SIZE + roundMagic_ * combineDataBuffSize_;
    }

    __aicore__ inline GM_ADDR GetRoundStateAddrByRankId(const int32_t rankId)
    {
        if (epRankId_ == rankId) {
            return (GM_ADDR)(epWinContext_->localWindowsExp) + roundMagic_ * Moe::ROUND_STATE_MAX_SIZE +
                   ROUND_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))->windowsExp) +
               roundMagic_ * Moe::ROUND_STATE_MAX_SIZE + ROUND_STATE_OFFSET;
    }

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
    uint32_t magic_{0};
    uint32_t roundMagic_{0};
    uint64_t winDataSizeOffset_{0};
    uint32_t hRecvXTypeLen_{0};
    uint32_t h32AlignFloatLen_{0};
    uint32_t h256AlignFloatLen_{0};
    uint32_t h32AlignRecvXLen_{0};
    uint32_t h512AlignRecvXLen_{0};
    uint32_t tokenIdx32AlignLen_{0};
    uint32_t roundIndex_{0};
    uint32_t realMaxBs_{0};
    uint32_t perRoundTokens_{0};
    uint32_t maxRound_{0};
    // send用到的数据
    uint32_t sendCostStatsBufSize_{0};
    uint32_t needSendTokenCnt_{0};
    uint32_t RecvTokenNum_{0};
    uint32_t perCoreBlockNum_{0};  // 每个core需要负责的block数，一个block表示某个expert从某个rank接收的token
    uint32_t startBlockId_{0};
    uint32_t endBlockId_{0};
    uint32_t preRecvCount_{0};
    // recv用到的数据
    uint32_t totalNeedRecvTokenCnt_{0};   // 剩余需要接收的token数，初始化为axisBS_
    uint32_t roundTotalRecvTokenCnt_{0};  // 每一轮所有核需要接收的总token数
    uint32_t roundRecvTokenCnt_{0};       // 每一轮每个核接收的token数，每一轮接收开始前重新计算
    uint32_t roundRecvStartTokenIdx_{0};  // 每一轮每个核从HCCL buffer接收的token的起始index，每一轮接收开始前重新计算
    uint32_t roundRecvEndTokenIdx_{0};  // 每一轮每个核从HCCL buffer接收的token的结束index，每一轮接收开始前重新计算
    // 这一轮接收的token需要存放在xOut的偏移，即前面几轮接收的token数，每一轮每个核从topkWeightsGM_拷贝权重也需要
    uint32_t xOutTokenOffset_{0};
    uint32_t stateOffset_{0};
    uint32_t combineDataBuffSize_{0};

    bool isEnableDiagnose_{false};

    TPipe *tpipe_{nullptr};
    TQue<QuePosition::VECIN, 1> weightedSumQueue_;
    TQue<QuePosition::VECOUT, 1> sendCostStatsOutQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> localCopyQueue_;
    TBuf<> setStateBuf_;
    TBuf<> waitStateBuf_;
    TBuf<> waitTempStateBuf_;
    TBuf<> topkWeightsBuf_;
    TBuf<> tokenFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> weightedMulBuf_;
    TBuf<> srcInfoBuf_;
    TBuf<> tokenIdxBuf_;
    TBuf<> xOutBuf_;
    TBuf<> setRoundStateBuf_;
    TBuf<> waitRoundStateBuf_;
    TBuf<> tempRoundStateBuf_;
    TBuf<> roundNeedSendCntBuf_;
    TBuf<> roundSendOffsetBuf_;
    TBuf<> tempRecvCountBuf_;

    LocalTensor<uint32_t> setStateLT_;
    LocalTensor<uint32_t> roundNeedSendCntLT_;
    LocalTensor<uint32_t> roundSendOffsetLT_;
    LocalTensor<SrcInfoType> srcInfoLT_;
    LocalTensor<SrcInfoType> tokenIdxLT_;
    LocalTensor<float> topkWeightsLT_;

    GlobalTensor<RecvXType> recvXGM_;
    GlobalTensor<SrcInfoType> tokenSrcInfoGM_;
    GlobalTensor<SrcInfoType> epRecvCountGM_;
    GlobalTensor<SrcInfoType> tokenIdxGM_;
    GlobalTensor<float> topkWeightsGM_;
    GlobalTensor<XType> xOutGlobal_;
    GlobalTensor<int32_t> sendCostStatsGT_;
    GlobalTensor<float> dstRoundStatusGT_;
    GM_ADDR workspaceGM_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitMagic()
{
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam *)contextGM0;

    GlobalTensor<int32_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer(
        (__gm__ int32_t *)((GM_ADDR)epWinContext_->localWindowsExp + MAGIC_WIN_OFFSET + coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitGlobalBuffer(
    GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR tokenIdx, GM_ADDR XOut,
    GM_ADDR sendCostStatsOut)
{
    recvXGM_.SetGlobalBuffer((__gm__ RecvXType *)recvX);
    tokenSrcInfoGM_.SetGlobalBuffer((__gm__ SrcInfoType *)tokenSrcInfo);
    epRecvCountGM_.SetGlobalBuffer((__gm__ int32_t *)epRecvCount);
    tokenIdxGM_.SetGlobalBuffer((__gm__ int32_t *)tokenIdx);
    topkWeightsGM_.SetGlobalBuffer((__gm__ float *)topkWeights);
    xOutGlobal_.SetGlobalBuffer((__gm__ XType *)XOut);
    if (isEnableDiagnose_) {
        sendCostStatsGT_.SetGlobalBuffer((__gm__ int32_t *)sendCostStatsOut);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void
CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitTilingData(const CamMoeCombineNormalTilingData *tilingData)
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
    realMaxBs_ = tilingData->camMoeCombineNormalInfo.realMaxBs;
    maxRound_ = tilingData->camMoeCombineNormalInfo.maxRound;
    perRoundTokens_ = tilingData->camMoeCombineNormalInfo.perRoundTokens;
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitBuffLen()
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
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitRoundSendData()
{
    SplitCoreCal(moeExpertNum_, perCoreBlockNum_, startBlockId_,
                 endBlockId_);  // 按专家分核，每个核负责向perBlockRankNum个rank发送数据
    if (perCoreBlockNum_ == 0) {
        return;
    }
    uint32_t sendBlockLen = perCoreBlockNum_ * sizeof(int32_t);
    tpipe_->Reset();
    tpipe_->InitBuffer(tempRecvCountBuf_, sendBlockLen);     // 64B
    tpipe_->InitBuffer(roundNeedSendCntBuf_, sendBlockLen);  // 64B
    tpipe_->InitBuffer(roundSendOffsetBuf_, sendBlockLen);   // 64B

    // 拷贝 epRecvCountGM_ 到 UB
    LocalTensor<int32_t> tempRecvCountTensor = tempRecvCountBuf_.Get<int32_t>();
    const DataCopyExtParams sendBlockCopyParams{1U, sendBlockLen, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> sendBlockPadParams{false, 0U, 0U, 0U};
    DataCopyPad(tempRecvCountTensor, epRecvCountGM_[startBlockId_], sendBlockCopyParams, sendBlockPadParams);
    SyncFunc<HardEvent::MTE2_S>();

    // 每个核计算需要给每个专家发送的token数以及token起始偏移，以及每个block的srcInfo偏移
    preRecvCount_ = startBlockId_ == 0 ? 0 : epRecvCountGM_(startBlockId_ - 1);  // 记录当前core发送token的起始偏移
    needSendTokenCnt_ = tempRecvCountTensor(perCoreBlockNum_ - 1) - preRecvCount_;
    roundNeedSendCntLT_ = roundNeedSendCntBuf_.Get<uint32_t>();
    roundSendOffsetLT_ = roundSendOffsetBuf_.Get<uint32_t>();
    roundSendOffsetLT_(0) = preRecvCount_;
    roundNeedSendCntLT_(0) = tempRecvCountTensor(0) - preRecvCount_;
    for (uint32_t i = 1; i < perCoreBlockNum_; ++i) {
        roundSendOffsetLT_(i) = tempRecvCountTensor(i - 1);
        roundNeedSendCntLT_(i) = tempRecvCountTensor(i) - tempRecvCountTensor(i - 1);
    }

    // 创建 srcInfoLT_
    // 为了支持一轮最大8192 bs，这里按照一批BATCH_SRC_INFO_LEN个srcInfo拷贝，这样可以保证UB占用少
    uint32_t srcInfoLen = static_cast<uint32_t>(BATCH_SRC_INFO_CNT * TOKEN_SRC_INFO_LEN * sizeof(SrcInfoType));
    tpipe_->InitBuffer(srcInfoBuf_, srcInfoLen);  // 128*3*4/1024=1.5KB
    srcInfoLT_ = srcInfoBuf_.Get<SrcInfoType>();

    // 创建 setStatusLT_
    tpipe_->InitBuffer(setStateBuf_, UB_32_ALIGN);  // 32B
    setStateLT_ = setStateBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(setStateLT_, 0x3F800000, FLOAT_NUM_PER_ALIGN);

    // 创建localCopyQueue_， 用于存放从GM拷贝到UB的token
    tpipe_->InitBuffer(localCopyQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);  // 28KB
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::InitRoundRecvData()
{
    totalNeedRecvTokenCnt_ = axisBS_;

    // 每个核一轮最多需要接收Ceil(perRoundTokens_, aivNum_) * aivNum_个token，topkWeightBuf_也只需要开这么大
    tpipe_->InitBuffer(xOutBuf_, h32AlignRecvXLen_);                                             // 14KB
    tpipe_->InitBuffer(tokenFloatBuf_, h32AlignFloatLen_);                                       // 28KB
    tpipe_->InitBuffer(weightedMulBuf_, h256AlignFloatLen_);                                     // 28KB
    tpipe_->InitBuffer(sumFloatBuf_, h32AlignFloatLen_);                                         // 28KB
    tpipe_->InitBuffer(weightedSumQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);                     // 14KB
    tpipe_->InitBuffer(waitStateBuf_, axisK_ * UB_32_ALIGN);                                     // 196B
    tpipe_->InitBuffer(waitTempStateBuf_, axisK_ * UB_32_ALIGN);                                 // 196B
    tpipe_->InitBuffer(setRoundStateBuf_, epWorldSize_ * FLOAT_NUM_PER_ALIGN * sizeof(float));   // 用于setRoundStatus
    tpipe_->InitBuffer(waitRoundStateBuf_, epWorldSize_ * FLOAT_NUM_PER_ALIGN * sizeof(float));  // 用于waitRoundStatus
    tpipe_->InitBuffer(tempRoundStateBuf_, epWorldSize_ * FLOAT_NUM_PER_ALIGN * sizeof(float));  // 用于waitRoundStatus

    // 创建topkWeightsLT_，存放每一轮每个核的权重信息
    uint32_t maxTopkWeightsLen = (perRoundTokens_ / aivNum_ + 1) * axisK_ * sizeof(float);
    tokenIdx32AlignLen_ = (perRoundTokens_ / aivNum_ + 1) * axisK_ * sizeof(int32_t);
    tpipe_->InitBuffer(tokenIdxBuf_, tokenIdx32AlignLen_);
    tpipe_->InitBuffer(topkWeightsBuf_, maxTopkWeightsLen);  // 512 分48核 需要352B
    tokenIdxLT_ = tokenIdxBuf_.Get<int32_t>();
    topkWeightsLT_ = topkWeightsBuf_.Get<float>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::Init(
    GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR tokenIdx,
    GM_ADDR tpRecvCount, GM_ADDR XOut, GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const CamMoeCombineNormalTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    stateOffset_ = STATE_OFFSET;

    InitMagic();
    InitTilingData(tilingData);
    InitGlobalBuffer(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, XOut, sendCostStatsOut);
    InitBuffLen();
    combineDataBuffSize_ = perRoundTokens_ * axisK_ * h512AlignRecvXLen_;
    PipeBarrier<PIPE_ALL>();
    winDataSizeOffset_ = static_cast<uint64_t>(magic_) * (tilingData->camMoeCombineNormalInfo.totalWinSize / 2UL);
    DataCacheCleanAndInvalid<SrcInfoType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epRecvCountGM_[moeExpertNum_ - 1]);

    InitRoundSendData();
    InitRoundRecvData();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::CopyBufferToShareAndSetStatus()
{
    if (needSendTokenCnt_ == 0) {
        return;
    }
    LocalTensor<int32_t> sendCostStatsTensor;
    if (isEnableDiagnose_) {
        tpipe_->InitBuffer(sendCostStatsOutQueue_, DOUBLE_BUFFER, sendCostStatsBufSize_);
        sendCostStatsTensor = sendCostStatsOutQueue_.AllocTensor<int32_t>();
        Duplicate<int32_t>(sendCostStatsTensor, 0, sendCostStatsBufSize_ / sizeof(int32_t));
    }

    uint32_t startTokenIndex = preRecvCount_;
    int64_t sendStartCycle;
    for (uint32_t blockIndex = 0; blockIndex < perCoreBlockNum_; ++blockIndex) {
        uint32_t roundMaxSendCount = roundNeedSendCntLT_(blockIndex) >= perRoundTokens_
                                         ? perRoundTokens_
                                         : roundNeedSendCntLT_(blockIndex);  // 这一轮最多发送 roundMaxSendCount 个token

        uint32_t sendTokenOffset = roundSendOffsetLT_(blockIndex);
        uint32_t startTokenId = sendTokenOffset;  // 这一轮要发的token在recvX中的偏移
        uint32_t roundActualSendCount = 0;        // 这一轮blockIndex实际发送的token数，<= roundMaxSendCount
        while (roundActualSendCount < roundMaxSendCount) {
            uint32_t recvXTokenIdx = startTokenId + roundActualSendCount;  // 要发送的token在recvX中的位置
            uint32_t tokenIdxInBatch = roundActualSendCount % BATCH_SRC_INFO_CNT;
            if (tokenIdxInBatch == 0) {
                uint32_t tokenCount = min(BATCH_SRC_INFO_CNT, roundMaxSendCount - roundActualSendCount);
                uint32_t srcInfoLen = tokenCount * TOKEN_SRC_INFO_LEN * sizeof(SrcInfoType);
                const DataCopyExtParams dataCopyParams{1U, srcInfoLen, 0U, 0U, 0U};
                const DataCopyPadExtParams<SrcInfoType> padParams{false, 0U, 0U, 0U};
                DataCopyPad(srcInfoLT_, tokenSrcInfoGM_[recvXTokenIdx * TOKEN_SRC_INFO_LEN], dataCopyParams, padParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
            }
            // 要发送的token的srcInfo在srcInfoLT_中的位置，所以起始偏移为0
            uint32_t srcInfoIdx = tokenIdxInBatch * TOKEN_SRC_INFO_LEN;
            uint32_t srcRankId = static_cast<uint32_t>(srcInfoLT_(srcInfoIdx + RANK_ID_OFFSET_IN_SRC_INFO));
            uint32_t srcTokenId = static_cast<uint32_t>(srcInfoLT_(srcInfoIdx + TOKEN_IDX_OFFSET_IN_SRC_INFO));
            if (srcTokenId >= (roundIndex_ + 1) * perRoundTokens_) {
                // 这一轮实际发送的token数，接收方一轮最多接收perRoundTokens_个token
                break;
            }
            uint32_t srcTopkId = static_cast<uint32_t>(srcInfoLT_(srcInfoIdx + TOPK_IDX_OFFSET_IN_SRC_INFO));
            // 每一轮 put token和state 到目标rank的hccl buffer的偏移都要从0开始计算
            uint32_t roundTokenId = srcTokenId % perRoundTokens_;
            if (isEnableDiagnose_) {
                sendStartCycle = GetSystemCycle();
            }
            CopyBufferToShare(srcRankId, roundTokenId, srcTopkId, recvXTokenIdx);
            PipeBarrier<PIPE_ALL>();
            SetStatusBySrcInfo(srcRankId, roundTokenId, srcTopkId);

            if (isEnableDiagnose_) {
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                int32_t durationTime = static_cast<int32_t>((GetSystemCycle() - sendStartCycle) / CYCLE_TO_TIME);  // us
                int32_t preTime = sendCostStatsTensor.GetValue(srcRankId);
                sendCostStatsTensor.SetValue(srcRankId, preTime + durationTime);
            }
            ++roundActualSendCount;
        }

        roundSendOffsetLT_(blockIndex) += roundActualSendCount;
        roundNeedSendCntLT_(blockIndex) -= roundActualSendCount;
        needSendTokenCnt_ -= roundActualSendCount;
    }

    if (isEnableDiagnose_) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        AscendC::SetAtomicAdd<int32_t>();
        DataCopyExtParams statsCopyOutParams = {1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad<int32_t>(sendCostStatsGT_, sendCostStatsTensor, statsCopyOutParams);
        AscendC::SetAtomicNone();
        sendCostStatsOutQueue_.FreeTensor<int32_t>(sendCostStatsTensor);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::CopyBufferToShare(uint32_t srcRankId,
                                                                                             uint32_t srcTokenId,
                                                                                             uint32_t srcTopkId,
                                                                                             uint32_t tkIndex)
{
    uint32_t tokenOffset = tkIndex * axisH_;
    GM_ADDR dstGM = GetBufferAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * h512AlignRecvXLen_;
    GlobalTensor<XType> dstWindow;
    dstWindow.SetGlobalBuffer((__gm__ XType *)dstGM);
    DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
    DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};

    LocalTensor<RecvXType> localCopyTensor;
    localCopyTensor = localCopyQueue_.AllocTensor<RecvXType>();
    DataCopyPad(localCopyTensor, recvXGM_[tokenOffset], xOutCopyParams, copyPadExtParams);
    localCopyQueue_.EnQue(localCopyTensor);
    localCopyTensor = localCopyQueue_.DeQue<RecvXType>();
    DataCopyPad(dstWindow, localCopyTensor, xOutCopyParams);
    localCopyQueue_.FreeTensor<RecvXType>(localCopyTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::SetStatusBySrcInfo(uint32_t srcRankId,
                                                                                              uint32_t srcTokenId,
                                                                                              uint32_t srcTopkId)
{
    uint32_t stateOffset = roundMagic_ * STATE_WIN_SIZE_HALF;
    GM_ADDR stateGM = GetStateAddrByRankId(srcRankId) + stateOffset + (srcTokenId * axisK_ + srcTopkId) * UB_32_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
    DataCopy<uint32_t>(stateGMTensor, setStateLT_, FLOAT_NUM_PER_ALIGN);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::WaitBuffCopy(uint32_t recvXTokenIdx,
                                                                                        uint32_t topkWeightTokenIdx)
{
    int tempValidCount = 0;
    for (int topkId = 0; topkId < axisK_; ++topkId) {
        int expertId = tokenIdxLT_.GetValue((topkWeightTokenIdx)*axisK_ + topkId);
        if (expertId < 0 || expertId >= moeExpertNum_) {
            continue;
        }
        ++tempValidCount;
    }
    uint32_t calCount = axisK_ * FLOAT_NUM_PER_ALIGN;
    uint32_t stateOffset = roundMagic_ * STATE_WIN_SIZE_HALF;
    GM_ADDR stateGM =
        GetStateAddrByRankId(epRankId_) + stateOffset + recvXTokenIdx * axisK_ * UB_32_ALIGN;  // 计算地址偏移
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float *)stateGM);
    float current = (float)0.0;
    float target = (float)1.0 * tempValidCount * FLOAT_NUM_PER_ALIGN;
    SumParams sumPerKParams{1, calCount, calCount};
    LocalTensor<float> stateTensorLocal = waitStateBuf_.Get<float>();
    LocalTensor<float> tempStateTensorLocal = waitTempStateBuf_.Get<float>();
    while (current != target) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensorLocal, stateGMTensor, calCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(tempStateTensorLocal, stateTensorLocal, sumPerKParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        current = tempStateTensorLocal(0);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(tempStateTensorLocal, (float)0.0, calCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(stateGMTensor, tempStateTensorLocal, calCount);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::ReadBufferAndWeightedSum(
    uint32_t recvXTokenIdx, uint32_t topkWeightTokenIdx)
{
    LocalTensor<float> tokenFloatLocal = tokenFloatBuf_.Get<float>();
    LocalTensor<float> weightedMulBufLocal = weightedMulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal = sumFloatBuf_.Get<float>();
    Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);
    const DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
    uint32_t xOutTokenIdx = recvXTokenIdx + xOutTokenOffset_;

    for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
        int expertId = tokenIdxLT_.GetValue(topkWeightTokenIdx * axisK_ + topkId);
        if (expertId < 0 || expertId >= moeExpertNum_) {
            continue;
        }
        float scale = topkWeightsLT_.GetValue(topkWeightTokenIdx * axisK_ + topkId);
        GM_ADDR localTokenAddr =
            GetBufferAddrByRankId(epRankId_) + (recvXTokenIdx * axisK_ + topkId) * h512AlignRecvXLen_;
        GlobalTensor<XType> localTokenTensor;
        localTokenTensor.SetGlobalBuffer((__gm__ XType *)localTokenAddr);

        LocalTensor<XType> tmpToken = weightedSumQueue_.AllocTensor<XType>();
        const DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpToken, localTokenTensor, xOutCopyParams, copyPadExtParams);
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
    DataCopyPad(xOutGlobal_[xOutTokenIdx * axisH_], xOutLocal, xOutCopyParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::ReadBufferFromRemote()
{
    if (totalNeedRecvTokenCnt_ == 0) {
        return;
    }
    roundTotalRecvTokenCnt_ = min(perRoundTokens_, totalNeedRecvTokenCnt_);
    SplitCoreCal(roundTotalRecvTokenCnt_, roundRecvTokenCnt_, roundRecvStartTokenIdx_, roundRecvEndTokenIdx_);
    if (roundRecvTokenCnt_ == 0) {
        return;
    }
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(roundRecvTokenCnt_ * axisK_ * sizeof(float)), 0U, 0U,
                                      0U};
    const DataCopyExtParams tokenIdxParams{1U, static_cast<uint32_t>(roundRecvTokenCnt_ * axisK_ * sizeof(int32_t)), 0U,
                                           0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> copyPadIntParams{false, 0U, 0U, 0U};
    DataCopyPad(topkWeightsLT_, topkWeightsGM_[(xOutTokenOffset_ + roundRecvStartTokenIdx_) * axisK_], bskParams,
                copyPadFloatParams);
    DataCopyPad(tokenIdxLT_, tokenIdxGM_[(xOutTokenOffset_ + roundRecvStartTokenIdx_) * axisK_], tokenIdxParams,
                copyPadIntParams);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t roundTokenIdx = roundRecvStartTokenIdx_; roundTokenIdx < roundRecvEndTokenIdx_;
         roundTokenIdx++) {  // 每轮都从从hccl buffer起始位置读put来的数据
        uint32_t topkWeightIdx = roundTokenIdx - roundRecvStartTokenIdx_;  // 用来计算每一轮token对应weight的偏移
        WaitBuffCopy(roundTokenIdx, topkWeightIdx);
        SyncFunc<AscendC::HardEvent::MTE3_V>();  // 与结果搬出datacopy同tensor
        ReadBufferAndWeightedSum(roundTokenIdx, topkWeightIdx);
    }
    totalNeedRecvTokenCnt_ -= roundTotalRecvTokenCnt_;
    xOutTokenOffset_ += roundTotalRecvTokenCnt_;
}
template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::SetRoundStatus()
{
    if (coreIdx_ != 0) {
        return;
    }
    LocalTensor<float> roundStateTensor = setRoundStateBuf_.Get<float>();
    Duplicate<float>(roundStateTensor, 1.0, FLOAT_NUM_PER_ALIGN);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        uint32_t targetRankId = i;
        uint32_t offset = stateOffset_ * epRankId_;
        GM_ADDR rankGM = GetRoundStateAddrByRankId(targetRankId) + offset;
        dstRoundStatusGT_.SetGlobalBuffer((__gm__ float *)rankGM);
        DataCopy<float>(dstRoundStatusGT_, roundStateTensor, FLOAT_NUM_PER_ALIGN);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::WaitRoundStatus()
{
    if (coreIdx_ != 0) {
        return;
    }
    uint32_t count = epWorldSize_ * FLOAT_NUM_PER_ALIGN;
    uint32_t inner = (count * sizeof(float) + 32 - 1) / 32 * 32 / sizeof(float);
    GM_ADDR roundStateGM = GetRoundStateAddrByRankId(epRankId_);
    GlobalTensor<float> roundStatusGMTensor;

    roundStatusGMTensor.SetGlobalBuffer((__gm__ float *)roundStateGM);
    float current = (float)0.0;
    float target = (float)(1.0) * epWorldSize_ * FLOAT_NUM_PER_ALIGN;
    SumParams sumPerRankParams{1, inner, count};
    LocalTensor<float> stateTensorLocal = waitRoundStateBuf_.Get<float>();
    LocalTensor<float> tempRoundStateTensorLocal = tempRoundStateBuf_.Get<float>();

    while (current != target) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensorLocal, roundStatusGMTensor, count);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(tempRoundStateTensorLocal, stateTensorLocal, sumPerRankParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        current = tempRoundStateTensorLocal.GetValue(0);
    }

    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(tempRoundStateTensorLocal, (float)0.0, count);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(roundStatusGMTensor, tempRoundStateTensorLocal, count);
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeCombineNormalMultiRound<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        uint32_t realRound = (realMaxBs_ + perRoundTokens_ - 1) / perRoundTokens_;
        while (roundIndex_ < realRound) {
            CopyBufferToShareAndSetStatus();
            ReadBufferFromRemote();
            if (realRound > 1) {
                SetRoundStatus();
                WaitRoundStatus();
                roundMagic_ = roundMagic_ == 0 ? 1 : 0;
                SyncAll<true>();
            }
            roundIndex_ += 1;
        }
    }
}

}  // namespace CamMoeCombineNormalMultiRoundImpl
#endif  // MOE_COMBINE_IMPL_H
