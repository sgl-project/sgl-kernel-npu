#ifndef SHMEM_MOE_COMBINE_NORMAL_H
#define SHMEM_MOE_COMBINE_NORMAL_H

#include "shmem_api.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem_moe_combine_normal_tiling.h"

namespace ShmemMoeCombineNormalImpl {
constexpr uint64_t NOTIFY_MAGIC_OFFSET = 50UL * 1024UL;
constexpr uint64_t WIN_MAGIC_OFFSET = 100UL * 1024UL;  // notify(50kb) + dispatch&combine(50kb)
constexpr uint64_t HALF_WIN_STATE_OFFSET = 8 * 1024UL * 1024UL;  // notify(2MB) + dispatch(3MB) + combine(3MB)
constexpr uint64_t COMBINE_WIN_STATE_OFFSET = 5 * 1024UL * 1024UL;  // notify+dispatch(5MB)
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;

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
class ShmemMoeCombineNormal
{
public:
    __aicore__ inline ShmemMoeCombineNormal(){};
    __aicore__ inline void Init(GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                GM_ADDR topkIdx, GM_ADDR sendTokenIdx, GM_ADDR XOut,
                                GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const ShmemMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMagic();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR recvX, GM_ADDR epRecvCount,
                                            GM_ADDR topkWeights, GM_ADDR topkIdx, GM_ADDR sendTokenIdx, GM_ADDR XOut,
                                            GM_ADDR sendCostStatsOut);
    __aicore__ inline void InitTilingData(const ShmemMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void InitBuffLen();
    __aicore__ inline void CopyBufferToShareAndSetStatus();
    __aicore__ inline void CopyBufferToShare(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId,
                                             uint32_t tkIndex);
    __aicore__ inline void ReadBufferFromRemote();
    __aicore__ inline void WaitBuffCopy(uint32_t tokenIndex);
    __aicore__ inline void SetStatusBySrcInfo(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId);
    __aicore__ inline void ReadBufferAndWeightedSum(uint32_t tokenIndex, uint32_t startTokenIndex);
    __aicore__ inline void AllGatherRecvCount();

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
    uint32_t k32AlignFloatLen_{0};
    uint32_t k32AlignLen_{0};

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

    GlobalTensor<RecvXType> dstGT;
    GlobalTensor<RecvXType> recvXGT_;
    GlobalTensor<SrcInfoType> epRecvCountGT_;
    GlobalTensor<float> topkWeightsGT_;
    GlobalTensor<int32_t> sendTokenIdxGT_;
    GlobalTensor<int32_t> topkIdxGT_;
    GlobalTensor<XType> xOutGlobal_;
    GlobalTensor<int32_t> sendCostStatsGT_;
    GlobalTensor<uint64_t> xOutAddrGT_;

    GM_ADDR recvXGM_;
    GM_ADDR localRankGM_;
    GM_ADDR XOutGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR metaDataGvaGM_;
    GM_ADDR metaStateGvaGM_;
    GM_ADDR dataStateGvaGM_;

    LocalTensor<float> tokenFloatLocal;
    LocalTensor<float> weightedMulBufLocal;
    LocalTensor<float> sumFloatBufLocal;
    LocalTensor<float> topkWeightsLocal;
    LocalTensor<int32_t> sendTokenIdxLocal;
    LocalTensor<uint32_t> stateTensorLocal;
    LocalTensor<int32_t> allRecvCountLocal;
    LocalTensor<int32_t> topkIdxLocal;
};

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitMagic()
{
    GM_ADDR magicAddr = (GM_ADDR)(metaDataGvaGM_ + NOTIFY_MAGIC_OFFSET);
    GlobalTensor<uint64_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer((__gm__ uint64_t *)(magicAddr + coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitGlobalBuffer(GM_ADDR recvX,
                                                                                  GM_ADDR epRecvCount,
                                                                                  GM_ADDR topkWeights, GM_ADDR topkIdx,
                                                                                  GM_ADDR sendTokenIdx, GM_ADDR XOut,
                                                                                  GM_ADDR sendCostStatsOut)
{
    recvXGT_.SetGlobalBuffer((__gm__ RecvXType *)recvX);
    epRecvCountGT_.SetGlobalBuffer((__gm__ int32_t *)epRecvCount);  // 放置allReccvCount信息，num_ranks * num_experts
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
ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitTilingData(const ShmemMoeCombineNormalTilingData *tilingData)
{
    axisBS_ = tilingData->moeCombineNormalInfo.bs;
    axisH_ = tilingData->moeCombineNormalInfo.h;
    axisK_ = tilingData->moeCombineNormalInfo.k;
    aivNum_ = tilingData->moeCombineNormalInfo.aivNum;
    moeExpertNum_ = tilingData->moeCombineNormalInfo.moeExpertNum;
    moeExpertPerRankNum_ = tilingData->moeCombineNormalInfo.moeExpertPerRankNum;
    epWorldSize_ = tilingData->moeCombineNormalInfo.epWorldSize;
    epRankId_ = tilingData->moeCombineNormalInfo.epRankId;
    isEnableDiagnose_ = tilingData->moeCombineNormalInfo.isEnableDiagnose;
    metaDataGvaGM_ = (GM_ADDR)tilingData->shmemPtr;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitBuffLen()
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
    k32AlignFloatLen_ = Ceil(axisK_ * static_cast<uint32_t>(sizeof(float)), UB_32_ALIGN) * UB_32_ALIGN;
    k32AlignLen_ = Ceil(axisK_ * static_cast<uint32_t>(sizeof(int32_t)), UB_32_ALIGN) * UB_32_ALIGN;
    // h32AlignFloatLen_:28672, h256AlignFloatLen_:28672, hRecvXTypeLen_:14336, h32AlignRecvXLen_:14336, 
    // h512AlignRecvXLen_:14336 k32AlignFloatLen_:32, k32AlignLen_:32
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::Init(
    GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR topkIdx,
    GM_ADDR sendTokenIdx, GM_ADDR XOut, GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const ShmemMoeCombineNormalTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    recvXGM_ = recvX;
    XOutGM_ = XOut;
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();

    InitTilingData(tilingData);
    InitGlobalBuffer(recvX, epRecvCount, topkWeights, topkIdx, sendTokenIdx, XOut, sendCostStatsOut);
    InitBuffLen();

    InitMagic();
    PipeBarrier<PIPE_ALL>();
    metaStateWinOffset_ = NOTIFY_MAGIC_OFFSET;
    dataStateWinOffset_ = metaStateWinOffset_ + magic_ * COMBINE_WIN_STATE_OFFSET;
    metaStateGvaGM_ = (GM_ADDR)(metaDataGvaGM_ + metaStateWinOffset_);
    dataStateGvaGM_ = (GM_ADDR)(metaDataGvaGM_ + dataStateWinOffset_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferAndWeightedSum(uint32_t tokenIndex,
                                                                                          uint32_t startTokenIndex)
{
    const DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
    const DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};
    Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);

    for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
        float scale = topkWeightsLocal.GetValue(topkId);
        int32_t expertId = topkIdxLocal.GetValue(topkId);
        int32_t remoteReadOffset = sendTokenIdxLocal(topkId);
        int32_t remoteReadBase = allRecvCountLocal(expertId * epWorldSize_ + epRankId_);
        uint64_t remoteReadAddr = static_cast<uint64_t>(remoteReadBase + remoteReadOffset) * hRecvXTypeLen_;

        int32_t dstRankId = expertId / moeExpertPerRankNum_;
        auto ptr = reinterpret_cast<__gm__ uint8_t *>(shmem_ptr(recvXGM_,  dstRankId));
        dstGT.SetGlobalBuffer((__gm__ XType *)(ptr + hRecvXTypeLen_ * (remoteReadBase + remoteReadOffset)));

        LocalTensor<XType> tmpToken = weightedSumQueue_.AllocTensor<XType>();
        DataCopyPad(tmpToken, dstGT, xOutCopyParams, copyPadExtParams);
        weightedSumQueue_.EnQue(tmpToken);
        tmpToken = weightedSumQueue_.DeQue<XType>();
        Cast(tokenFloatLocal, tmpToken, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(weightedMulBufLocal, tokenFloatLocal, scale, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, weightedMulBufLocal, axisH_);
        weightedSumQueue_.FreeTensor<XType>(tmpToken);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> xOutLocal = xOutBuf_.Get<XType>();
    Cast(xOutLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(xOutGlobal_[tokenIndex * axisH_], xOutLocal, xOutCopyParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferFromRemote()
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
    tpipe_->InitBuffer(tokenFloatBuf_, h32AlignFloatLen_);                    // 28KB
    tpipe_->InitBuffer(weightedMulBuf_, h256AlignFloatLen_);                  // 28KB
    tpipe_->InitBuffer(sumFloatBuf_, h32AlignFloatLen_);                      // 28KB
    tpipe_->InitBuffer(weightedSumQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);  // 2 * 14KB = 28KB
    tpipe_->InitBuffer(topkWeightsBuf_, k32AlignFloatLen_);                   // 32b
    tpipe_->InitBuffer(sendTokenIdxBuf_, k32AlignLen_);                       // 32b
    tpipe_->InitBuffer(topkIdxBuf_, k32AlignLen_);                            // 32b
    // moeExpertNum最大为512，tensor大小为 64*512*4=128kb
    uint32_t recvCountAlignLen_ = Ceil(epWorldSize_ * moeExpertNum_ * sizeof(int32_t), UB_32_ALIGN) * UB_32_ALIGN;
    tpipe_->InitBuffer(allRecvCountBuf_, recvCountAlignLen_);

    topkWeightsLocal = topkWeightsBuf_.Get<float>();
    tokenFloatLocal = tokenFloatBuf_.Get<float>();
    weightedMulBufLocal = weightedMulBuf_.Get<float>();
    sumFloatBufLocal = sumFloatBuf_.Get<float>();
    sendTokenIdxLocal = sendTokenIdxBuf_.Get<int32_t>();
    allRecvCountLocal = allRecvCountBuf_.Get<int32_t>();
    topkIdxLocal = topkIdxBuf_.Get<int32_t>();

    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(axisK_ * sizeof(float)), 0U, 0U, 0U};
    const DataCopyExtParams bskParams1{1U, static_cast<uint32_t>(axisK_ * sizeof(int32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> copyPadint32Params{false, 0U, 0U, 0U};

    const DataCopyExtParams countParams{1U, static_cast<uint32_t>(epWorldSize_ * moeExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};

    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPad(allRecvCountLocal, epRecvCountGT_, countParams, copyPadint32Params);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = startTokenIndex; tokenIndex < endTokenIndex; tokenIndex++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopyPad(topkWeightsLocal, topkWeightsGT_[tokenIndex * axisK_], bskParams, copyPadFloatParams);
        DataCopyPad(topkIdxLocal, topkIdxGT_[tokenIndex * axisK_], bskParams1, copyPadint32Params);
        DataCopyPad(sendTokenIdxLocal, sendTokenIdxGT_[tokenIndex * axisK_], bskParams1, copyPadint32Params);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        ReadBufferAndWeightedSum(tokenIndex, startTokenIndex);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        ReadBufferFromRemote();
        SyncAll<true>();
        shmem_barrier_all();  // 全卡同步，确保数据已经获取完
    }
}

}  // namespace ShmemMoeCombineNormalImpl
#endif  // MOE_COMBINE_IMPL_H
