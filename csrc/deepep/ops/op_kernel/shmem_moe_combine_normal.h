#ifndef SHMEM_MOE_COMBINE_NORMAL_H
#define SHMEM_MOE_COMBINE_NORMAL_H

#include "shmem_api.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shmem_moe_combine_normal_tiling.h"
#include "shmem_comm_args.h"

namespace ShmemMoeCombineNormalImpl {
constexpr uint64_t COMBINE_STATUS_OFFSET = 20UL * 1024UL;

constexpr uint32_t MUL_256_ALIGN = 256U;
constexpr uint64_t WIN_512_ALIGN = 512UL;
constexpr uint32_t FLOAT_NUM_PER_ALIGN = 8U;
constexpr uint8_t DOUBLE_BUFFER = 2;

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
using namespace ShmemMoe;
template <TemplateMC2TypeClass>
class ShmemMoeCombineNormal
{
public:
    __aicore__ inline ShmemMoeCombineNormal(){};
    __aicore__ inline void Init(GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR topkIdx,
                                GM_ADDR sendTokenIdx, GM_ADDR XOut, GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM,
                                TPipe *pipe, const ShmemMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalBuffer(GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR topkIdx,
                                            GM_ADDR sendTokenIdx, GM_ADDR XOut, GM_ADDR sendCostStatsOut);
    __aicore__ inline void InitTilingData(const ShmemMoeCombineNormalTilingData *tilingData);
    __aicore__ inline void InitBuffLen();
    __aicore__ inline void ResetMetaFlag();
    __aicore__ inline void PutShareAddr();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitStatus();
    __aicore__ inline void GetShareAddr();
    __aicore__ inline void ReadTokenFromRemote();
    __aicore__ inline void ReadTokenAndWeightedSum(uint32_t tokenIndex, uint32_t startTokenIndex);

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        auto ptr = shmem_ptr(gva_gm, rankId);
        return (GM_ADDR)(ptr);
    }

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx)
    {
        perCoreNum = totalNum / aivNum_;
        uint32_t remainderRankNum = totalNum % aivNum_;

        startIdx = perCoreNum * blockIdx;
        if (blockIdx < remainderRankNum) {
            perCoreNum++;
            startIdx += blockIdx;
        } else {
            startIdx += remainderRankNum;
        }
        endIdx = startIdx + perCoreNum;
    }

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epRankSize{0};
    uint32_t epRankId{0};
    uint32_t blockIdx{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint64_t magic_{0};
    uint32_t hRecvXTypeLen_{0};
    uint32_t h32AlignFloatLen_{0};
    uint32_t h256AlignFloatLen_{0};
    uint32_t h32AlignRecvXLen_{0};
    uint32_t h512AlignRecvXLen_{0};
    uint32_t sendCostStatsBufSize_{0};
    uint32_t k32AlignFloatLen_{0};
    uint32_t k32AlignLen_{0};
    uint32_t addrUint64AlignLen_{0};

    bool isEnableDiagnose_{false};

    uint32_t rankNumPerBlock;
    uint32_t curBlockStartRankId;
    uint32_t curBlockEndRankId;

    TPipe *tpipe_{nullptr};
    TQue<QuePosition::VECIN, 1> weightedSumQueue_;
    TQue<QuePosition::VECOUT, 1> sendCostStatsOutQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> localCopyQueue_;
    TBuf<> addrBuf;
    TBuf<> statusBuf;
    TBuf<> waitStatusBuf;
    TBuf<> gatherMaskOutBuf;
    TBuf<> statusSumBuf;

    TBuf<> topkWeightsBuf_;
    TBuf<> sendTokenIdxBuf_;
    TBuf<> tokenFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> weightedMulBuf_;
    TBuf<> xOutBuf_;
    TBuf<> allRecvCountBuf_;
    TBuf<> topkIdxBuf_;

    GlobalTensor<RecvXType> dstGT;
    GlobalTensor<RecvXType> recvXGT_;
    GlobalTensor<SrcInfoType> epRecvCountGT_;
    GlobalTensor<float> topkWeightsGT_;
    GlobalTensor<int32_t> sendTokenIdxGT_;
    GlobalTensor<int32_t> topkIdxGT_;
    GlobalTensor<XType> xOutGlobal_;
    GlobalTensor<int32_t> sendCostStatsGT_;

    GM_ADDR recvXGM_;
    GM_ADDR localRankGM_;
    GM_ADDR XOutGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR metaStateGvaGM_;
    GM_ADDR dataStateGvaGM_;

    GM_ADDR gva_gm;
    uint64_t shareRecvXAddrs[CAM_MAX_RANK_SIZE];  // List of shmem asymmetric output addresses (recvXGM_)
    uint32_t shareAddrNum{1};

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
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitGlobalBuffer(GM_ADDR recvX, GM_ADDR epRecvCount,
                                                                                    GM_ADDR topkWeights,
                                                                                    GM_ADDR topkIdx,
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
    epRankSize = tilingData->moeCombineNormalInfo.epWorldSize;
    epRankId = tilingData->moeCombineNormalInfo.epRankId;
    isEnableDiagnose_ = tilingData->moeCombineNormalInfo.isEnableDiagnose;
    gva_gm = (GM_ADDR)(tilingData->shmemPtr);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::InitBuffLen()
{
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    h32AlignFloatLen_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    h256AlignFloatLen_ = Ceil(hFloatSize, MUL_256_ALIGN) * MUL_256_ALIGN;
    hRecvXTypeLen_ = axisH_ * sizeof(RecvXType);
    h32AlignRecvXLen_ = Ceil(hRecvXTypeLen_, UB_ALIGN) * UB_ALIGN;
    h512AlignRecvXLen_ = Ceil(hRecvXTypeLen_, WIN_512_ALIGN) * WIN_512_ALIGN;
    if (isEnableDiagnose_) {
        sendCostStatsBufSize_ = Ceil(epRankSize * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    }
    k32AlignFloatLen_ = Ceil(axisK_ * static_cast<uint32_t>(sizeof(float)), UB_ALIGN) * UB_ALIGN;
    k32AlignLen_ = Ceil(axisK_ * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;

    addrUint64AlignLen_ = Ceil(shareAddrNum * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(addrBuf, addrUint64AlignLen_);
    // h32AlignFloatLen_:28672, h256AlignFloatLen_:28672, hRecvXTypeLen_:14336, h32AlignRecvXLen_:14336,
    // h512AlignRecvXLen_:14336 k32AlignFloatLen_:32, k32AlignLen_:32
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::Init(
    GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR topkIdx, GM_ADDR sendTokenIdx, GM_ADDR XOut,
    GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM, TPipe *pipe, const ShmemMoeCombineNormalTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    recvXGM_ = recvX;
    XOutGM_ = XOut;
    tpipe_ = pipe;
    blockIdx = GetBlockIdx();

    InitTilingData(tilingData);
    InitGlobalBuffer(recvX, epRecvCount, topkWeights, topkIdx, sendTokenIdx, XOut, sendCostStatsOut);
    InitBuffLen();

    // if (blockIdx == 0) {
    //     printf("[combine_Init] rank:%d, blockId:%d, gva_gm:%p\n", epRankId, blockIdx, gva_gm);
    // }

    // rank分核
    SplitCoreCal(epRankSize, rankNumPerBlock, curBlockStartRankId, curBlockEndRankId);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::ResetMetaFlag()
{
    if (rankNumPerBlock == 0U) {
        return;
    }

    uint32_t waitStatusBufSize = (((rankNumPerBlock * UB_ALIGN) > 256) ? (rankNumPerBlock * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf, waitStatusBufSize);  // ranks/48 * 32B = 1 * 32B

    GlobalTensor<float> statusFp32TensorGT;
    statusFp32TensorGT.SetGlobalBuffer((__gm__ float *)(GetWindStateAddrByRankId(epRankId)));

    DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusFp32TensorGT[curBlockStartRankId * STATE_OFFSET / sizeof(float)],
            cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // printf("[ResetMetaFlag] rank:%d, blockId:%d\n", epRankId, blockIdx);
    // AscendC::DumpTensor(statusFp32TensorGT, 253, 40);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::PutShareAddr()
{
    // 一个核将地址写入本rank的meta
    if (blockIdx != 0) {
        return;
    }

    LocalTensor<uint64_t> addrTensor_ = addrBuf.Get<uint64_t>();
    uint64_t recvXAddr = reinterpret_cast<__gm__ uint64_t>(recvXGM_);
    addrTensor_(0) = recvXAddr;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

    // if (blockIdx == 0) {
    //     printf("[PutShareAddr] rank:%d, blockId:%d, expandXOutAddr:%p, dynamicScalesOutAddr:%p\n", 
    //         epRankId, blockIdx, expandXOutAddr, dynamicScalesOutAddr);
    //     AscendC::DumpTensor(addrTensor_, 272, 4);
    // }

    AscendC::GlobalTensor<uint64_t> metaDataGt;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(shareAddrNum * sizeof(uint64_t)), 0, 0, 0};
    GM_ADDR remote_meta = GetWindStateAddrByRankId(epRankId);
    metaDataGt.SetGlobalBuffer((__gm__ uint64_t *)(remote_meta) + COMBINE_STATUS_OFFSET);
    DataCopyPad(metaDataGt, addrTensor_, copyParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::SetStatus()
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
        GM_ADDR remote_meta = (__gm__ uint8_t *)(GetWindStateAddrByRankId(i) + epRankId * STATE_OFFSET);
        gmRemoteStatusGt.SetGlobalBuffer((__gm__ int32_t *)remote_meta);
        DataCopy<int32_t>(gmRemoteStatusGt, statusTensor[(i - curBlockStartRankId) * 8], 8UL);

        // if (epRankId_ == 0) {
        //     printf("[SetStatus] rank:%d, blockId:%d, i:%d, offset:%d\n", epRankId_, blockIdx_, i, epRankId_ * STATE_OFFSET);
        //     AscendC::DumpTensor(statusTensor[(i - curBlockStartRankId) * 8], 470, 8);
        // }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::WaitStatus()
{
    if (rankNumPerBlock == 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t waitStatusBufSize = (((rankNumPerBlock * UB_ALIGN) > 256) ? (rankNumPerBlock * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf, waitStatusBufSize);  // ranks/48 * 32B = 1 * 32B
    uint32_t maskAlign = Ceil(epRankSize * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf, maskAlign);  // rankSize * 4B
    tpipe_->InitBuffer(statusSumBuf, UB_ALIGN);  // 32B
    
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf.Get<float>();
    LocalTensor<float> statusSumOutTensor = statusSumBuf.Get<float>(UB_ALIGN);
    LocalTensor<float> statusFp32Tensor = waitStatusBuf.Get<float>();
    GlobalTensor<float> statusFp32TensorGT;
    statusFp32TensorGT.SetGlobalBuffer((__gm__ float *)(GetWindStateAddrByRankId(epRankId)));
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

        // if (epRankId_ == 0 && sumOfFlag == compareTarget) {
        //     printf("[WaitStatus] rank:%d, blockId:%d, sumOfFlag:%f, offset:%d\n", epRankId_, blockIdx_, sumOfFlag, curBlockStartRankId * STATE_OFFSET / sizeof(float));
        //     AscendC::DumpTensor(statusFp32Tensor, 511, 8);
        //     AscendC::DumpTensor(statusFp32TensorGT, 512, 40);
        // }
    }

    // // 清状态
    // SyncFunc<AscendC::HardEvent::MTE3_S>();
    // DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock), 1, 0, 0};
    // uint64_t duplicateMask[2] = {0x101010101010101, 0};
    // LocalTensor<int32_t> cleanStateTensor = waitStatusBuf.Get<int32_t>();
    // SyncFunc<AscendC::HardEvent::S_V>();
    // Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock, 8), 1, 8);
    // SyncFunc<AscendC::HardEvent::V_MTE3>();
    // DataCopy(statusFp32TensorGT[curBlockStartRankId * STATE_OFFSET / sizeof(float)],
    //         cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    // SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::GetShareAddr()
{
    LocalTensor<uint64_t> addrTensor_ = addrBuf.Get<uint64_t>();
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(addrUint64AlignLen_), 0, 0, 0};
    DataCopyPadExtParams<uint64_t> copyExtParams{false, 0U, 0U, 0U};

    // 从远端获取共享地址
    for (uint32_t i = 0; i < epRankSize; i++) {
        GM_ADDR remote_meta = GetWindStateAddrByRankId(i);
        AscendC::GlobalTensor<uint64_t> shareAddrGt;
        shareAddrGt.SetGlobalBuffer((__gm__ uint64_t *)(remote_meta) + COMBINE_STATUS_OFFSET);

        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopyPad(addrTensor_, shareAddrGt, copyParams, copyExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        shareRecvXAddrs[i] = addrTensor_(0);

        // if (epRankId == 0) {
        //     printf("[combine_GetShareAddr] rank:%d, blockId:%d, i:%d, recvX:%p\n", 
        //         epRankId, blockIdx, i, shareRecvXAddrs[i]);
        //     // AscendC::DumpTensor(addrTensor_, 387, 4);
        // }
    }

    // if (epRankId == 0) {
    //     printf("[combine_GetShareAddr1] rank:%d, blockId:%d, shareRecvXAddrs:%p %p %p %p, %p %p %p %p\n", 
    //         epRankId, blockIdx, 
    //         shareRecvXAddrs[0], shareRecvXAddrs[1], shareRecvXAddrs[2], shareRecvXAddrs[3],
    //         shareRecvXAddrs[4], shareRecvXAddrs[5], shareRecvXAddrs[6], shareRecvXAddrs[7]);
    // }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::ReadTokenAndWeightedSum(uint32_t tokenIndex,
                                                                                            uint32_t startTokenIndex)
{
    const DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
    const DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};
    Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);

    for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
        float scale = topkWeightsLocal.GetValue(topkId);
        int32_t expertId = topkIdxLocal.GetValue(topkId);
        int32_t remoteReadOffset = sendTokenIdxLocal(topkId);
        int32_t remoteReadBase = allRecvCountLocal(expertId * epRankSize + epRankId);
        uint64_t remoteReadAddr = static_cast<uint64_t>(remoteReadBase + remoteReadOffset) * hRecvXTypeLen_;

        int32_t dstRankId = expertId / moeExpertPerRankNum_;
        auto ptr = shareRecvXAddrs[dstRankId];
        // if (epRankId == 0) {
        //     printf("[ReadToken] rank:%d, blockId:%d, tokenIndex:%d, dstRankId:%d, ptr:%p, ptr2:%p, ori_ptr:%p\n", 
        //         epRankId, blockIdx, tokenIndex, dstRankId, ptr, ptr2, shareRecvXAddrs[dstRankId]);
        // }
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
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::ReadTokenFromRemote()
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
    uint32_t recvCountAlignLen_ = Ceil(epRankSize * moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
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

    const DataCopyExtParams countParams{1U, static_cast<uint32_t>(epRankSize * moeExpertNum_ * sizeof(int32_t)), 0U,
                                        0U, 0U};

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

        ReadTokenAndWeightedSum(tokenIndex, startTokenIndex);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ShmemMoeCombineNormal<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        ResetMetaFlag();
        
        // 交换所有卡的output地址
        PutShareAddr();
        shmem_barrier_all(); // 清除其他算子残留的flag TODO: 验证是否可以删除
        GetShareAddr();

        ReadTokenFromRemote();
        // SyncAll<true>();
        // shmem_barrier_all();  // 全卡同步，确保数据已经获取完
        SetStatus();
        WaitStatus();
    }
}

}  // namespace ShmemMoeCombineNormalImpl
#endif  // MOE_COMBINE_IMPL_H
