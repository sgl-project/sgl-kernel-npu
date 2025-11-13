#ifndef MOE_DISTRIBUTE_DISPATCH_V2_SINGLE_H
#define MOE_DISTRIBUTE_DISPATCH_V2_SINGLE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_base.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "check_winsize.h"

namespace MoeDistributeDispatchA2Impl {

#define TemplateMC2TypeA2SingleClass                                                                       \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsNeedAllgater, bool IsShareExpertRank
#define TemplateMC2TypeA2SingleFunc \
    XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgater, IsShareExpertRank

// #define ENABLE_PRINT
#ifdef ENABLE_PRINT
#define CAM_PRINT(fmt, ...)                  \
    do {                                     \
        AscendC::printf(fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define CAM_PRINT(fmt, ...)
#endif

using namespace AscendC;
template <TemplateMC2TypeA2SingleClass>
class MoeDistributeDispatchV2Single
{
public:
    constexpr static uint8_t BUFFER_NUM = 2;             // 多buf
    constexpr static uint32_t STATE_OFFSET = 32U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024 * 1024;  // 1M
    constexpr static uint32_t UB_ALIGN = 32U;            // UB按32字节对齐
    constexpr static uint8_t COMM_NUM = 2;               // 通信域大小
    constexpr static uint8_t COMM_EP_IDX = 0;
    constexpr static uint8_t COMM_TP_IDX = 1;
    // 先写死这个偏移，如果TP固定为2，可直接往起始数据偏移开始读写
    constexpr static uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
    constexpr static uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
    constexpr static uint64_t TOTAL_STATE_OFFSET = 3 * 1024UL * 1024UL;  // dispatch+combine使用的status space
    constexpr static uint32_t TP_STATE_SIZE = 100U * 1024U;
    constexpr static uint32_t WORKSPACE_ELEMENT_OFFSET = 512U;
    constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;
    constexpr static uint64_t ALIGNED_LEN_256 = 256UL;
    constexpr static uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id

    constexpr static uint64_t SCALE_EXPAND_IDX_BUFFER = 44UL;  // scale32B + 3*4expandIdx
    constexpr static uint64_t DOUBLE_DATA_BUFFER = 2UL;
    constexpr static uint64_t MAX_OUT_DTYPE_SIZE = 2UL;

    template <AscendC::HardEvent event>
    __aicore__ inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

public:
    __aicore__ inline MoeDistributeDispatchV2Single(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalVaildBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalVaildExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SetStatus();
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t totalCount);
    __aicore__ inline void AllGatherSetStatusAndWait();
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void AllgatherProcessOut();
    __aicore__ inline void UpdataTokenNumsOut();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void SyncCntOnCore(LocalTensor<float> &gatherMaskOutTensor,
                                         LocalTensor<uint32_t> &gatherTmpTensor,
                                         LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        // uint32_t curRankId = ((ctxIdx == COMM_EP_IDX) ? epRankId_ : tpRankId_);
        // if (curRankId == rankId) {
        //     return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + winDataSizeOffset_;
        // }
        // return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn)
        //                  + winDataSizeOffset_;
        return hccl_.GetWindowsInAddr(rankId) + winDataSizeOffset_ + TOTAL_STATE_OFFSET;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        // uint32_t curRankId = ctxIdx == COMM_EP_IDX ? epRankId_ : tpRankId_;
        // if (curRankId == rankId) {
        //     return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + dataState_ * WIN_STATE_OFFSET;
        // }
        // return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn)
        //                  + dataState_ * WIN_STATE_OFFSET;
        return hccl_.GetWindowsInAddr(rankId) + dataState_ * (totalWinSize_ / 2) + dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int typeSize = static_cast<int>(sizeof(int32_t));
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<ExpandXOutType> winTpGatherOutGMTensor_;
    GlobalTensor<float> fpWinTpGatherOutGMTensor_;
    GlobalTensor<int32_t> winTpEpCntGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<int32_t> tpTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> vaildExpertsIdTensor_;
    LocalTensor<int32_t> vaildExpIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<int32_t> vaildBsIndexTensor_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_;  // gather mask输出buf
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;  // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> maskBuf_;
    TBuf<> vaildExpertsBuf_;
    TBuf<> vaildExpertIdxBuf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> vaildBsIndexTBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;  // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                         // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                       // 量化使用，量化后的输出

    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR tpStatusWindowGM_;
    GM_ADDR tpLocalWindowGM_;
    GM_ADDR tpLocalStatusWindowGM_;
    GM_ADDR recvCntWorkspaceGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t tpGatherRankId_{0};  // gather 对端ID
    uint32_t tpRankId_{0};        // 本卡 ID
    uint32_t aivId_{0};           // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};     // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0};  // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};  // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t axisBsAlignSize_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_;  // 接收Win区块数
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    float sumTarget_;
    uint64_t totalWinSize_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t preCnt_{0};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t sendToSharedExpTokenCnt_{0};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};

    DataCopyExtParams floatDataCopyParams_;
    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
};

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    CAM_PRINT("[Init] enter init...\n");
    // return;
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    auto tiling = (__gm__ MoeDistributeDispatchV2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_[COMM_TP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<1>();  // 没有相关公共宏

    // windowInGM_ = hccl_.GetWindowsInAddr(epRankId_);
    // windowOutGM_ = hccl_.GetWindowsOutAddr(epRankId_);
    epRankId_ = tilingData.moeDistributeDispatchV2Info.epRankId;

    // 检查hcclwinsize是否越界
    totalWinSize_ = static_cast<uint64_t>(tilingData.moeDistributeDispatchV2Info.totalWinSize);
    auto realWinSize = winContext_[COMM_EP_IDX]->winSize;
    CheckWindowSize(totalWinSize_, realWinSize, tpipe_, expandXOut);

    GlobalTensor<int32_t> selfDataStatusTensor;
    // GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_[0]->localWindowsExp);
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(hccl_.GetWindowsInAddr(epRankId_));  // TODO:预留状态位置到windows开头
    selfDataStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));

    axisBS_ = tilingData.moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData.moeDistributeDispatchV2Info.h;
    epWorldSize_ = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    axisMaxBS_ = tilingData.moeDistributeDispatchV2Info.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData.moeDistributeDispatchV2Info.sharedExpertNum;
    sharedExpertRankNum_ = tilingData.moeDistributeDispatchV2Info.sharedExpertRankNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    expertTokenNumsType_ = tilingData.moeDistributeDispatchV2Info.expertTokenNumsType;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;

    tpRankId_ = tilingData.moeDistributeDispatchV2Info.tpRankId;
    tpGatherRankId_ = ((tpRankId_ == 0) ? 1 : 0);
    isTokenMaskFlag_ = tilingData.moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData.moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData.moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData.moeDistributeDispatchV2Info.aivNum;
    tpWorldSize_ = tilingData.moeDistributeDispatchV2Info.tpWorldSize;
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;  // 无GlobalTensor
    sendTpCountOutGM_ = tpSendCountsOut;
    recvCntWorkspaceGM_ = workspaceGM;

    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN;  // scale起始放置偏移
    uint32_t hScaleSizeAlign = hOutSizeAlign_ + UB_ALIGN;   // 填充三元组起始偏移
    tokenQuantAlign_ = hScaleSizeAlign / sizeof(int32_t);
    // 实际搬运大小，搬运token_align32B + 32B(float) + 3*4B(三元组)
    uint32_t hScaleIdxSize = hScaleSizeAlign + EXPAND_IDX_INFO * sizeof(int32_t);
    hAlignWinSize_ = Ceil(hScaleIdxSize, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;  // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    dealRankPerCore_ = (recvWinBlockNum_ + aivNum_ - 1) / aivNum_;
    stateOffset_ = STATE_OFFSET;

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    if (dataState_ == 0) {
        selfDataStatusTensor(0) = 1;
    } else {
        selfDataStatusTensor(0) = 0;
    }

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    PipeBarrier<PIPE_ALL>();

    if constexpr (IsShareExpertRank) {
        rscvStatusNum_ = epWorldSize_;
    } else {
        rscvStatusNum_ = recvWinBlockNum_;
    }
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_;  // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_;  // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                   // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    uint32_t statusBufCntAlign = Ceil(Ceil(totalExpertNum_, aivNum_), 8) * 8;  // 8 = UB_ALIGN / sizeof(int32_t)
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign * UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8);  // 8 = UB_ALIGN / sizeof(int32_t)

    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankId_);

#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_EP_IDX, tempepRankId)),
                                          totalWinSize_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWindStateAddrByRankId(COMM_EP_IDX, tempepRankId)), STATE_SIZE);
    }
#endif
    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = {0x101010101010101, 0};  // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8);  // 0x3F800000是float的1

    // 当前tpWin区划分为前后两半区，连续两次dispatch，切换半区, combine 数据区使用前面，
    // 即axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine, dispatch使用后面
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (tilingData.moeDistributeDispatchV2Info.totalWinSize / 2) +
                         axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankId_);

#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    GlobalTensor<ExpandXOutType> winDouble;
    winDouble.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    winDouble.SetGlobalBuffer((__gm__ ExpandXOutType *)(windowGM_));
    OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(winDouble.GetPhyAddr()), totalWinSize_);
#endif
    if constexpr (IsNeedAllgater) {
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        for (int temptpRankId = 0; temptpRankId < tpWorldSize_; temptpRankId++) {
            OOMCheckAddrRange<ExpandXOutType>((__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_TP_IDX, temptpRankId)),
                                              totalWinSize_);
            OOMCheckAddrRange<int32_t>((__gm__ int32_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, temptpRankId)),
                                       STATE_SIZE);
        }
#endif
        tpLocalWindowGM_ = GetWindAddrByRankId(COMM_TP_IDX, tpRankId_);
        tpLocalStatusWindowGM_ = GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_);
        tpWindowGM_ = GetWindAddrByRankId(COMM_TP_IDX, tpGatherRankId_);
        tpStatusWindowGM_ = GetWindStateAddrByRankId(COMM_TP_IDX, tpGatherRankId_);
    }
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)windowGM_);
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_));
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    if constexpr (IsNeedAllgater) {
        winTpGatherOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)tpWindowGM_);
        fpWinTpGatherOutGMTensor_.SetGlobalBuffer((__gm__ float *)tpWindowGM_);
        winTpEpCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(tpStatusWindowGM_ + TP_STATE_SIZE));
    }

    hOutAlignUbSize_ = Ceil(hScaleIdxSize, UB_ALIGN) * UB_ALIGN;
    expertIdsCnt_ = axisBS_ * axisK_;
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    if constexpr (DynamicQuant || StaticQuant) {
        QuantInit(scales);
        dstExpBuf_ = receiveDataCastFloatBuf_;  // 内存复用
        subExpBuf_ = smoothScalesBuf_;          // 内存复用
    } else {
        tpipe_->InitBuffer(dstExpBuf_, expertIdsSize);              // BS * K * 4 = 32K
        tpipe_->InitBuffer(subExpBuf_, expertIdsSize);              // BS * K * 4 = 32K
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_);  // 7k*2 + 32 + 12
    }
    if (isExpertMaskFlag_) {
        uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(xActMaskTBuf_, activeMaskAlignSize);
        tpipe_->InitBuffer(xActMaskCastTBuf_, activeMaskAlignSize * sizeof(half));
        tpipe_->InitBuffer(vaildBsIndexTBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(vaildExpertsBuf_, expertIdsSize);
        tpipe_->InitBuffer(vaildExpertIdxBuf_, expertIdsSize);
        gatherMaskTensor_ = dstExpBuf_.Get<uint32_t>();
        vaildExpIndexTensor_ = vaildExpertIdxBuf_.Get<int32_t>();
        vaildExpertsIdTensor_ = vaildExpertsBuf_.Get<int32_t>();
        vaildBsIndexTensor_ = vaildBsIndexTBuf_.Get<int32_t>();
    }
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);  // BS * K * 4 = 32K
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    // reduceSum计算所需的Tensor空间，取最大统一前面申请
    int32_t reduceSumWorkNeedSize = ReduceSumWorkNeedSize(expertIdsCnt_);
    tpipe_->InitBuffer(workLocalBuf_, reduceSumWorkNeedSize * sizeof(int32_t));
    workLocalTensor_ = workLocalBuf_.Get<float>();

    uint32_t axisHCommu = hScaleIdxSize / sizeof(ExpandXOutType);  // 有效搬运长度
    floatDataCopyParams_ = {1U, sizeof(float), 0U, 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(ExpandXOutType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::QuantInit(GM_ADDR scales)
{
    uint32_t hAlignSize = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hAlignSize);         // 14K * 2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hOutAlignUbSize_);  // 7K * 2 + 32 + 6
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN);  // 32B
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);               // 28K
    uint32_t expertIdxSize = expertIdsCnt_ * sizeof(int32_t);  // BS * K * 4B
    uint32_t maxSize = hFp32Size > expertIdxSize ? hFp32Size : expertIdxSize;

    tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize);  // max{28K, BS * K * 4B}
    tpipe_->InitBuffer(smoothScalesBuf_, maxSize);          // max{28K, BS * K * 4B}
    smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_;  // 由于是后面的核作为发送的共享专家，因此需要换算
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

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::FillTriple(
    LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex;
    xOutTint32(tokenQuantAlign_ + 2) = k;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::SendToSharedExpert()
{
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    uint32_t startTokenId, endTokenId, sendTokenNum;  // 每个aiv发送时的起始rankid
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {
        return;
    }

    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;  // 计算目的共享专家卡在其所在共享专家组的id
    GlobalTensor<ExpandXOutType> dstWinGMTensor;

    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t tokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        uint32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_EP_IDX, toRankId) + expertPerSizeOnWin_ * epRankId_));
        uint32_t srcTokenIndex = tokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = vaildBsIndexTensor_.GetValue(tokenIndex);
        }

        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopyPad(xInTensor_, xGMTensor_[srcTokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess(toSharedExpertIndex);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            FillTriple(xOutTensor_, srcTokenIndex, axisK_ + toSharedExpertIndex);
            DataCopyPad(dstWinGMTensor[tokenIndex * hAlignWinCnt_], xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[srcTokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            FillTriple(xTmpTensor_, srcTokenIndex, axisK_ + toSharedExpertIndex);
            DataCopyPad(dstWinGMTensor[tokenIndex * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::CalTokenSendExpertCnt(
    uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    if (isExpertMaskFlag_) {
        Sub(subExpIdTensor_, vaildExpertsIdTensor_, dstExpIdTensor_, calCnt);
    } else {
        Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::SendToMoeExpert()
{
    CAM_PRINT("[SendToMoeExpert] rank:%d, aivId:%d ...\n", epRankId_, aivId_);
    uint32_t startTokenId, endTokenId, sendTokenNum;  // 每个aiv发送时的起始rankid
    SplitToCore(sendToMoeExpTokenCnt_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= sendToMoeExpTokenCnt_) {
        return;
    }
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    int32_t tokenIndex = 0;
    int32_t topKIndex = 0;
    uint32_t dstExpertId = 0;
    int32_t expertIdx = 0;
    if (isExpertMaskFlag_) {
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    for (int32_t index = startTokenId; index < endTokenId; ++index) {
        if (isExpertMaskFlag_) {
            expertIdx = vaildExpIndexTensor_(index);
            tokenIndex = expertIdx / axisK_;
        } else {
            expertIdx = index;
            tokenIndex = index / axisK_;
        }
        int32_t curExpertCnt = 0;
        topKIndex = expertIdx % axisK_;
        dstExpertId = expertIdsTensor_(expertIdx);
        if ((tokenIndex > 0) && (index > 0)) {
            CalTokenSendExpertCnt(dstExpertId, index, curExpertCnt);
        }
        uint32_t toRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;

        if (epRankId_ == 0) {
            CAM_PRINT("[SendToMoeExpert0] rank:%d, aivId:%d, toRankId:%d, dstExpertId:%d, curExpertCnt:%d ...\n",
                      epRankId_, aivId_, toRankId, dstExpertId, curExpertCnt);
        }

        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, toRankId) +
                                            (expertPerSizeOnWin_ *
                                             (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                            hAlignWinSize_ * curExpertCnt);  // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);

        if (epRankId_ == 0) {
            CAM_PRINT("[SendToMoeExpert1] rank:%d, aivId:%d, dstWinGMTensor(0):%d ...\n", epRankId_, aivId_,
                      dstWinGMTensor.GetValue(0));
        }

        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess(dstExpertId + sharedExpertNum_);
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            FillTriple(xOutTensor_, tokenIndex, topKIndex);
            DataCopyPad(dstWinGMTensor, xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            FillTriple(xTmpTensor_, tokenIndex, topKIndex);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopyPad(dstWinGMTensor, xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }

        if (epRankId_ == 0) {
            CAM_PRINT("[SendToMoeExpert2] rank:%d, aivId:%d, dstWinGMTensor(0):%d ...\n", epRankId_, aivId_,
                      dstWinGMTensor.GetValue(0));
        }
    }
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = expertIdsBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void
MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::CalVaildExpIdx(LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = dstExpBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(vaildExpIndexTensor_, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void
MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::CalVaildBSCnt(LocalTensor<bool> maskStrideTensor)
{
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = vaildBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = xActMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = xActMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(vaildBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::ExpertActiveMaskCal()
{
    // 计算vaildExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalVaildExpIdx(maskInputTensor);
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = xActMaskTBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{true, 0U, static_cast<uint8_t>(UB_ALIGN - axisK_), 0U};
    DataCopyExtParams maskStrideParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U,
                                       0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalVaildBSCnt(maskStrideTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::AlltoAllDispatch()
{
    CAM_PRINT("[AlltoAllDispatch1] rank:%d, aivId:%d ...\n", epRankId_, aivId_);
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }
    CAM_PRINT("[AlltoAllDispatch2] rank:%d, aivId:%d, isTokenMaskFlag_:%d ...\n", epRankId_, aivId_, isTokenMaskFlag_);
    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    CAM_PRINT("[AlltoAllDispatch3] rank:%d, aivId:%d, isExpertMaskFlag_:%d ...\n", epRankId_, aivId_,
              isExpertMaskFlag_);

    if (activeMaskBsCnt_ == 0) {
        return;
    }
    // 后面的核向共享专家发数据
    bool isSendShared = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared) {
        SendToSharedExpert();
    }

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    CAM_PRINT("[AlltoAllDispatch4] rank:%d, aivId:%d ...\n", epRankId_, aivId_);
    if (isExpertMaskFlag_) {
        uint64_t rsvdCnt = 0;
        uint32_t mask = expertIdsCnt_;
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(vaildExpertsIdTensor_, expertIdsTensor_, gatherMaskTensor_, true, mask, {1, 1, 1, 0}, rsvdCnt);
    } else {
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    CAM_PRINT("[AlltoAllDispatch5] rank:%d, aivId:%d isSendShared:%d...\n", epRankId_, aivId_, isSendShared);
    if (isSendShared) {  // 用于send共享专家数据的核，也需要搬运expertIds，后续会重新分核写状态位置，该核可能用于写moe专家flag
        return;
    }
    SendToMoeExpert();
    CAM_PRINT("[AlltoAllDispatch6] rank:%d, aivId:%d ...\n", epRankId_, aivId_);
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::SetStatus()
{
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);
    if (sendToMoeExpTokenCnt_ > 0) {  // 需要发送的再算
        for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
            int32_t curExpertCnt = 0;
            // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
            int32_t cntPosIndex = (curExpertId - startExpertId_) * 8 + 1;
            if (curExpertId <
                sharedExpertRankNum_) {  // 当前处理专家为共享专家 shared:Cnt -> win -> LocalCopy(moe+share)
                if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                    curExpertCnt = activeMaskBsCnt_;
                }
            } else {  // 当前处理卡为moe专家卡
                int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
                CalTokenSendExpertCnt(curMoeExpertId, sendToMoeExpTokenCnt_, curExpertCnt);
            }
            statusTensor_(cntPosIndex) = curExpertCnt;
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GlobalTensor<int32_t> rankGMTensor;

    for (uint32_t expertIndex = startExpertId_; expertIndex < endExpertId_; ++expertIndex) {
        uint32_t dstRankId = expertIndex;
        uint32_t offset = stateOffset_ * epRankId_;
        if (expertIndex >= sharedExpertRankNum_) {
            dstRankId = ((expertIndex - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((expertIndex - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset);  // 计算地址偏移
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        // 按32对齐拷贝，8是32字节包含的元素个数, 本卡数据需要去掉起始index偏移
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(expertIndex - startExpertId_) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
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

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();

    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        if constexpr (DynamicQuant) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();  // ub复用，循环同步
        }
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        rowMaxTensor_ = rowMaxBuf_.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / floatLocalAbsTemp.GetValue(0);
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

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign_ / sizeof(float), float(1.0) / dynamicScale);  // int8->float32
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::SyncCntOnCore(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家recv cnt收集到gatherMaskOutTensor
    uint64_t rsvdCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和
    SumParams sumParams{1, recStatusNumPerCore_, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);

    // 把当前核的所有专家的recv cnt之和写到workspace
    uint32_t coreOffset = WORKSPACE_ELEMENT_OFFSET * aivNum_;
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + coreOffset * aivId_;  // 写workspace需要按照512字节对齐
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    uint16_t workCoreNum = MIN(recvWinBlockNum_, aivNum_);
    // 每个核把sumOfRecvCnt重复写workCoreNum份
    LocalTensor<int32_t> sumCoreTensor = sumCoreBuf_.Get<int32_t>();
    // 仅处理每个datablock的首元素（对应maskArray[0]的bit0）。操作数为32bit情况下，maskArray只有第0个元素有效
    // 每个元素占4字节，每个32字节处理8份，mask中每8个bit的填充第1位
    uint64_t maskArray[2] = {0x0101010101010101, 0};
    // 每个核一个datablock，总共需要处理workCoreNum个核。每个repeat总共256字节，可以处理8个datablock
    uint8_t repeatTimes = (workCoreNum + 7) / 8;
    // 1代表单个repeat内不同的datablock连续，没有跳过
    // 8代表不同repeat的首元素间隔8个datablock
    Duplicate<int32_t>(sumCoreTensor, sumOfRecvCnt, maskArray, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(workCoreNum), 1, 0, 15};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(sumTensor, sumCoreTensor, sumIntriParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::BufferInit()
{
    tpipe_->Reset();
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);                    // 1024/24 * 32B = 43 * 32B
    tpipe_->InitBuffer(gatherMaskOutBuf_, recvWinBlockNum_ * sizeof(float));  // 1024 * 4B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);                      // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);                     // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, aivNum_ * sizeof(float));             // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);                             // 96B
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_);                // 7k*2 + 32 + 12
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::WaitDispatch()
{
    BufferInit();
    startExpertId_ = startStatusIndex_;  // 后面LocalWinCopy分核与此处保持一致
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;
    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;  // gatherMask + sum 相关参数
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    // 清状态
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};  // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    // 核间同步token cnt
    SyncCntOnCore(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);

    SyncAll<true>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::GetCumSum(
    LocalTensor<int32_t> &outLocal, uint32_t totalCount)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    // 获取workspace中每个核的recvcnt
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_) + WORKSPACE_ELEMENT_OFFSET * aivId_;
    GlobalTensor<int32_t> sumTensor;
    sumTensor.SetGlobalBuffer((__gm__ int32_t *)wAddr);

    // 不支持allgather场景，只需要拷贝totalCount个核的recv cnt
    uint16_t copySumNum = totalCount;
    if constexpr (IsNeedAllgater) {
        copySumNum = MIN(recvWinBlockNum_, aivNum_);
    }
    uint16_t copyStride = 16 * aivNum_ - 1;
    DataCopyParams sumIntriParams{static_cast<uint16_t>(copySumNum), 1, copyStride, 0};
    LocalTensor<int32_t> sumLocalTensor = sumLocalBuf_.Get<int32_t>();
    DataCopy(sumLocalTensor, sumTensor, sumIntriParams);
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherSumPattern.SetValue(0, 1);
    uint32_t mask = 1;
    uint64_t rsvdCnt = 0;
    LocalTensor<int32_t> sumContinueTensor = sumContinueBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask, {1, copySumNum, 1, 0}, rsvdCnt);
    // height, width(按照32字节对齐padding后总元素个数), nNum，结果矩阵第一列为对应行的求和结果
    uint32_t innerSumParams = (copySumNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / sizeof(float);
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = sumContinueTensor.ReinterpretCast<float>();

    if constexpr (IsNeedAllgater) {
        SumParams allCoreSumParams{1, innerSumParams, copySumNum};
        Sum(recvCntSumOutTensor, tmpFp32, allCoreSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        totalCnt_ = recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
    }
    // 0核前面所有核recv cnt总和是0
    if (totalCount == 0) {
        outLocal.SetValue(0, 0);
        return;
    }
    SumParams sumParams{1, innerSumParams, totalCount};
    Sum(recvCntSumOutTensor, tmpFp32, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    // 最终输出outLocal第0个元素是当前核前面所有核recv cnt总和
    outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::LocalWindowCopy()
{
    LocalTensor<int32_t> outCountLocal;
    if (startExpertId_ >= rscvStatusNum_) {  // 分核已与前面的waitDispatch里保持一致
        return;
    }
    GetCumSum(outCountLocal, aivId_);
    uint32_t index = 0;
    uint32_t beginIdx = outCountLocal.GetValue(0);
    preCnt_ = beginIdx;
    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    DataCopyPadExtParams<ExpandXOutType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyExandIdxParams{1U, sizeof(int32_t) * EXPAND_IDX_INFO, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1);
        outCountLocal.SetValue(i, beginIdx + count);
        if constexpr (IsNeedAllgater) {
            gatherCount_ += count;
        }
        uint32_t winOffset = index;
        if constexpr (!IsShareExpertRank) {
            if (moeExpertNumPerRank_ > 1) {  // moe专家卡且一卡多专家场景 转换成数据区的排布偏移
                winOffset = index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;
            }
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<ExpandXOutType> tokGlobal;
        GlobalTensor<ExpandXOutType> expandXOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<int32_t> xTmpTensorInt;
        for (uint32_t j = 0; j < count; j++) {
            tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr + j * hAlignWinSize_));
            // 将数据从Window拷贝到UB
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, tokGlobal, hCommuCopyOutParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            xTmpTensorInt = xTmpTensor_.template ReinterpretCast<int32_t>();
            DataCopyPad(expandIdxGMTensor_[(beginIdx + j) * EXPAND_IDX_INFO], xTmpTensorInt[tokenQuantAlign_],
                        dataCopyExandIdxParams);
            if constexpr (DynamicQuant || StaticQuant) {
                xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[beginIdx + j], xOutFp32Tensor_[hOutSizeAlign_ / sizeof(float)],
                            floatDataCopyParams_);
            }
            if constexpr (IsNeedAllgater) {
                DataCopyPad(winTpGatherOutGMTensor_[(beginIdx + j) * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            }
            expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (beginIdx + j) * axisH_,
                                             axisH_);
            DataCopyPad(expandXOutGlobal, xTmpTensor_, expandXCopyParams_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        beginIdx += count;
    }
    if constexpr (!IsNeedAllgater) {
        totalCnt_ = beginIdx;
    }
    lastCore_ = MIN(rscvStatusNum_, aivNum_) - 1;

    if constexpr (IsNeedAllgater) {
        DataCopyPad(winTpEpCntGMTensor_[startExpertId_], outCountLocal, dataCopyOutParams);
    }

    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    DataCopyPad(sendCountsGlobal[startExpertId_], outCountLocal, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::AllGatherSetStatusAndWait()
{
    PipeBarrier<PIPE_ALL>();
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }

    GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpGatherRankId_) +
                                        WIN_ADDR_ALIGN * aivId_);  // 计算地址偏移
    GlobalTensor<float> tpwindowInstatusFp32Tensor_;
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(rankGM));
    statusTensor_(aivId_ * 8 + 1) = gatherCount_;
    statusTensor_(aivId_ * 8 + 2) = preCnt_;
    LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    statusFp32Tensor_(aivId_ * 8) = sumTarget_;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(tpwindowInstatusFp32Tensor_, statusFp32Tensor_[aivId_ * 8],
                    UB_ALIGN);  // 12是数据大小，按32对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    float sumOfFlag = static_cast<float>(-1.0);
    rankGM =
        (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_) + WIN_ADDR_ALIGN * aivId_);  // 计算地址偏移
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(rankGM));
    while (sumOfFlag != sumTarget_) {
        DataCopy(statusFp32Tensor_, tpwindowInstatusFp32Tensor_, UB_ALIGN);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        sumOfFlag = statusFp32Tensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    tpwindowInstatusFp32Tensor_(0) = static_cast<float>(0.0);
    DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tpwindowInstatusFp32Tensor_);
}

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::AllgatherProcessOut()
{
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    // 获取需要allgather的tokens数量
    GlobalTensor<float> tpwindowInstatusFp32Tensor_;
    GM_ADDR rankGM =
        (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_) + WIN_ADDR_ALIGN * aivId_);  // 计算地址偏移
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)rankGM);
    LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    DataCopy(statusFp32Tensor_, tpwindowInstatusFp32Tensor_, UB_ALIGN);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t coreGatherCount = statusFp32Tensor_.ReinterpretCast<int32_t>().GetValue(1);
    uint32_t preCount = statusFp32Tensor_.ReinterpretCast<int32_t>().GetValue(2);
    gatherCount_ = coreGatherCount;
    preCnt_ = preCount;
    GlobalTensor<int32_t> sendCountsGlobal;
    GlobalTensor<int32_t> tpGlobal;

    // 搬运另一个tp域卡传来的epRcvCnt
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    tpGlobal.SetGlobalBuffer((__gm__ int32_t *)(tpLocalStatusWindowGM_ + TP_STATE_SIZE));
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    tpTmpTensor_ = xQueue_.AllocTensor<int32_t>();
    DataCopyPad(tpTmpTensor_, tpGlobal[startExpertId_], dataCopyParams, copyPadParams);
    xQueue_.EnQue(tpTmpTensor_);
    tpTmpTensor_ = xQueue_.DeQue<int32_t>();
    DataCopyPad(sendCountsGlobal[epWorldSize_ + startExpertId_], tpTmpTensor_, dataCopyParams);
    xQueue_.FreeTensor(tpTmpTensor_);

    if (coreGatherCount == 0) {
        return;
    }
    // 输出起始偏移本卡数据
    GlobalTensor<ExpandXOutType> tokGlobal;
    GlobalTensor<ExpandXOutType> expandXOutGlobal;
    DataCopyPadExtParams<ExpandXOutType> copyPadExtParams{false, 0U, 0U, 0U};
    for (uint32_t i = 0; i < coreGatherCount; i++) {
        tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(tpLocalWindowGM_ + (preCount + i) * hAlignWinSize_));
        xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
        DataCopyPad(xTmpTensor_, tokGlobal, hCommuCopyOutParams_, copyPadExtParams);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
        expandXOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(expandXOutGM_ + (preCount + totalCnt_ + i) * hOutSize_));
        DataCopyPad(expandXOutGlobal, xTmpTensor_, expandXCopyParams_);
        if constexpr (StaticQuant || DynamicQuant) {
            xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
            DataCopyPad(dynamicScalesOutGMTensor_[preCount + totalCnt_ + i],
                        xOutFp32Tensor_[hOutSizeAlign_ / sizeof(float)], floatDataCopyParams_);
        }
        xQueue_.FreeTensor(xTmpTensor_);
    }
}

// 更新tokenNumsOut tensor
template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::UpdataTokenNumsOut()
{
    // 最后一个核做更新，Moe专家只有最后一个核有计算出所有 sendCountsGlobal
    if constexpr (!IsShareExpertRank) {
        if (moeExpertNumPerRank_ > 1) {
            SyncAll<true>();
        }
    }

    if (aivId_ == lastCore_) {
        // Moe专家token总数在Cumsum内计算得出
        uint32_t tokenNum = totalCnt_;
        if constexpr (IsNeedAllgater) {
            tokenNum += preCnt_;
            tokenNum += gatherCount_;
        }
        expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsOutGMTensor_);
        // moe一卡多专家场景下更新moe专家卡对应expertTokenNums数据
        if (moeExpertNumPerRank_ != 1) {
            if constexpr (!IsShareExpertRank) {
                uint32_t tokenSums = 0;
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                GlobalTensor<int32_t> sendCountsGlobal;
                sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    sendCountsGlobal[epWorldSize_ - 1]);

                uint32_t firstMoeCnt = sendCountsGlobal.GetValue(epWorldSize_ - 1);
                tokenSums = firstMoeCnt + gatherCount_;
                expertTokenNumsOutGMTensor_.SetValue(0, tokenSums);
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    expertTokenNumsOutGMTensor_[0]);
                for (uint32_t localMoeIndex = 1; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
                    uint32_t preOffset = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
                    uint32_t curOffset = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[preOffset]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        sendCountsGlobal[curOffset]);
                    uint32_t preMoeIndexCnt = sendCountsGlobal.GetValue(preOffset);
                    uint32_t curMoeIndexCnt = sendCountsGlobal.GetValue(curOffset);
                    tokenSums = ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt) +
                                gatherCount_;
                    expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                        expertTokenNumsOutGMTensor_[localMoeIndex]);
                }
            }
        }

        // token总数 = 其他专家搬进来的token数 + allgather拿到的另一张卡token数
        if constexpr (IsNeedAllgater) {
            GlobalTensor<int32_t> sendTpCountsGlobal;
            sendTpCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendTpCountOutGM_));
            sendTpCountsGlobal.SetValue(tpRankId_, totalCnt_);
            sendTpCountsGlobal.SetValue(tpGatherRankId_, gatherCount_ + preCnt_);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sendTpCountsGlobal);
        }
    }
}

// 流水
//          24 win->ub ub->gm
// 48 alltoall                                 syncAll 48 AllgatherOut
//                              1 setStatus
//          24 win->ub ub->win
//                              1 waitStatus

template <TemplateMC2TypeA2SingleClass>
__aicore__ inline void MoeDistributeDispatchV2Single<TemplateMC2TypeA2SingleFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        CAM_PRINT("[Process] enter process...\n");
        AlltoAllDispatch();
        CAM_PRINT("[Process] AlltoAllDispatch, rank:%d, aivId:%d ...\n", epRankId_, aivId_);

        SetStatus();
        CAM_PRINT("[Process] SetStatus, rank:%d, aivId:%d ...\n", epRankId_, aivId_);
        WaitDispatch();
        CAM_PRINT("[Process] WaitDispatch, rank:%d, aivId:%d ...\n", epRankId_, aivId_);
        LocalWindowCopy();
        CAM_PRINT("[Process] LocalWindowCopy, rank:%d, aivId:%d, IsNeedAllgater:%d ...\n", epRankId_, aivId_,
                  IsNeedAllgater);
        if constexpr (IsNeedAllgater) {
            AllGatherSetStatusAndWait();
            AllgatherProcessOut();
        }
        UpdataTokenNumsOut();
        CAM_PRINT("[Process] UpdataTokenNumsOut, rank:%d, aivId:%d ...\n", epRankId_, aivId_);

        SyncAll<true>();
        hccl_.Finalize();
        CAM_PRINT("[Process] Finalize, rank:%d, aivId:%d ...\n", epRankId_, aivId_);
    }
}

}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_V2_SINGLE_H
