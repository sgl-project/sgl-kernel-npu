#ifndef CAM_MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
#define CAM_MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../cam_moe_distribute_dispatch_tiling.h"
#include "../moe_distribute_base.h"

namespace MoeDistributeDispatchA2Impl {
constexpr uint32_t STATE_OFFSET = 512;                 // 状态空间偏移地址
constexpr uint32_t STATUS_SIZE_LAYERED = 1024 * 1024;  // 1M
constexpr uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;  // 本卡状态空间偏移地址
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t INFO_NUM_IN_TOKENSTRUCK = 4;  // 在Token后加入3种信息:expIds, weights, tokenIdx, scales
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint32_t PER_MSG_RDMA_SEND_TIME = 2;
constexpr uint32_t B32_PER_BLOCK = 8;
constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 176 * 1024;
constexpr uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 64 * 32;
constexpr uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
constexpr uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
constexpr uint32_t NOTIFY_OFFSET = 204 * 1024 * 1024;
constexpr uint32_t IPC_BUFF_ALIGN = 512;
constexpr uint32_t TOKEN_COUNT_SIZE = 32;
constexpr uint32_t FLAG_U32_CNT = TOKEN_COUNT_SIZE / 4;
constexpr int32_t IPC_FLAG_STEP_1 = 1;
constexpr int32_t IPC_FLAG_STEP_2 = 2;
constexpr uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
constexpr uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
constexpr uint32_t RDMA_DATA_SIZE = 100U * 1024U * 1024U;
constexpr uint32_t EXTRA_TOKEN_INFO_NUM = 4U;  // 专家信息 权重信息 量化Scale 到达标志位
constexpr uint32_t BITS32_PER_BLOCK = 8U;
constexpr static uint32_t BW_ITEM_SIZE = 32;
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint32_t BS_UPPER = 16;

#define TemplateMC2TypeA2layeredClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2layeredFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeA2layeredClass>
class CamMoeDistributeDispatchA2Layered
{
    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

public:
    __aicore__ inline CamMoeDistributeDispatchA2Layered(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales,
                                GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt, GM_ADDR epRankTokenCnt,
                                GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR epRecvCountsOut, GM_ADDR expandScales, GM_ADDR workspaceGM, TPipe *pipe,
                                GM_ADDR tilingGM);
    __aicore__ inline void Process();
    template <AscendC::HardEvent event>
    __aicore__ inline void SyncFunc()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

private:
    __aicore__ inline void Input2Win();
    __aicore__ inline uint32_t GetExpRank(uint32_t expertId);
    __aicore__ inline bool IsInSameServer(uint32_t targetRankId);
    __aicore__ inline void SetTokenCnt(GlobalTensor<int32_t> globalSet);
    __aicore__ inline void CopyTokenToWinOut(uint32_t localTokenIdx, uint32_t tokenIdx, uint32_t dstServerId);
    __aicore__ inline void WaitWindow();

    __aicore__ inline void Win2Ipc();
    __aicore__ inline void Ipc2Out();
    __aicore__ inline void DispatchBetweenServer();
    __aicore__ inline void ConstructDataAndFlagBatchWriteInfo();
    __aicore__ inline void WaitIpcFlag(int32_t flagVal = 1);
    __aicore__ inline void SetIpcFlag(int32_t flagVal = 1);
    __aicore__ inline void WriteRdmaCntInfo();
    __aicore__ inline void CleanUp();
    __aicore__ inline void QuantProcess(uint32_t sendTokenNum, LocalTensor<XType> xTokenLt,
                                        LocalTensor<float> tokenCastLt);
    __aicore__ inline int64_t MergeMagicWithValue(int32_t magic, int32_t value);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGt;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint8_t> readTokensU8Tensor_;
    GlobalTensor<uint8_t> sendTokensU8Tensor_;
    GlobalTensor<uint32_t> sendTokensU32Tensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<uint32_t> expertToServerGlobalTensor_;
    GlobalTensor<int32_t> readStatusTensor_;
    GlobalTensor<int32_t> tokenServerIdxGMTensor_;
    GlobalTensor<int32_t> tokenServerCntGMTensor_;

    GlobalTensor<int32_t> epRankTokenCntGMTensor_;
    GlobalTensor<int32_t> srcOffsetRankTokenIdxGMTensor_;
    GlobalTensor<int32_t> dstOffsetRankTokenIdxGMTensor_;

    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertToServerCntTensor_;
    LocalTensor<uint32_t> expertToServerIdxTensor_;

    LocalTensor<int32_t> tokenServerIdxTensor_;
    LocalTensor<int32_t> serverCountTensor_;

    TBuf<> tokenServerIdxBuf_;
    TBuf<> serverCountBuf_;

    TBuf<> expertCountBuf_;
    TBuf<> statusBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertToServerCntsBuf_;  // 总表，int类型只写1/0
    TBuf<> expertToServerIdxBuf_;
    TBuf<QuePosition::VECCALC> tBuf;

    GM_ADDR expandXGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR weightsGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR dataBatchWriteInfo_;
    GM_ADDR expertToServerCntGM_;
    GM_ADDR shareAddrs[8];
    GM_ADDR shareAddrWins[8];

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t kAlign_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0};         // aiv id
    uint32_t moeExpertNum_{0};  // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumInServer_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t SERVER_SIZE_ON_WIN{0};
    uint32_t RANK_SIZE_ON_IPC{0};
    uint32_t WIN_SIZE{0};
    uint32_t bufferId_{0};
    uint32_t totalSize_{0};
    uint32_t totalWinSize_{0};
    uint32_t halfWinSize_{0};
    uint32_t serverNum{0};
    uint32_t expertTokenNumsType_{0};
    uint32_t shareMemOffset_{0};
    // TokenStruck相关
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    int32_t magicVal_{0};

    uint32_t combineInnerCntOffset;
    uint32_t combineInnerCntIndexOffset;
    uint32_t combineOuterCntOffset;
    uint32_t combineOuterCntIndexOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt,
    GM_ADDR epRankTokenCnt, GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
    GM_ADDR expandScales, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    PRINTF("[A2layer Init]\n");
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(CamMoeDistributeDispatchA2TilingData);
    auto tiling = (__gm__ CamMoeDistributeDispatchA2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    winContext_ = (__gm__ HcclOpResParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_) + NOTIFY_OFFSET;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);
    // return;

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    globalBs_ = tilingData.moeDistributeDispatchInfo.globalBs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    kAlign_ = RoundUp(axisK_, (uint32_t)8);
    totalSize_ = winContext_->winSize;
    totalWinSize_ = 1000 * 1024 * 1024;  // RDMA 1000 MB空间
    shareMemOffset_ = totalWinSize_;
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - STATUS_SIZE_LAYERED;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;
    // 校验待完善
    /*
        uint64_t winSizeMin =
            moeExpertNum_ * axisBS_ * (axisH_ * sizeof(XType) + EXTRA_TOKEN_INFO_NUM * kAlign_ * sizeof(uint32_t)) +
            IPC_DATA_OFFSET + RDMA_DATA_SIZE;  // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
        assert(winContext_->winSize >= winSizeMin,
            "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
            epWorldSize:%u, epRankId:%u, moeExpertNum:%u, quantMode:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
            isQuant:%d, totalUbSize:%llu, expertTokenNumsType:%u\n",
            winContext_->winSize / MB_SIZE,
            winSizeMin / MB_SIZE,
            tilingData.moeDistributeDispatchInfo.epWorldSize,
            tilingData.moeDistributeDispatchInfo.epRankId,
            tilingData.moeDistributeDispatchInfo.moeExpertNum,
            tilingData.moeDistributeDispatchInfo.quantMode,
            tilingData.moeDistributeDispatchInfo.globalBs,
            tilingData.moeDistributeDispatchInfo.bs,
            tilingData.moeDistributeDispatchInfo.k,
            tilingData.moeDistributeDispatchInfo.h,
            tilingData.moeDistributeDispatchInfo.aivNum,
            tilingData.moeDistributeDispatchInfo.isQuant,
            tilingData.moeDistributeDispatchInfo.totalUbSize,
            tilingData.moeDistributeDispatchInfo.expertTokenNumsType);
    */
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_ +
            NOTIFY_OFFSET));
        shareAddrWins[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + NOTIFY_OFFSET +
            halfWinSize_ * bufferId_));
    }

    // struce相关信息初始化计算
    tokenStructLen_ =
        axisH_ * sizeof(ExpandXOutType) + INFO_NUM_IN_TOKENSTRUCK * (kAlign_ * sizeof(uint32_t));  // token和四元组大小
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);                                           // 纯token大小
    expLenInStruct_ = kAlign_ * sizeof(uint32_t);                                                  // topkId大小
    weightLenInStruct_ = kAlign_ * sizeof(uint32_t);                                               // weight大小
    cntLenInStruct_ = kAlign_ * sizeof(uint32_t);                                                  // tokenIdx大小
    realLenInStruct_ = axisK_ * sizeof(uint32_t);                 // 内存中实际有效部分，跟 axisK_ 有关
    expOffsetInStruct_ = tokenLenInStruct_;                       // 开始写topkId的起始位置
    weightOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_;  // 开始写weight的起始位置
    cntOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_;  // 开始写tokenIdx的起始位置
    scaleOffsetInStruct_ =
        tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + cntLenInStruct_;  // 开始写scales的起始位置
    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;

    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;  // 共享内存上每个server块的大小

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);
    PRINTF("[Init] rank:%d, blockId:%d, bufferId:%d \n", rankId_, aivId_, bufferId_);

    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;

    tokenServerIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerIdx);
    tokenServerCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerCnt);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    epRankTokenCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)epRankTokenCnt);
    srcOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)srcOffsetRankTokenIdx);
    dstOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstOffsetRankTokenIdx);

    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));

    weightsOutGt.SetGlobalBuffer((__gm__ float *)(expandScales));

    sendTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    readTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_));
    sendTokensU32Tensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + WIN_SIZE));
    readStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + WIN_SIZE));

    expertTokenNumsOutGM_ = expertTokenNumsOut;  // 无GlobalTensor
    epRecvCountsGM_ = epRecvCountsOut;           // 无GlobalTensor
    statusSpaceGm_ = windowInGM_ + WIN_SIZE;

    expandXGM_ = x;
    expandIdxGM_ = expertIds;
    weightsGM_ = expertScales;

    dataBatchWriteInfo_ = workspaceGM;
    dataBatchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint64_t *)(dataBatchWriteInfo_),
                                              serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK);

    expertToServerCntGM_ = dataBatchWriteInfo_ + serverNum * PER_MSG_RDMA_SEND_TIME * B64_PER_BLOCK * sizeof(uint64_t);
    expertToServerGlobalTensor_.SetGlobalBuffer((__gm__ uint32_t *)(expertToServerCntGM_),
                                                RoundUp(axisBS_ * serverNum, B32_PER_BLOCK));

    combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int32_t);
    combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t);
    combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;

    tpipe_->InitBuffer(batchWriteInfoBuf_, PER_MSG_RDMA_SEND_TIME * BW_ITEM_SIZE);  // 2 * 32

    batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
    batchWriteU32Tensor_ = batchWriteU64Tensor_.template ReinterpretCast<uint32_t>();

    tpipe_->InitBuffer(expertToServerCntsBuf_, RoundUp(static_cast<uint32_t>(axisBS_ * serverNum * sizeof(uint32_t)),
                                                       UB_32B_ALIGN));  // bs * rankSize / 8 * 4
    expertToServerCntTensor_ = expertToServerCntsBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(expertToServerCntTensor_, 0,
                        static_cast<uint32_t>(RoundUp(static_cast<uint32_t>(axisBS_ * serverNum), B32_PER_BLOCK)));

    tpipe_->InitBuffer(statusBuf_, UB_32B_ALIGN);  // 32

    tpipe_->InitBuffer(expertToServerIdxBuf_, serverNum * sizeof(uint32_t));  // rankSize / 8 * 4
    expertToServerIdxTensor_ = expertToServerIdxBuf_.Get<uint32_t>();

    tpipe_->InitBuffer(expertCountBuf_, moeExpertNum_ * sizeof(int32_t));  // moeNum * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertCountTensor_, 0, moeExpertNum_);

    tpipe_->InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);  // 176K

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    int32_t state = selfStatusTensor(aivId_ * UB_32B_ALIGN);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        sendStatusTensor_.SetValue(0, FLAG_VALUE);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            sendStatusTensor_);
    }

    LocalTensor<int32_t> tempLocal = tBuf.Get<int32_t>();

    // 每次调用magic++,用来区分不同轮次
    GlobalTensor<int32_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
                            aivId_ * EXP_TOKEN_COUNT_FLAG_CNT);
    tempLocal(0) = 1;
    // 使用atomic方式实现+1
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // 等待SetValue完成
    DataCopy(magicGt, tempLocal, EXP_TOKEN_COUNT_FLAG_CNT);
    AscendC::SetAtomicNone();
    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // 等待SetValue完成
    magicVal_ = magicGt.GetValue(0);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::Input2Win()
{
    uint32_t sendTokenNum = axisBS_ / aivNum_;
    uint32_t remainderTokenNum = axisBS_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    // 分核，每个Core处理sendTokenNum个Token的遍历
    if (aivId_ < remainderTokenNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    if (sendTokenNum == 0) {
        return;
    }
    int32_t expertId = 0;
    uint32_t dstServerId = 0;
    uint32_t tokenIndex = 0;

    uint32_t tokenUbSize = tokenStructLen_;
    if constexpr (DynamicQuant || StaticQuant) {
        tokenUbSize = axisH_ * sizeof(XType);
    }

    tpipe_->InitBuffer(tokenServerIdxBuf_, sendTokenNum * serverNum * sizeof(int32_t));
    tpipe_->InitBuffer(serverCountBuf_, serverNum * sizeof(int32_t));

    tokenServerIdxTensor_ = tokenServerIdxBuf_.Get<int32_t>();
    DataCopyExtParams tokenServerIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * serverNum * sizeof(int32_t)), 0U,
                                              0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(tokenServerIdxTensor_, tokenServerIdxGMTensor_[startTokenId * serverNum], tokenServerIdxParams,
                copyPadExtParams);

    // 这几个tensor是相同的地址空间，只是数据类型不一样
    LocalTensor<uint8_t> tokenTempTensorU8_ =
        tBuf.GetWithOffset<uint8_t>(((tokenUbSize) / sizeof(uint8_t)), TBUF_TEMP_OFFSET);
    LocalTensor<uint32_t> tokenTempTensorU32_ =
        tBuf.GetWithOffset<uint32_t>(((tokenUbSize) / sizeof(uint32_t)), TBUF_TEMP_OFFSET);
    LocalTensor<XType> tokenLt = tBuf.GetWithOffset<XType>(((tokenUbSize) / sizeof(XType)), TBUF_TEMP_OFFSET);

    GlobalTensor<uint8_t> xGMTensorU8_;
    xGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandXGM_);
    GlobalTensor<uint8_t> expertIdsGMTensorU8_;
    expertIdsGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandIdxGM_);

    GlobalTensor<uint32_t> expertIdsGMTensorU32_;
    expertIdsGMTensorU32_.SetGlobalBuffer((__gm__ uint32_t *)expandIdxGM_);

    GlobalTensor<uint8_t> weightGt;
    weightGt.SetGlobalBuffer((__gm__ uint8_t *)weightsGM_);

    DataCopyExtParams tokenCopyParamsQuant{1, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0, 0, 0};
    DataCopyExtParams tokenCopyParamsNoQuant{static_cast<uint16_t>(1), static_cast<uint16_t>(tokenLenInStruct_), 0, 0,
                                             0};
    DataCopyPadExtParams<uint8_t> tokenPadParams;

    DataCopyExtParams expCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(realLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> expPadParams;

    DataCopyExtParams weightCopyParams{static_cast<uint16_t>(1), static_cast<uint16_t>(realLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> weightPadParams;

    for (int i = 0; i < sendTokenNum; i++) {
        if constexpr (DynamicQuant || StaticQuant) {
            DataCopyPad(tokenTempTensorU8_, xGMTensorU8_[(startTokenId + i) * axisH_ * sizeof(XType)],
                        tokenCopyParamsQuant, tokenPadParams);
            LocalTensor<float> tokenCastLt = tBuf.GetWithOffset<float>(
                ((axisH_ * sizeof(float)) / sizeof(float)), RoundUp(TBUF_TEMP_OFFSET + tokenUbSize, B32_PER_BLOCK));
            QuantProcess(1, tokenLt, tokenCastLt);
        } else {
            DataCopyPad(tokenTempTensorU8_, xGMTensorU8_[(startTokenId + i) * tokenLenInStruct_],
                        tokenCopyParamsNoQuant, tokenPadParams);
        }
        // 拷贝topkIds 可省略
        DataCopyPad(tokenTempTensorU8_[expOffsetInStruct_], expertIdsGMTensorU8_[(startTokenId + i) * realLenInStruct_],
                    expCopyParams, expPadParams);

        // LocalTensor<int> exd =tokenTempTensorU8_[expOffsetInStruct_].template ReinterpretCast<int>();
        // AscendC::DumpTensor(exd, 475, 32);
        // 拷贝weight
        PRINTF("[Input2Win] rank:%d, coreId:%d, weightGt:%d \n", rankId_, aivId_, weightGt[(startTokenId + i) * realLenInStruct_].GetValue(0));
        DataCopyPad(tokenTempTensorU8_[weightOffsetInStruct_], weightGt[(startTokenId + i) * realLenInStruct_],
                    weightCopyParams, weightPadParams);

        // LocalTensor<float> weigt = tokenTempTensorU8_[weightOffsetInStruct_].template ReinterpretCast<float>();
        // AscendC::DumpTensor(weigt, 482, 32);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (int j = 0; j < serverNum; j++) {
            if (tokenServerIdxTensor_(i * serverNum + j) == -1) {
                continue;
            }
            uint32_t destOffset =
                j * SERVER_SIZE_ON_WIN + tokenStructLen_ * tokenServerIdxTensor_(i * serverNum + j) + TOKEN_COUNT_SIZE;
            // uint32_t destOffset =
            //     j * SERVER_SIZE_ON_WIN + tokenStructLen_ * tokenServerIdxTensor_(i * serverNum + j);
            DataCopy(sendTokensU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenStructLen_);

            GlobalTensor<int32_t> sendTokenU32;
            sendTokenU32.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_));
            AscendC::DumpTensor(sendTokenU32[(destOffset + expOffsetInStruct_) / 4], 495, 32);

            GlobalTensor<float> sendTokenU32_wt;
            sendTokenU32_wt.SetGlobalBuffer((__gm__ float *)(windowOutGM_));
            AscendC::DumpTensor(sendTokenU32_wt[(destOffset + weightOffsetInStruct_) / 4], 499, 32);

            if (j == rankId_ / SERVER_RANK_SIZE) {
                DataCopy(readTokensU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenStructLen_);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    // AscendC::DumpTensor(sendTokensU8Tensor_, 497, SERVER_SIZE_ON_WIN);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::QuantProcess(
    uint32_t sendTokenNum, LocalTensor<XType> xTokenLt, LocalTensor<float> tokenCastLt)
{
    constexpr uint32_t maxArrUbOffset = 6 * 1024;
    constexpr uint32_t maxArrLen = 3;
    constexpr uint32_t maxValOffset = 0;
    constexpr uint32_t minValOffset = 1;
    constexpr uint32_t resValOffset = 2;
    constexpr float quantMax = 127.0f;
    const half deqScale = static_cast<half>(1.000000e+00f);
    float dynamicScale = 0.0;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> workLt = tBuf.GetWithOffset<float>(maxArrUbOffset / sizeof(float), 0);
    LocalTensor<float> maxLt = tBuf.GetWithOffset<float>(maxArrLen, maxArrUbOffset);
    Cast(tokenCastLt, xTokenLt, RoundMode::CAST_NONE, sendTokenNum * axisH_);
    for (int32_t i = 0; i < sendTokenNum; ++i) {
        PipeBarrier<PIPE_V>();
        if constexpr (DynamicQuant) {
            ReduceMax(maxLt[maxValOffset], tokenCastLt[i * axisH_], workLt, axisH_, false);
            SyncFunc<AscendC::HardEvent::V_S>();
            PipeBarrier<PIPE_V>();
            ReduceMin(maxLt[minValOffset], tokenCastLt[i * axisH_], workLt, axisH_, false);
            PipeBarrier<PIPE_V>();
            Abs(maxLt, maxLt, maxArrLen - 1);
            PipeBarrier<PIPE_V>();
            ReduceMax(maxLt[resValOffset], maxLt, workLt, maxArrLen - 1, false);

            SyncFunc<AscendC::HardEvent::V_S>();
            float maxVal = maxLt(resValOffset);
            dynamicScale = float(quantMax) / float(maxVal);
            SyncFunc<AscendC::HardEvent::S_V>();
            Muls(tokenCastLt[i * axisH_], tokenCastLt[i * axisH_], dynamicScale, axisH_);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<half> halfLocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, tokenCastLt[i * axisH_], RoundMode::CAST_RINT, axisH_);
        PipeBarrier<PIPE_V>();
        SetDeqScale(deqScale);
        PipeBarrier<PIPE_V>();

        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXOutType> xOutTensor;
        LocalTensor<uint8_t> tokenUnitLt;
        tokenUnitLt = xTokenLt.template ReinterpretCast<uint8_t>();
        xOutTensor = tokenUnitLt[i * tokenStructLen_].template ReinterpretCast<ExpandXOutType>();
        Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

        LocalTensor<float> scaleTensor =
            tokenUnitLt[i * tokenStructLen_ + scaleOffsetInStruct_].template ReinterpretCast<float>();
        scaleTensor.SetValue(0, float(1.0) / dynamicScale);  // int8->float32
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::CopyTokenToWinOut(
    uint32_t localTokenIdx, uint32_t globalTokenIdx, uint32_t dstServerId)
{
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;
    uint32_t toServerCntSum = 0;
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIdx = 0; tokenIdx < globalTokenIdx; tokenIdx++) {
        uint32_t tensorOffset = tokenIdx * serverNum + dstServerId;
        toServerCntSum += expertToServerCntTensor_(tensorOffset);
    }

    LocalTensor<uint8_t> tokenTempTensorU8_ = tBuf.GetWithOffset<uint8_t>((DISPATCH_TOKEN_UB_SIZE), TBUF_TEMP_OFFSET);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    uint32_t destOffset = dstServerId * SERVER_SIZE_ON_WIN + tokenStructLen_ * toServerCntSum + TOKEN_COUNT_SIZE;
    DataCopy(sendTokensU8Tensor_[destOffset], tokenTempTensorU8_[localTokenIdx * tokenStructLen_], tokenStructLen_);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::WriteRdmaCntInfo()
{
    uint32_t destServerNum = serverNum / aivNum_;  // 每个AIV要处理的server数
    uint32_t remaServerNum = serverNum % aivNum_;
    uint32_t startServerId = destServerNum * aivId_;
    if (aivId_ < remaServerNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        destServerNum += 1;
        startServerId += aivId_;
    } else {
        startServerId += remaServerNum;
    }
    if (destServerNum == 0) {
        return;
    }

    tpipe_->InitBuffer(serverCountBuf_, serverNum * sizeof(int32_t));
    serverCountTensor_ = serverCountBuf_.Get<int32_t>();
    DataCopyExtParams serverCountParams = {1U, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(serverCountTensor_, tokenServerCntGMTensor_[0], serverCountParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstServerId = startServerId; dstServerId < startServerId + destServerNum; ++dstServerId) {
        uint32_t dstServerCnt = serverCountTensor_(dstServerId);
        expertToServerIdxTensor_(dstServerId) = dstServerCnt;
        LocalTensor<uint32_t> writeCntLt = tBuf.GetWithOffset<uint32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
        writeCntLt.SetValue(0, dstServerCnt);
        uint32_t destOffset = (dstServerId * SERVER_SIZE_ON_WIN) / sizeof(uint32_t);

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        // DataCopy(sendTokensU32Tensor_[destOffset], writeCntLt, EXP_TOKEN_COUNT_FLAG_CNT);
    }
}

// 构建发往其他server的所有data报文
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void
CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::ConstructDataAndFlagBatchWriteInfo()
{
    // 计算当前core要处理的server
    uint32_t batchWriteItemNum = serverNum / aivNum_;     // 一个aiv负责的server数量
    uint32_t remainderItemNum = serverNum % aivNum_;      // 多出来的server没人处理
    uint32_t startServerId = batchWriteItemNum * aivId_;  // 当前aiv负责[startServerId,endServerId)个server
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;    // 当前serverId

    if (aivId_ < remainderItemNum) {
        startServerId += aivId_;  // aiv0:1*0+0=0，aiv1:1*1+1=2，aiv2:1*2+2=4，... aiv23:1*23+23=46，
        batchWriteItemNum += 1;   // 前remainderItemNum个aiv需要多处理1个server的数据
    } else {
        startServerId += remainderItemNum;  // aiv24:1*24+24=48, aiv25:1*25+24=49
    }
    uint32_t endServerId = startServerId + batchWriteItemNum;
    if (batchWriteItemNum == 0) {
        return;
    }
    // 当前aiv负责 [startServerId,endServerId) 个 server
    for (uint32_t dstserverInd = startServerId; dstserverInd < endServerId; ++dstserverInd) {
        uint32_t sendIdx = dstserverInd - startServerId;
        uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + dstserverInd * SERVER_RANK_SIZE;  // 目标Rank
        PipeBarrier<PIPE_ALL>();
        uint64_t dstDataRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + NOTIFY_OFFSET +
                                              halfWinSize_ * bufferId_ + curServerId * SERVER_SIZE_ON_WIN);
        // src卡GetWindowsInAddr地址, 要发给serverIndex，即是本端的rdma地址
        uint64_t srcDataRdmaAddr =
            (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ + dstserverInd * SERVER_SIZE_ON_WIN);

        for (int j = 0; j < 16; ++j) {
            GlobalTensor<int32_t> sendTokenU32;
            sendTokenU32.SetGlobalBuffer((__gm__ int32_t *)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ + dstserverInd * SERVER_SIZE_ON_WIN + TOKEN_COUNT_SIZE));
            AscendC::DumpTensor(sendTokenU32[(expOffsetInStruct_) / 4], 658, 32);

            GlobalTensor<float> sendTokenU32_wt;
            sendTokenU32_wt.SetGlobalBuffer((__gm__ float *)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ + dstserverInd * SERVER_SIZE_ON_WIN + TOKEN_COUNT_SIZE));
            AscendC::DumpTensor(sendTokenU32_wt[(weightOffsetInStruct_) / 4], 662, 32);
        }

        // 去往该Server的传输的数据量
        uint32_t validTokenCount = expertToServerIdxTensor_(dstserverInd);
        PRINTF("[BatchWriteInfo] rank:%d, aivId_:%d, dstServer:%d, tokenCnt:%d\n", rankId_, aivId_, dstserverInd,
               validTokenCount);
        uint32_t validDataLength = TOKEN_COUNT_SIZE + validTokenCount * tokenStructLen_;
        // uint32_t validDataLength = validTokenCount * tokenStructLen_;
        uint64_t winInAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_) + NOTIFY_OFFSET);
        uint64_t winOutAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_));
        PipeBarrier<PIPE_ALL>();
        batchWriteU64Tensor_(0) = srcDataRdmaAddr;  // 源地址
        batchWriteU64Tensor_(1) = dstDataRdmaAddr;  // 目的地址
        batchWriteU64Tensor_(2) = validDataLength;  // 数据长度
        batchWriteU32Tensor_(6) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(7) = dstRankId;  // dst卡

        uint64_t dstFlagRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + NOTIFY_OFFSET +
                                              halfWinSize_ * bufferId_ + WIN_SIZE + curServerId * STATE_OFFSET);

        // src卡，即是本端的rdma地址
        uint64_t srcFlagRdmaAddr = (uint64_t)(sendStatusTensor_.GetPhyAddr());
        uint32_t flagLen = TOKEN_COUNT_SIZE;
        PipeBarrier<PIPE_ALL>();
        batchWriteU64Tensor_(4) = srcFlagRdmaAddr;  // 源地址
        batchWriteU64Tensor_(5) = dstFlagRdmaAddr;  // 目的地址
        batchWriteU64Tensor_(6) = flagLen;          // 数据长度
        batchWriteU32Tensor_(14) = HcclDataType::HCCL_DATA_TYPE_INT8;
        batchWriteU32Tensor_(15) = dstRankId;  // dst卡

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstServerOffset = dstserverInd;
        uint32_t sendInfoCount = B64_PER_BLOCK * PER_MSG_RDMA_SEND_TIME;
        DataCopy(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount], batchWriteU64Tensor_, sendInfoCount);
    }
}

// 机间同平面RDMA通信
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::DispatchBetweenServer()
{
    ConstructDataAndFlagBatchWriteInfo();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if ASCEND_IS_AIV {
        if (aivId_ == 0) {
            HcclHandle batchWriteResultData = hccl_.BatchWrite<true>((GM_ADDR)(dataBatchWriteInfoTensor_.GetPhyAddr()),
                                                                     serverNum * PER_MSG_RDMA_SEND_TIME);
            bufferChosenGlobal_(0) = bufferId_ ^ 1;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                bufferChosenGlobal_);
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline uint32_t
CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::GetExpRank(uint32_t expertId)
{
    return expertId / localMoeExpertNum_;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline bool
CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::IsInSameServer(uint32_t targetRankId)
{
    return targetRankId / SERVER_RANK_SIZE == rankId_ / SERVER_RANK_SIZE;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline int64_t
CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::MergeMagicWithValue(int32_t magic, int32_t value)
{
    return (static_cast<int64_t>(magic) << 32) | static_cast<int64_t>(value);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::SetIpcFlag(int32_t flagVal)
{
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int64_t> globalSet;
    globalSet.SetGlobalBuffer((__gm__ int64_t *)(shareAddrs[destRankIdx]) + localRankId * B64_PER_BLOCK);
    LocalTensor<int64_t> localSet = tBuf.GetWithOffset<int64_t>(B64_PER_BLOCK, 0);
    int64_t setVal = MergeMagicWithValue(magicVal_, flagVal);
    localSet.SetValue(0, setVal);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, B64_PER_BLOCK);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::WaitIpcFlag(int32_t flagVal)
{
    int64_t waitVal = MergeMagicWithValue(magicVal_, flagVal);
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    LocalTensor<int64_t> localWait = tBuf.GetWithOffset<int64_t>(B64_PER_BLOCK, 0);
    bool isSync = true;
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int64_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ int64_t *)(shareAddrs[localRankId]) + destRankIdx * B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
    do {
        DataCopy(localWait, flagIpcGt, B64_PER_BLOCK);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        // 当有core未达到checkValue的阶段时，继续等待
        int64_t tempVal = localWait.GetValue(0);
        if (tempVal >= waitVal) {
            break;
        }
    } while (isSync);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void
CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::SetTokenCnt(GlobalTensor<int32_t> globalSet)
{
    AscendC::SetAtomicAdd<int32_t>();
    LocalTensor<int32_t> localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    localSet(0) = 1;  // AtomicAdd每次+1
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::WaitWindow()
{
    // 前ServerNum个卡进行等待，等待本server的也保留
    if (aivId_ >= serverNum) {
        return;
    }
    uint32_t waitFlagIdx = aivId_;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
    while (true) {
        DataCopy(statusTensor, readStatusTensor_[(waitFlagIdx)*STATE_OFFSET / sizeof(int32_t)], FLAG_U32_CNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        int32_t sumOfFlag = statusTensor.GetValue(0);
        if (sumOfFlag == FLAG_VALUE) {
            break;
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::Win2Ipc()
{
    uint32_t coresPerServer = (aivNum_ - serverNum) / serverNum;  // 48/2 = 24
    if (aivId_ >= coresPerServer * serverNum) {
        return;
    }
    // 计算本core需要处理的ServerId
    uint32_t formServerId = aivId_ / coresPerServer;  // 前24处理0， 后24处理1

    // 获取tokenCnt,计算本卡收到对端server多少Token，用于后续分核计算
    __gm__ uint8_t *tokenCntGlobalAddr;
    if (formServerId == rankId_ / SERVER_RANK_SIZE) {
        tokenCntGlobalAddr = (__gm__ uint8_t *)(windowOutGM_) + formServerId * SERVER_SIZE_ON_WIN;
    } else {
        tokenCntGlobalAddr = (__gm__ uint8_t *)(windowInGM_) + formServerId * SERVER_SIZE_ON_WIN;
    }
    GlobalTensor<uint32_t> tokenCntGlobalTensor;
    tokenCntGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)(tokenCntGlobalAddr));
    LocalTensor<uint32_t> localWait = tBuf.GetWithOffset<uint32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);

    DataCopy(localWait, tokenCntGlobalTensor, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t tokenCnt = localWait.GetValue(0);

    GlobalTensor<uint8_t> targetTokenIpcGt;  // 对端IPC的TokenTensor，写数据用

    uint32_t WinInTokenOffset = formServerId * SERVER_SIZE_ON_WIN + TOKEN_COUNT_SIZE;
    uint32_t localAivId = aivId_ % coresPerServer;  // 0,1，2,3...19
    // 平均每个核处理多少token
    uint32_t tokenCntPerAiv = tokenCnt / coresPerServer;  // 16/20
    // 平分后剩下多少token
    uint32_t tokenCntRemain = tokenCnt % coresPerServer;  // 16%20
    // 前面的核共分到了多少剩余
    uint32_t tokenCntPreRemain = (localAivId < tokenCntRemain) ? localAivId : tokenCntRemain;  // 小于16为
    // 当前核分到多少token
    uint32_t tokenCntCurAiv = (localAivId < tokenCntRemain) ? (tokenCntPerAiv + 1) : tokenCntPerAiv;

    LocalTensor<uint8_t> localUB =
        tBuf.GetWithOffset<uint8_t>(DISPATCH_TOKEN_UB_SIZE / sizeof(uint8_t), TBUF_TEMP_OFFSET);
    uint32_t tokenCntInUB = DISPATCH_TOKEN_UB_SIZE / tokenStructLen_;
    // ceil div
    uint32_t batchCnt = (tokenCntCurAiv + tokenCntInUB - 1) / tokenCntInUB;
    for (uint32_t batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
        uint32_t tokenCntInBatch = tokenCntInUB;
        if (batchIdx == batchCnt - 1) {
            tokenCntInBatch = tokenCntCurAiv - (batchCnt - 1) * tokenCntInUB;
        }
        // 计算当前Core处理的Token偏移
        uint32_t tokenStruceIdx = localAivId * tokenCntPerAiv + tokenCntPreRemain + batchIdx * tokenCntInUB;
        // 等待GM->UB
        if (formServerId == rankId_ / SERVER_RANK_SIZE) {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopy(localUB, sendTokensU8Tensor_[WinInTokenOffset + tokenStruceIdx * tokenStructLen_],
                     tokenCntInBatch * tokenStructLen_);
        } else {
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopy(localUB, readTokensU8Tensor_[WinInTokenOffset + tokenStruceIdx * tokenStructLen_],
                     tokenCntInBatch * tokenStructLen_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t tokenIdx = 0; tokenIdx < tokenCntInBatch; ++tokenIdx) {
            // 逐个处理Token to Ipc
            uint32_t expPos = tokenIdx * tokenStructLen_ + expOffsetInStruct_;
            LocalTensor<uint32_t> expInfoTensor = localUB[expPos].ReinterpretCast<uint32_t>();
            // 当前Token的ExpIds信息
            uint32_t tokenCntPos = tokenIdx * tokenStructLen_ + cntOffsetInStruct_;
            LocalTensor<uint32_t> cntInfoTensor = localUB[tokenCntPos].ReinterpretCast<uint32_t>();
            // 当前Token的Cnt信息
            for (uint32_t expIdx = 0; expIdx < axisK_; ++expIdx) {
                uint32_t targetexpertId = expInfoTensor[expIdx].GetValue(0);
                uint32_t targetRankId = GetExpRank(targetexpertId);
                if (!IsInSameServer(targetRankId)) {
                    continue;
                }
                uint32_t tokenPosInBlock = cntInfoTensor(expIdx);
                PipeBarrier<PIPE_ALL>();
                // 在IPC的当前Block中，前面还有tokenPosInBlock个Token
                uint32_t targetExpOffset = (targetexpertId % localMoeExpertNum_) * worldSize_ * RANK_SIZE_ON_IPC;
                // 第几个Exp段
                uint32_t targetServerOffset = formServerId * SERVER_RANK_SIZE * RANK_SIZE_ON_IPC;
                // 第几个Server段
                uint32_t targetRankOffset = (rankId_ % SERVER_RANK_SIZE) * RANK_SIZE_ON_IPC;
                // 第几个Rank段
                uint32_t targetTokenOffset = tokenPosInBlock * tokenStructLen_;  // 第几个Token位
                uint32_t targetOffset = targetExpOffset + targetServerOffset + targetRankOffset + targetTokenOffset;

                targetTokenIpcGt.SetGlobalBuffer(
                    (__gm__ uint8_t *)(shareAddrs[targetRankId % SERVER_RANK_SIZE] + IPC_DATA_OFFSET + targetOffset));
                PipeBarrier<PIPE_ALL>();
                DataCopy(targetTokenIpcGt, localUB[tokenIdx * tokenStructLen_], tokenStructLen_);
                // 对应token个数加1
                GlobalTensor<int32_t> targetCntIpcGt;  // 对端IPC的CntTensor，统计对端收到的次数
                targetCntIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[targetRankId % SERVER_RANK_SIZE] +
                                                                  IPC_TOKEN_CNT_OFFSET));  // 前面记录有几个token
                uint32_t setTokenCntOffset = (targetexpertId % localMoeExpertNum_) * worldSize_ +
                                             formServerId * SERVER_RANK_SIZE + (rankId_ % SERVER_RANK_SIZE);
                SetTokenCnt(targetCntIpcGt[EXP_TOKEN_COUNT_FLAG_CNT * setTokenCntOffset]);
            }
        }
    }
}

// 每个专家从不同的server块取数据
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::Ipc2Out()
{
    uint32_t curRankExpertStart = rankId_ * localMoeExpertNum_;               // 9*8=72
    uint32_t curRankExpertEnd = curRankExpertStart + localMoeExpertNum_ - 1;  // 72+8-1=79

    for (int i =0 ; i < serverNum; ++i) {
        GlobalTensor<uint8_t> srcIpcU; 
        srcIpcU.SetGlobalBuffer((__gm__ uint8_t *)(shareAddrWins[rankId_]) + i * SERVER_SIZE_ON_WIN);
        
        for (int j = 0; j < 16; ++j) {
            GlobalTensor<int32_t> sendTokenU32;
            sendTokenU32.SetGlobalBuffer((__gm__ int32_t *)((shareAddrWins[rankId_]) + i * SERVER_SIZE_ON_WIN + j * tokenStructLen_ + TOKEN_COUNT_SIZE));
            AscendC::DumpTensor(sendTokenU32[(expOffsetInStruct_) / 4], 920, 32);

            GlobalTensor<float> sendTokenU32_wt;
            sendTokenU32_wt.SetGlobalBuffer((__gm__ float *)((shareAddrWins[rankId_]) + i * SERVER_SIZE_ON_WIN + j * tokenStructLen_ + TOKEN_COUNT_SIZE));
            AscendC::DumpTensor(sendTokenU32_wt[(weightOffsetInStruct_) / 4], 924, 32);
        }
    }

    for (uint32_t srcRank = 0; srcRank < worldSize_; ++srcRank) {
        uint32_t localRankIdx = srcRank % SERVER_RANK_SIZE;  // 20%8=4  server上的序号4rank，即第5个
        uint32_t curServerIdx = rankId_ / SERVER_RANK_SIZE;  // 9/8=1 server1,即第2个
        uint32_t targetRankId =
            localRankIdx + curServerIdx * SERVER_RANK_SIZE;       // 4+1*8=12 当前server上的rank，全局rankid=12
        uint32_t tarServerBlockIdx = srcRank / SERVER_RANK_SIZE;  // 20/8=2  目标rank上的block序号2，即第3块

        GlobalTensor<uint8_t> srcIpcGt;  // TODO: 取地址可能有问题，需要为 targetRankId 的 tarServerBlockIdx 的地址
        srcIpcGt.SetGlobalBuffer((__gm__ uint8_t *)(shareAddrWins[localRankIdx]) +
                                 tarServerBlockIdx * SERVER_SIZE_ON_WIN + TOKEN_COUNT_SIZE);
        // srcIpcGt.SetGlobalBuffer((__gm__ uint8_t *)(shareAddrWins[localRankIdx]) +
        //                          tarServerBlockIdx * SERVER_SIZE_ON_WIN);

        for (uint32_t recvExpId = curRankExpertStart; recvExpId <= curRankExpertEnd; ++recvExpId) {
            int recvTokenCnt = epRankTokenCntGMTensor_.GetValue(recvExpId * worldSize_ +
                                                                srcRank);  // 专家recvExpId从srcRank收的token个数
            uint32_t beginIndex = 0;
            uint32_t endIndex = 0;
            // 分核处理token数量

            uint32_t tokenCntPerAiv = recvTokenCnt / aivNum_;
            uint32_t remainTokenNum = recvTokenCnt % aivNum_;
            beginIndex = tokenCntPerAiv * aivId_;
            if (aivId_ < remainTokenNum) {
                tokenCntPerAiv++;
                beginIndex += aivId_;
            } else {
                beginIndex += remainTokenNum;
            }
            endIndex = beginIndex + tokenCntPerAiv;
            if (beginIndex >= recvTokenCnt) {
                continue;
            }
            LocalTensor<uint8_t> localUB = tBuf.GetWithOffset<uint8_t>(
                (DISPATCH_TOKEN_UB_SIZE - TBUF_TEMP_OFFSET) / sizeof(uint8_t), TBUF_TEMP_OFFSET);

            DataCopyExtParams copyParams{1, static_cast<uint32_t>(tokenStructLen_), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams;
            DataCopyPadExtParams<ExpandXOutType> tokenExtParams{false, 0U, 0U, 0U};
            DataCopyExtParams weightParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<uint32_t> weightExtParams{false, 0U, 0U, 0U};
            DataCopyExtParams scalesParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<uint32_t> scalesExtParams{false, 0U, 0U, 0U};

            for (int i = beginIndex; i < endIndex; ++i) {
                // 假设当前shape为[expertNum, rank, maxBs]，专家recvExpId从srcRank读取第i个token的src与dst
                int32_t srcOffset =
                    srcOffsetRankTokenIdxGMTensor_.GetValue(recvExpId * worldSize_ * BS_UPPER + srcRank * BS_UPPER + i);
                int32_t dstOffset =
                    dstOffsetRankTokenIdxGMTensor_.GetValue(recvExpId * worldSize_ * BS_UPPER + srcRank * BS_UPPER + i);

                uint32_t tokenOffset =
                    (tokenStructLen_ * srcOffset);  // 包含token, 以及token后的信息:expIds, weights, tokenIdx, scales

                DataCopyPad(localUB, srcIpcGt[tokenOffset], copyParams, padParams);  // winIn --> local
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                LocalTensor<ExpandXOutType> tokenLt = localUB.ReinterpretCast<ExpandXOutType>();
                DataCopyExtParams tokenParams{1, static_cast<uint32_t>(tokenLenInStruct_), 0, 0, 0};
                DataCopyPad(expandXOutGMTensor_[dstOffset], tokenLt, tokenParams);  // local --> out

                LocalTensor<int> expLt = localUB[expOffsetInStruct_].ReinterpretCast<int>();
                int index;
                for (int j = 0; j < axisK_; j++) {
                    PRINTF("[Ipc2Out] rank:%d, aivId_:%d, topk:%d\n", rankId_, aivId_, expLt.GetValue(j));

                    if (expLt.GetValue(j) == recvExpId) {
                        index = j;
                    }
                }
                // weight to output
                LocalTensor<float> weightLt = localUB[weightOffsetInStruct_].ReinterpretCast<float>();
                float weightVal = weightLt.GetValue(index);

                PRINTF("[Ipc2Out] rank:%d, aivId_:%d, curRankExpertStart:%d, curRankExpertEnd:%d, \
                    localRankIdx:%d, curServerIdx:%d, targetRankId:%d, tarServerBlockIdx:%d, recvTokenCnt:%d, \
                    i:%d, recvExpId:%d, srcRank:%d, srcOffset:%d, dstOffset:%d, tokenOffset:%d, weightVal:%f\n", 
                    rankId_, aivId_, curRankExpertStart, curRankExpertEnd, localRankIdx, curServerIdx, targetRankId,
                    tarServerBlockIdx, recvTokenCnt, i, recvExpId, srcRank, srcOffset, dstOffset, tokenOffset, weightVal);


                // weightsOutGt[dstOffset].SetValue(0, weightVal);
                // DataCopyPad(weightsOutGt[dstOffset], weightLt, weightParams);  // local --> out

                // dynamic scales to output
                if constexpr (DynamicQuant) {
                    LocalTensor<float> quantTempUB = localUB[scaleOffsetInStruct_].ReinterpretCast<float>();
                    DataCopyPad(dynamicScalesOutGMTensor_[dstOffset], quantTempUB, scalesParams);
                }
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::CleanUp()  // 清除status
{
    uint32_t cleanBuffSize = worldSize_ * localMoeExpertNum_ * TOKEN_COUNT_SIZE;
    if (cleanBuffSize < STATE_OFFSET * serverNum) {
        cleanBuffSize = STATE_OFFSET * serverNum;
    }
    LocalTensor<int32_t> cleanTempLt_ = tBuf.GetWithOffset<int32_t>(cleanBuffSize / sizeof(int32_t), TBUF_TEMP_OFFSET);
    GlobalTensor<int32_t> flagIpcGt;
    Duplicate<int32_t>(cleanTempLt_, 0, cleanBuffSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE]));
    PipeBarrier<PIPE_ALL>();
    DataCopy(readStatusTensor_, cleanTempLt_, cleanBuffSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    DataCopy(flagIpcGt[IPC_TOKEN_CNT_OFFSET / sizeof(int32_t)], cleanTempLt_, cleanBuffSize / sizeof(int32_t));
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void CamMoeDistributeDispatchA2Layered<TemplateMC2TypeA2layeredFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        PRINTF("[A2layer Process blockIdx %d]\n", aivId_);
        Input2Win();
        PRINTF("[A2layer Input2Win blockIdx %d]\n", aivId_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        PRINTF("[A2layer b4WriteRdmaCntInfo blockIdx %d]\n", aivId_);
        WriteRdmaCntInfo();
        PRINTF("[A2layer b4DispatchBetweenServer blockIdx %d]\n", aivId_);
        DispatchBetweenServer();
        PRINTF("[A2layer b4WaitWindow blockIdx %d]\n", aivId_);
        WaitWindow();
        PRINTF("[A2layer AfterWaitWindow blockIdx %d]\n", aivId_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        PRINTF("[A2layer Win2Ipc blockIdx %d]\n", aivId_);
        // 最后serverNum个核不参与Win2Ipc，只进行reduceInfo计算
        if (aivId_ < aivNum_ - serverNum) {
            // Win2Ipc();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        
        PRINTF("[A2layer b4SetIpcFlag blockIdx %d]\n", aivId_);
        SetIpcFlag(IPC_FLAG_STEP_1);
        PRINTF("[A2layer b4WaitIpcFlag blockIdx %d]\n", aivId_);
        WaitIpcFlag(IPC_FLAG_STEP_1);
        PRINTF("[A2layer AfterWaitIpcFlag blockIdx %d]\n", aivId_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        PRINTF("[A2layer b4Ipc2Out blockIdx %d]\n", aivId_);
        Ipc2Out();
        PRINTF("[A2layer AfterIpc2Out blockIdx %d]\n", aivId_);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        
        PRINTF("[A2layer b4CleanUp blockIdx %d]\n", aivId_);
        if (aivId_ == 0) {
            CleanUp();
        }
        PRINTF("[A2layer AfterCleanUp blockIdx %d]\n", aivId_);
        PipeBarrier<PIPE_ALL>();
        SetIpcFlag(IPC_FLAG_STEP_2);  // 为何同步？
        WaitIpcFlag(IPC_FLAG_STEP_2);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
