#ifndef MOE_DISTRIBUTE_DISPATCH_A2_PIPELINE_H
#define MOE_DISTRIBUTE_DISPATCH_A2_PIPELINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../cam_moe_distribute_dispatch_tiling.h"
#include "../moe_distribute_base.h"
#include "../comm_args.h"

namespace MoeDistributeDispatchA2Impl {
constexpr uint32_t STATE_OFFSET = 512;                 // 状态空间偏移地址
constexpr uint32_t STATUS_SIZE_LAYERED = 1024 * 1024;  // 1M
constexpr uint32_t HCCS_RING_BUFFER_HEAD_TAIL = 8 * 2 * 32;
constexpr uint32_t EACH_HCCS_RING_BUFFER_HEAD_TAIL = 2 * 32;
constexpr uint32_t RING_BUFFER_HEAD_TAIL = 8 * 32;
constexpr uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
constexpr uint32_t SELF_STATE_OFFSET = 512 * 1024;  // 本卡状态空间偏移地址
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t INFO_NUM_IN_TOKENSTRUCK = 4;  // 在Token后加入3种信息:expIds, weights, tokenIdx, dstRank, scales;
constexpr uint32_t B64_PER_BLOCK = 4;
constexpr uint32_t PER_MSG_RDMA_SEND_TIME = 2;
constexpr uint32_t B32_PER_BLOCK = 8;
constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 176 * 1024;
constexpr uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 64 * 32;
constexpr uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
constexpr uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
constexpr uint32_t NOTIFY_OFFSET = 0 * 1024 * 1024;
constexpr uint32_t IPC_BUFF_ALIGN = 512;
constexpr uint32_t TOKEN_COUNT_SIZE = 32;
constexpr uint32_t MAX_SERVER_NUM = 32;
constexpr uint32_t RDMA_CHUNK = 32;
constexpr uint32_t FLAG_U32_CNT = TOKEN_COUNT_SIZE / 4;
constexpr int32_t IPC_FLAG_STEP_1 = 1ULL;
constexpr int32_t IPC_FLAG_STEP_2 = 2ULL;
constexpr uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
constexpr uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
constexpr uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;  // normal/low_latency dispatch&combine的预留大小一致
constexpr uint32_t EXTRA_TOKEN_INFO_NUM = 4U;              // 专家信息 权重信息 量化Scale 到达标志位
constexpr uint32_t BITS32_PER_BLOCK = 8U;
constexpr static uint32_t BW_ITEM_SIZE = 32;
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint32_t BS_UPPER = 4096;
constexpr uint32_t RDMA_SENDER = 1;
constexpr uint32_t RDMA_HCCS_FORWARDER = 2;
constexpr uint32_t FORWARDER_COORDINATOR = 3;
constexpr uint32_t HCCS_RECEIVER = 4;
constexpr uint32_t RDMA_COORDINATOR = 5;

#define TemplateMC2TypeA2PipelineClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2PipelineFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
using namespace Cam;
template <TemplateMC2TypeA2PipelineClass>
class MoeDistributeDispatchA2Pipeline
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
    __aicore__ inline MoeDistributeDispatchA2Pipeline(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales,
                                GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt, GM_ADDR epRankTokenCnt,
                                GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR tokenIdxPerExpert,GM_ADDR expandXOut,
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
    __aicore__ inline void CoreRoleAssign();
    __aicore__ inline void PrepareRdmaSend();
    __aicore__ inline void TriggerRdmaSend();
    __aicore__ inline void Rdma2HCCS();
    __aicore__ inline void CreditRecycle();
    __aicore__ inline void HCCS2Out();
    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGt;
    GlobalTensor<uint64_t> dataBatchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint8_t> readTokensU8Tensor_;
    GlobalTensor<uint8_t> sendTokensU8Tensor_;
    GlobalTensor<uint8_t> rdmaSendU8Tensor_;
    GlobalTensor<uint8_t> rdmaRecvU8Tensor_;
    GlobalTensor<uint8_t> hccsRecvU8Tensor_;
    GlobalTensor<uint32_t> sendTokensU32Tensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<uint32_t> expertToServerGlobalTensor_;
    GlobalTensor<int32_t> readStatusTensor_;
    GlobalTensor<int32_t> tokenServerIdxGMTensor_;
    GlobalTensor<int32_t> tokenServerCntGMTensor_;
    GlobalTensor<uint8_t> rdmaSendRingU8Tensor_;
    GlobalTensor<uint8_t> rdmaRecvRingU8Tensor_;
    GlobalTensor<uint8_t> hccsRecvRingU8Tensor_;
    GlobalTensor<uint32_t> rdmaHeadTailTensor_;
    GlobalTensor<uint32_t> hccsHeadTailTensor_;

    GlobalTensor<int32_t> epRankTokenCntGMTensor_;
    GlobalTensor<int32_t> srcOffsetRankTokenIdxGMTensor_;
    GlobalTensor<int32_t> dstOffsetRankTokenIdxGMTensor_;
    GlobalTensor<int32_t> tokenIdxPerExpertGMTensor_;
    GlobalTensor<int32_t> tokenPerRankGMTensor_;

    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertToServerCntTensor_;
    LocalTensor<uint32_t> expertToServerIdxTensor_;

    LocalTensor<int32_t> tokenServerIdxTensor_;
    LocalTensor<int32_t> serverCountTensor_;
    LocalTensor<uint8_t> tokenStructInRdmaTensor_;
    LocalTensor<uint8_t> tokenStructInHccsTensor_;
    LocalTensor<uint8_t> rdmaUseTokenStructInHccsTensor_;

    TBuf<> tokenServerIdxBuf_;
    TBuf<> serverCountBuf_;

    TBuf<> expertCountBuf_;
    TBuf<> statusBuf_;
    TBuf<> batchWriteInfoBuf_;
    TBuf<> expertToServerCntsBuf_;  // 总表，int类型只写1/0
    TBuf<> expertToServerIdxBuf_;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<> weightBuf_;
    TBuf<> tokenStructInRdmaBuf_;
    TBuf<> tokenStructInHccsBuf_;
    TBuf<> rdmaUseTokenStructInHccsBuf_;

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
    GM_ADDR hccsHeadTailGM[8];

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
    uint32_t aivRole_{0};
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
    uint32_t rdmaItemNum{0};
    uint32_t hccsItemNum{0};
    uint32_t expertTokenNumsType_{0};
    uint32_t shareMemOffset_{0};
    uint32_t rdmaSendHead[MAX_SERVER_NUM] = {0};
    uint32_t rdmaSendTail[MAX_SERVER_NUM] = {0};
    uint32_t rdmaRecvHead[MAX_SERVER_NUM] = {0};
    uint32_t rdmaRecvTail[MAX_SERVER_NUM] = {0};
    uint32_t hccsRecvHead[MAX_SERVER_NUM] = {0};
    uint32_t hccsRecvTail[MAX_SERVER_NUM] = {0};
    uint32_t senderNum{0};
    uint32_t triggerNum{0};
    // TokenStruck相关
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t srcRankInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    uint64_t magicVal_{0};
    //当前server处理的专家范围
    uint32_t expertIdxStart_{0};
    uint32_t expertIdxEnd_{0};
    uint32_t combineInnerCntOffset;
    uint32_t combineInnerCntIndexOffset;
    uint32_t combineOuterCntOffset;
    uint32_t combineOuterCntIndexOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expertScales, GM_ADDR tokenServerIdx, GM_ADDR tokenServerCnt,
    GM_ADDR epRankTokenCnt, GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx, GM_ADDR tokenIdxPerExpert, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
    GM_ADDR expandScales, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
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
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    globalBs_ = tilingData.moeDistributeDispatchInfo.globalBs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;
    kAlign_ = RoundUp(axisK_, (uint32_t)8);
    totalSize_ = winContext_->winSize;
    totalWinSize_ = RDMA_DATA_SIZE;  // RDMA 800 MB空间, 与low_latency一致
    shareMemOffset_ = totalWinSize_;
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - STATUS_SIZE_LAYERED - serverNum * RING_BUFFER_HEAD_TAIL;
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;  // 共享内存上每个server块的大小

    // struce相关信息初始化计算
    tokenStructLen_ =
        axisH_ * sizeof(ExpandXOutType) + INFO_NUM_IN_TOKENSTRUCK * (kAlign_ * sizeof(uint32_t));  // token和五元组大小
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);                                           // 纯token大小
    expLenInStruct_ = kAlign_ * sizeof(uint32_t);                                                  // topkId大小
    weightLenInStruct_ = kAlign_ * sizeof(uint32_t);                                               // weight大小
    cntLenInStruct_ = kAlign_ * sizeof(uint32_t);                                                  // tokenIdx大小
    realLenInStruct_ = axisK_ * sizeof(uint32_t);                 // 内存中实际有效部分，跟 axisK_ 有关
    expOffsetInStruct_ = tokenLenInStruct_;                       // 开始写topkId的起始位置
    weightOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_;  // 开始写weight的起始位置
    cntOffsetInStruct_ = tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_;  // 开始写tokenIdx的起始位置
    scaleOffsetInStruct_ =
        tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + cntLenInStruct_ ;  // 开始写scales的起始位置
    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    rdmaItemNum = (SERVER_SIZE_ON_WIN / 2 - TOKEN_COUNT_SIZE) / tokenStructLen_;
    hccsItemNum = (SERVER_SIZE_ON_WIN / 2 - TOKEN_COUNT_SIZE) / tokenStructLen_;

    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;

    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_ +
            NOTIFY_OFFSET));
        shareAddrWins[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + NOTIFY_OFFSET +
            halfWinSize_ * bufferId_));
    }
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;

    tokenServerIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerIdx);
    tokenServerCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenServerCnt);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    epRankTokenCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)epRankTokenCnt);
    srcOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)srcOffsetRankTokenIdx);
    dstOffsetRankTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstOffsetRankTokenIdx);
    tokenIdxPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenIdxPerExpert);

    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));

    weightsOutGt.SetGlobalBuffer((__gm__ float *)(expandScales));

    sendTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    rdmaSendRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    rdmaRecvRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_));
    hccsRecvRingU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_ + halfWinSize_ / 2));
    sendTokensU32Tensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + WIN_SIZE));
    readStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + WIN_SIZE));
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        hccsHeadTailGM[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + halfWinSize_ - 
                                                        HCCS_RING_BUFFER_HEAD_TAIL));
    }
    // hccsHeadTailTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + halfWinSize_ - 
    //                                                     HCCS_RING_BUFFER_HEAD_TAIL));
    rdmaHeadTailTensor_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + halfWinSize_ - HCCS_RING_BUFFER_HEAD_TAIL - 
                                                        RING_BUFFER_HEAD_TAIL * serverNum));

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

    tpipe_->InitBuffer(statusBuf_, UB_32B_ALIGN);  // 32

    tpipe_->InitBuffer(expertToServerIdxBuf_, serverNum * sizeof(uint32_t));  // rankSize / 8 * 4
    expertToServerIdxTensor_ = expertToServerIdxBuf_.Get<uint32_t>();

    tpipe_->InitBuffer(tokenStructInRdmaBuf_, tokenLenInStruct_);
    tokenStructInRdmaTensor_ = tokenStructInRdmaBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(tokenStructInHccsBuf_, tokenLenInStruct_);
    tokenStructInHccsTensor_ = tokenStructInHccsBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(rdmaUseTokenStructInHccsBuf_, tokenLenInStruct_);
    rdmaUseTokenStructInHccsTensor_ = rdmaUseTokenStructInHccsBuf_.Get<uint8_t>();

    tpipe_->InitBuffer(expertCountBuf_, moeExpertNum_ * sizeof(int32_t));  // moeNum * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertCountTensor_, 0, moeExpertNum_);

    tpipe_->InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);  // 176K
    tpipe_->InitBuffer(weightBuf_, UB_32B_ALIGN);      // 32

    CoreRoleAssign();

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    int32_t state = selfStatusTensor(aivId_ * UB_32B_ALIGN);
    PipeBarrier<PIPE_ALL>();

    if (aivId_ == 0) {
        sendStatusTensor_.SetValue(0, FLAG_VALUE);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            sendStatusTensor_);
    }

    LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();

    // 每次调用magic++,用来区分不同轮次
    GlobalTensor<uint64_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ uint64_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
                            aivId_ * UB_32B_ALIGN / sizeof(uint64_t));
    DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    tempLocal(0) += 1ULL;
    magicVal_ = tempLocal(0);
    DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}

// 分配各个核的角色，各个角色负责不同的任务
template <TemplateMC2TypeA2PipelineClass>
__aicore__ void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::CoreRoleAssign()
{
    senderNum = aivNum_ / 4 - 1;
    aivRole_ = aivNum_ / senderNum + 1;
    triggerNum = aivNum_ - 4 * senderNum;
}

// 由RDMA_SENDER执行，负责将要发送的token数据进行筛选打包，装载到rdmaSendRingU8Tensor这个环形buffer上，并更新headTailTensor中环形buffer头和尾的数据
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::PrepareRdmaSend()
{
    if (aivRole_ != RDMA_SENDER) {
        return ;
    }
    uint32_t localIndex = aivId_ % senderNum;
    if (localIndex >= serverNum) {
        return ;
    }
    uint32_t sendTokenNum = axisBS_;
    uint32_t startTokenId = 0;
    uint32_t endTokenId = axisBS_;
    int32_t expertId = 0;
    uint32_t dstServerId = 0;
    uint32_t tokenIndex = 0;
    uint32_t startMutexIndex = 0;
    uint32_t tokenUbSize = tokenStructLen_;
    if constexpr (DynamicQuant || StaticQuant) {
        tokenUbSize = axisH_ * sizeof(XType) + INFO_NUM_IN_TOKENSTRUCK * (kAlign_ * sizeof(uint32_t));
    }
    tpipe_->InitBuffer(tokenServerIdxBuf_, sendTokenNum * serverNum * sizeof(int32_t));

    tokenServerIdxTensor_ = tokenServerIdxBuf_.Get<int32_t>();
    DataCopyExtParams tokenServerIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * serverNum * sizeof(int32_t)), 0U,
                                              0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(tokenServerIdxTensor_, tokenServerIdxGMTensor_[startTokenId * serverNum], tokenServerIdxParams,
                copyPadExtParams);
    // 这几个tensor是相同的地址空间，只是数据类型不一样，用于组装要发送的一个token和后面的metadata
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

        // 拷贝weight
        DataCopyPad(tokenTempTensorU8_[weightOffsetInStruct_], weightGt[(startTokenId + i) * realLenInStruct_],
                    weightCopyParams, weightPadParams);

        tokenTempTensorU8_.SetValue(cntOffsetInStruct_, startTokenId + i);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (int j = localIndex; j < localIndex + 1; j++) {
            if (tokenServerIdxTensor_(i * serverNum + j) == -1) {
                continue;
            }
            if (j == rankId_ / SERVER_RANK_SIZE) {
                rdmaRecvTail[j] = rdmaHeadTailTensor_.GetValue(j * 2 * UB_32B_ALIGN + 2 * 8);
                rdmaHeadTailTensor_.SetValue(j * 2 * UB_32B_ALIGN + 2 * 8, rdmaRecvTail[j] + 1);
                while (rdmaRecvTail[j] - rdmaRecvHead[j] >= rdmaItemNum) {
                    continue;
                }
                uint32_t destOffset = 
                    j * SERVER_SIZE_ON_WIN + tokenStructLen_ * (rdmaRecvTail[j] % rdmaItemNum) + TOKEN_COUNT_SIZE;
                DataCopy(rdmaRecvRingU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenStructLen_);
            } else {
                rdmaSendTail[j] = rdmaHeadTailTensor_.GetValue(j * 2 * UB_32B_ALIGN);
                rdmaHeadTailTensor_.SetValue(j * 2 * UB_32B_ALIGN, rdmaRecvTail[j] + 1);
                while (rdmaSendTail[j] - rdmaSendHead[j] >= rdmaItemNum) {
                    continue;
                }
                uint32_t destOffset = 
                    j * SERVER_SIZE_ON_WIN + tokenStructLen_ * (rdmaSendTail[j] % rdmaItemNum) + TOKEN_COUNT_SIZE;
                DataCopy(rdmaSendRingU8Tensor_[destOffset], tokenTempTensorU8_[0], tokenStructLen_);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    // for token in assigned_tokens:
    //     if token_dst_rank == my_rank:
    //         wait_until(rdma_tail - rdma_head < capacity)
    //         pack_token(token, rdma_buffer[rdma_tail])
    //         rdma_tail++
}



// 由RDMA_COORDINATOR执行，负责触发准备好的RDMA发送任务，发送到rdmaRecvRingU8Tensor这个环形buffer上，并更新环形buffer中头和尾的值
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::TriggerRdmaSend()
{
    if (aivRole_ != RDMA_COORDINATOR) {
        return ;
    }
    uint32_t localIndex = aivId_ % triggerNum;
    uint32_t destServerNum = serverNum / triggerNum;  // 每个AIV要处理的server数
    uint32_t rdmaServerNum = serverNum % triggerNum;
    uint32_t startServerId = destServerNum * localIndex;
    uint32_t curServerId = rankId_ / SERVER_RANK_SIZE;    // 当前serverId
    if (localIndex < rdmaServerNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        destServerNum += 1;
        startServerId += localIndex;
    } else {
        startServerId += rdmaServerNum;
    }
    if (destServerNum == 0) {
        return;
    }
    uint32_t endServerId = startServerId + destServerNum;

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
    }

    // 当前aiv负责 [startServerId,endServerId) 个 server
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        uint32_t sendIdx = dstServerInd - startServerId;
        uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + dstServerInd * SERVER_RANK_SIZE;  // 目标Rank
        rdmaRecvTail[dstServerInd] = rdmaHeadTailTensor_.GetValue(dstServerInd * 2 * UB_32B_ALIGN + 2 * 8);
        rdmaHeadTailTensor_.SetValue(dstServerInd * 2 * UB_32B_ALIGN + 2 * 8, rdmaRecvTail[dstServerInd] + RDMA_CHUNK);
        rdmaRecvHead[dstServerInd] = rdmaHeadTailTensor_.GetValue(dstServerInd * 2 * UB_32B_ALIGN + 3 * 8);
        rdmaSendHead[dstServerInd] = rdmaHeadTailTensor_.GetValue(dstServerInd * 2 * UB_32B_ALIGN + 1 * 8);
        int32_t realTail = rdmaRecvTail[dstServerInd] % rdmaItemNum;
        int32_t realHead = rdmaRecvHead[dstServerInd] % rdmaItemNum;
        int32_t sendHead = rdmaSendHead[dstServerInd] % rdmaItemNum;
        PipeBarrier<PIPE_ALL>();
        uint64_t dstDataRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + NOTIFY_OFFSET +
                                              halfWinSize_ * bufferId_ + curServerId * SERVER_SIZE_ON_WIN + realTail * tokenStructLen_);
        // src卡GetWindowsInAddr地址, 要发给serverIndex，即是本端的rdma地址
        uint64_t srcDataRdmaAddr =
            (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ + dstServerInd * SERVER_SIZE_ON_WIN + sendHead);

        // 去往该Server的传输的数据量
        uint32_t validTokenCount = RDMA_CHUNK;
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
        uint32_t dstServerOffset = dstServerInd;
        uint32_t sendInfoCount = B64_PER_BLOCK * PER_MSG_RDMA_SEND_TIME;
        DataCopy(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount], batchWriteU64Tensor_, sendInfoCount);

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if ASCEND_IS_AIV {
            while (((rdmaSendTail[dstServerInd] - sendHead + rdmaItemNum) % rdmaItemNum < RDMA_CHUNK) || (realTail + RDMA_CHUNK) % rdmaItemNum == realHead) {
                continue;
            }
            HcclHandle batchWriteResultData = hccl_.BatchWrite<true>((GM_ADDR)(dataBatchWriteInfoTensor_[dstServerOffset * sendInfoCount].GetPhyAddr()),
                                                                    PER_MSG_RDMA_SEND_TIME);
            bufferChosenGlobal_(0) = bufferId_ ^ 1;
            DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                bufferChosenGlobal_);
        }
    }

    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
    for (uint32_t dstServerInd = startServerId; dstServerInd < endServerId; ++dstServerInd) {
        while (true) {
            DataCopy(statusTensor, readStatusTensor_[(dstServerInd)*STATE_OFFSET / sizeof(int32_t)], FLAG_U32_CNT);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            int32_t sumOfFlag = statusTensor.GetValue(0);
            if (sumOfFlag == FLAG_VALUE) {
                break;
            }
        }
        rdmaSendHead[dstServerInd] = rdmaHeadTailTensor_.GetValue(dstServerInd * 2 * UB_32B_ALIGN + 1 * 8);
        rdmaSendHead[dstServerInd] += RDMA_CHUNK;
        rdmaHeadTailTensor_.SetValue(dstServerInd * 2 * UB_32B_ALIGN + 1 * 8, rdmaSendHead[dstServerInd]);
    }
    // if local_tail - last_sent >= RDMA_CHUNK:
    //     roce_write(rdma_buffer[last_sent : local_tail])
    //     rdma_atomic_add(remote_tail, local_tail - last_sent)
    //     last_sent = local_tail
}

// 由RDMA_HCCS_FORWARDER执行，负责将RDMA通信中接收到的数据进行HCCS的再发送，发送到hccsRecvRingU8Tensor这个环形buffer上
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Rdma2HCCS()
{
    if (aivRole_ != RDMA_HCCS_FORWARDER) {
        return ;
    }

    uint32_t serverId = rankId_ / SERVER_RANK_SIZE;
    uint32_t expertIdxStart = serverId * moeExpertNumInServer_;
    uint32_t expertIdxEnd = expertIdxStart_ + moeExpertNumInServer_;
    uint32_t senderNum = aivNum_ / 4 - 1;
    uint32_t eachChunked = rdmaItemNum;
    uint32_t localStartWorkerCoreIdx = (RDMA_HCCS_FORWARDER - 1) * senderNum;
    uint32_t localWorkCoreId = aivId_ - localStartWorkerCoreIdx;
    uint32_t tokenStart = 0;
    uint32_t tokenEnd = tokenStart + serverNum * eachChunked;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    if (localWorkCoreId < SERVER_RANK_SIZE) {
        return;
    }
    DataCopyExtParams tokenStructParams{1, static_cast<uint32_t>(tokenStructLen_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> tokenStructPadParams{false, 0U, 0U, 0U};
    DataCopyParams hccsHesdTailParams{2, sizeof(uint32_t), 0, 0};
    uint32_t processedTokenNum = 0;
    uint32_t tokenGlobalCnt = 0;
    for (int i = 0; i < serverNum; i++) {
        tokenGlobalCnt += rdmaRecvRingU8Tensor_.GetValue(i * rdmaItemNum);
    }
    while (processedTokenNum <= tokenGlobalCnt) {
        for (int i = 0; i < serverNum * rdmaItemNum; i++) {
            uint32_t currentServerId = i / rdmaItemNum;
            uint32_t rdmaHead = rdmaHeadTailTensor_.GetValue(currentServerId * RING_BUFFER_HEAD_TAIL + 2);
            uint32_t rdmaTail = rdmaHeadTailTensor_.GetValue(currentServerId * RING_BUFFER_HEAD_TAIL + 3);

            if (rdmaTail == rdmaHead) {
                continue;
            }
            DataCopyPad(tokenStructInRdmaTensor_, rdmaRecvRingU8Tensor_[sizeof(uint32_t) * currentServerId + (currentServerId * rdmaItemNum + rdmaHead) * tokenStructLen_], tokenStructParams, 
            tokenStructPadParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            LocalTensor<int> topkIdxInStructTensor = tokenStructInRdmaTensor_[expOffsetInStruct_].ReinterpretCast<int>();
            LocalTensor<int> tokenIdxInStructTensor = tokenStructInRdmaTensor_[cntOffsetInStruct_].ReinterpretCast<int>();
            LocalTensor<uint32_t> srcRankInStructTensor = tokenStructInRdmaTensor_[cntOffsetInStruct_].ReinterpretCast<uint32_t>();
            int localTokenIdx = tokenIdxInStructTensor.GetValue(0);
            int recvTokenCnt = 0;
            uint32_t srcRank = i * SERVER_RANK_SIZE + rankId_ % SERVER_RANK_SIZE;
            srcRankInStructTensor.SetValue(0, srcRank);
            for (int j = 0; j < rankId_; j++) {
                recvTokenCnt += tokenPerRankGMTensor_.GetValue(j); 
            }
            uint32_t globalTokenIdx = recvTokenCnt + localTokenIdx;
            tokenIdxInStructTensor.SetValue(0, globalTokenIdx);

            //同一个token可能由于expertId的原因重复发往同一个hccs环形缓冲区，需要在hccs缓冲区做处理
            for (int j = 0; j < axisK_; j++) {
                int dstExpert = topkIdxInStructTensor.GetValue(j);
                if (dstExpert < expertIdxStart || dstExpert >= expertIdxEnd) {
                    topkIdxInStructTensor.SetValue(j, -1);
                    continue;
                }
                uint32_t localDstRank = (dstExpert - expertIdxStart) / localMoeExpertNum_; 
                if (localDstRank != localWorkCoreId) {
                    continue;
                }
                GlobalTensor<uint8_t> dstRankRecvRingU8Tensor;
                dstRankRecvRingU8Tensor.SetGlobalBuffer((__gm__ uint8_t *) (hccl_.GetWindowsInAddr(localDstRank)) + halfWinSize_ / 2);
                LocalTensor<uint32_t> localHccsHeadTailTensor;
                GlobalTensor<uint32_t> globalHccsHeadTailTensor;
                globalHccsHeadTailTensor.SetGlobalBuffer((__gm__ uint32_t *)hccsHeadTailGM[localDstRank]);
                DataCopy(localHccsHeadTailTensor, globalHccsHeadTailTensor[localRankId], hccsHesdTailParams);
                uint32_t hcclTail = localHccsHeadTailTensor.GetValue(1); 
                uint32_t hcclHead = localHccsHeadTailTensor.GetValue(0);
                uint32_t index = 0;
                while (hcclHead == (hcclTail + 1) % hccsItemNum) {
                    hcclHead = localHccsHeadTailTensor.GetValue(0); //优化点，当前处理完一整个token后再进行下一个token的处理，此处可以有优化空间，尝试跳过无空闲的hccs环形缓冲区
                }
                for (int k = 0; k < hccsItemNum; k++) {
                    DataCopyPad(rdmaUseTokenStructInHccsTensor_, dstRankRecvRingU8Tensor[k * tokenStructLen_], 
                    tokenStructParams, tokenStructPadParams);
                    LocalTensor<int> tokenIdTensor = rdmaUseTokenStructInHccsTensor_[cntOffsetInStruct_].ReinterpretCast<int>();
                    int tokenId = tokenIdTensor.GetValue(0);
                    if (tokenId == -1) {
                        index = k;
                        break;
                    }
                }
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                DataCopyPad(dstRankRecvRingU8Tensor[hccsItemNum * localDstRank + tokenStructLen_ * index], tokenStructInRdmaTensor_, 
                tokenStructParams);
                DataCopyPad(rdmaRecvRingU8Tensor_[(i * rdmaItemNum + rdmaHead) * tokenStructLen_], tokenStructInRdmaTensor_, 
                tokenStructParams);
                rdmaHead = (rdmaHead + 1) % rdmaItemNum;
                hcclTail = (hcclTail + 1) % hccsItemNum;
                rdmaHeadTailTensor_.SetValue(i * RING_BUFFER_HEAD_TAIL + 2, rdmaHead);
                localHccsHeadTailTensor.SetValue(1, hcclTail);
                DataCopy(globalHccsHeadTailTensor[localRankId], localHccsHeadTailTensor, hccsHesdTailParams);
            }
            processedTokenNum++;
        }
    }
    // while rdma_head < rdma_tail:
    //     wait_hccs_space()
    //     dma_async(rdma_buf[rdma_head], hccs_buf[hccs_tail])
    //     rdma_head++
    //     hccs_tail++
}

// 由FORWARDER_COORDINATOR执行，负责将rdmaRecvRingU8Tensor这个环形buffer中已发送到hccsRecvRingU8Tensor的部分进行回收
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::CreditRecycle()
{
    if (aivRole_ != FORWARDER_COORDINATOR) {
        return ;
    }
    // if rdma_head - last_reported >= CREDIT_BATCH:
    //     rdma_atomic_add(remote_head, CREDIT_BATCH)
    //     last_reported += CREDIT_BATCH
}

// 由HCCS_RECEIVER执行，负责将hccsRecvRingU8Tensor的数据发送到最终输出
template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::HCCS2Out()
{
    if (aivRole_ != HCCS_RECEIVER) {
        return ;
    }
    //每个核在hccs环形缓冲区上划分一部分长度，本核在工作周期就只负责这段长度上的token
    uint32_t senderNum = aivNum_ / 4 - 1;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    uint32_t tokenNumPerCore = hccsItemNum / senderNum;
    uint32_t remaind = hccsItemNum % senderNum;
    uint32_t localStartWorkerCoreIdx = (HCCS_RECEIVER - 1) * senderNum;
    uint32_t localWorkerCoreIdx = aivId_ - localStartWorkerCoreIdx;
    uint32_t tokenStart = 0;
    uint32_t tokenEnd = 0;
    if (localWorkerCoreIdx < remaind) {
        tokenNumPerCore++;
        tokenStart = localWorkerCoreIdx * tokenNumPerCore;
    } else {
        tokenStart = localWorkerCoreIdx * tokenNumPerCore + remaind;
    }
    tokenEnd = tokenStart + tokenNumPerCore;
    uint32_t processedTokens = 0;
    DataCopyExtParams tokenStructParams{1, static_cast<uint32_t>(tokenStructLen_), 0, 0, 0};
    DataCopyExtParams tokenParams{1, static_cast<uint32_t>(tokenLenInStruct_), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> tokenStructPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams weightParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> weightExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams scalesParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> scalesExtParams{false, 0U, 0U, 0U};
    uint32_t tokenRankCnt = epRankTokenCntGMTensor_.GetValue(rankId_);
    uint32_t sumTokenPerCore = tokenRankCnt / senderNum;
    uint32_t remaindSumTokenPerCore = tokenRankCnt % senderNum;
    if (localWorkerCoreIdx < remaindSumTokenPerCore) {
        sumTokenPerCore++;
    }
    while (processedTokens < sumTokenPerCore) {
        for (int i = tokenStart; i < tokenEnd; i++) {
            DataCopyPad(tokenStructInHccsTensor_, hccsRecvRingU8Tensor_[tokenStructLen_ * i], 
            tokenStructParams, tokenStructPadParams);
            uint32_t expertIdxStart = localMoeExpertNum_ * rankId_;
            uint32_t expertIdxEnd = expertIdxStart + localMoeExpertNum_;
            LocalTensor<int> tokenIdxInStructTensor = tokenStructInHccsTensor_[cntOffsetInStruct_].ReinterpretCast<int>();
            LocalTensor<uint8_t> tokenIdxInStructToGmTensor = tokenStructInHccsTensor_[cntOffsetInStruct_];
            uint32_t tokenIdx = tokenIdxInStructTensor.GetValue(0);
            if (tokenIdx < 0) {
                continue;
            }
            LocalTensor<float> weightTensor = tokenStructInHccsTensor_[weightOffsetInStruct_].ReinterpretCast<float>();
            LocalTensor<ExpandXOutType> tokenOutTensor = tokenStructInHccsTensor_.ReinterpretCast<ExpandXOutType>();
            LocalTensor<int> topkIdxTensor = tokenStructInHccsTensor_[expOffsetInStruct_].ReinterpretCast<int>();
            uint32_t dstOffset = 0;
            for (int j = 0; j < axisK_; j++) {
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                uint32_t dstExpert = topkIdxTensor.GetValue(j);
                if (dstExpert < expertIdxStart || dstExpert >= expertIdxEnd) {
                    continue;
                }
                dstOffset = tokenIdxPerExpertGMTensor_.GetValue(tokenIdx * moeExpertNum_ + dstExpert);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                DataCopyPad(weightsOutGt[dstOffset], weightTensor[j], weightParams);
                DataCopyPad(expandXOutGMTensor_[dstOffset], tokenOutTensor, tokenParams);
                // dynamic scales to output
                if constexpr (DynamicQuant) {
                    LocalTensor<float> quantTempUB = tokenStructInHccsTensor_[scaleOffsetInStruct_].ReinterpretCast<float>();
                    DataCopyPad(dynamicScalesOutGMTensor_[dstOffset], quantTempUB, scalesParams);
                }
            }
            tokenIdxInStructTensor.SetValue(0, -1);
            DataCopyPad(hccsRecvRingU8Tensor_[tokenStructLen_ * i], tokenIdxInStructToGmTensor, tokenStructParams);
            uint32_t hcclHead = hccsHeadTailTensor_.GetValue(localRankId * 2); //需要一个锁，避免多个core同时更新本rank的head
            hccsHeadTailTensor_.SetValue(localRankId, hcclHead + 1);
            ++processedTokens;
        }
    }
    
    // while hccs_head < hccs_tail:
    //     dma_async(hccs_buf[hccs_head], recv_x[offset])
    //     hccs_head++
}

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline uint64_t
MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::MergeMagicWithValue(uint64_t magic, uint64_t value)
{
    return (magic * 2ULL + value);
}

template <TemplateMC2TypeA2PipelineClass>
__aicore__ inline void MoeDistributeDispatchA2Pipeline<TemplateMC2TypeA2PipelineFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        PrepareRdmaSend();
        TriggerRdmaSend();
        Rdma2HCCS();
        CreditRecycle();
        HCCS2Out();

        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
