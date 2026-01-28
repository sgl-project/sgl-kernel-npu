#ifndef MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#define MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_a2_tiling.h"
#include "moe_distribute_base.h"

namespace MoeDistributeCombineA2Impl {

#define COMBINE_2SERVER_VERSION

#define TemplateMC2TypeA2layeredClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA2layeredFunc ExpandXType, ExpandIdxType
using namespace AscendC;
template <TemplateMC2TypeA2layeredClass>
class MoeDistributeCombineA2Layered
{
public:
    constexpr static uint32_t BUFFER_NUM = 2U;                   // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;               // 状态空间偏移地址
    constexpr static uint32_t STATE_SPACE_SIZE = 1024U * 1024U;  // 1M
    constexpr static uint32_t UB_ALIGN = 32U;                    // UB按32字节对齐
    constexpr static uint32_t SELF_STATE_OFFSET = 512U * 1024U;  // 本卡状态空间偏移地址
    constexpr static uint32_t BATCH_WRITE_ITEM_OFFSET =
        8U * 1024U;  // batchWriteInfo结构体地址相对于windowOut最后1M的偏移
    constexpr static uint32_t BATCH_WRITE_ITEM_SIZE = 32U;
    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static uint32_t SERVER_RANK_SIZE = 8U;
    constexpr static uint32_t IPC_DATA_OFFSET = 4U * 1024U * 1024U;
    constexpr static uint32_t IPC_DISPATCH_DATA_OFFSET = 2U * 1024U * 1024U;
    constexpr static uint32_t NOTIFY_DATA_SIZE = 400U * 1024U * 1024U;
    constexpr static uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U;  // 专家信息 权重信息 量化Scale 到达标志位
    constexpr static uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr static uint32_t MAX_BS = 4096;  // 每卡支持的最大bs

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
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

    __aicore__ inline MoeDistributeCombineA2Layered(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR offsetInner,
                                GM_ADDR offsetOuter, GM_ADDR countOuter, GM_ADDR scales, GM_ADDR XOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData,
                                __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void SumToServer();
    __aicore__ inline void Preload();

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<ExpandIdxType> bkCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_;  // 用于存对端window的变量
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> bufferIdGlobal_;     // 用于存对端状态window的变量
    GlobalTensor<int32_t> statusSpaceGlobal_;   // win区状态位置拷入相关参数
    GlobalTensor<uint64_t> workspaceGlobal_;    // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> workspaceGlobal32_;  // 存储batchWriteInfo结构体信息
    GlobalTensor<int32_t> readStateGlobal_;
    GlobalTensor<int32_t> dstRankStateGlobal_;
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
    uint64_t shareAddreRank[8];
    GlobalTensor<ExpandXType> selfRankshareMemGlobal_;

    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR XOutGM_;

    // 分层所需的参数
    GM_ADDR shareAddrGM_;
    GM_ADDR offsetInnerGM_;
    GM_ADDR countInnerGM_;
    GM_ADDR offsetOuterGM_;
    GM_ADDR countOuterGM_;
    GM_ADDR recvCountInnerGM_;
    GlobalTensor<int32_t> shareAddrGlobal_;
    GlobalTensor<int64_t> shareFlagGlobal_;
    GlobalTensor<ExpandXType> shareMemGlobal_;
    GlobalTensor<ExpandXType> dstshareMemGlobal_;
    GlobalTensor<int32_t> offsetInnerGlobal_;
    //    GlobalTensor<int32_t> countInnerGlobal_;
    GlobalTensor<int32_t> offsetOuterGlobal_;
    GlobalTensor<int32_t> countOuterGlobal_;
    GlobalTensor<int32_t> recvCountInnerGlobal_;
    TBuf<> offsetReduceBuf_;
    TBuf<> countReduceBuf_;
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t countReL{0};
    uint32_t axisBS_{0};
    uint32_t globalBs{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};  // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};              // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    __gm__ HcclOpResParam *winContext_{nullptr};
    uint32_t moeExpertNum_{0};       // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0};  // 每张卡的专家数
    uint32_t expandXRows_;
    uint64_t rankSizeOnWin_{0};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint64_t dataOffsetOnWin_{0};
    uint64_t stateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};
    uint32_t tokenNumPerCore_{0};
    uint32_t tokenIndex_{0};
    uint32_t serverNum{0};
    uint32_t ipcSliceSize{0};
    uint32_t ipcSliceNodeSize{0};
    uint64_t send_counts_inner_offset{0};
    uint64_t offset_inner_offset{0};
    uint64_t send_counts_outer_offset{0};
    uint64_t offset_outer_offset{0};
    uint64_t share_offset{0};
    uint32_t IPC_DATA_SIZE{0};
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> moeQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> moeSumQueue_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> sendCountBuf_;
    TBuf<> statusBuf_;
    TBuf<> statusSumOutBuf_;
    TBuf<> batchWriteItemBuf_;
    TBuf<> recvCountBuf_;
    TBuf<> scaleBuf_;
    TBuf<> expertWindowOffsetBuf_;
    TBuf<> innerOffsetBuf_;
    int32_t sumTarget_{0};
    int32_t stateValue_{0};
    uint32_t startBs{0};
    uint32_t endBs{0};
    uint32_t processNum{0};
    uint32_t resNum{0};
    uint32_t resLen{0};
    uint32_t offsetIndex{0};
    uint32_t maxLocalBs{0};
    LocalTensor<int32_t> offsetReduceLocal_;
    LocalTensor<int32_t> countReduceLocal_;
};

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Init(
    GM_ADDR expandX, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR offsetInner, GM_ADDR offsetOuter, GM_ADDR countOuter,
    GM_ADDR scales, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData,
    __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    expandXGM_ = expandX;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    XOutGM_ = XOut;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    globalBs = tilingData->moeDistributeCombineInfo.globalBs;

    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;

    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)contextGM;
    hccl_.Init(contextGM, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

    halfWinSize_ = RDMA_DATA_SIZE / 2U;
    IPC_DATA_SIZE = winContext_->winSize - RDMA_DATA_SIZE - IPC_DATA_OFFSET - NOTIFY_DATA_SIZE;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_ + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferIdGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();
    serverNum = worldSize_ / SERVER_RANK_SIZE;
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    offsetInnerGlobal_.SetGlobalBuffer((__gm__ int32_t *)offsetInner);
    countOuterGlobal_.SetGlobalBuffer((__gm__ int32_t *)countOuter);
    offsetOuterGlobal_.SetGlobalBuffer((__gm__ int32_t *)offsetOuter);
    bkCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)(sendCount + worldSize_ * localMoeExpertNum_ * 4));
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    readStateGlobal_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + dataSpaceSize_));
    workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));
    workspaceGlobal32_.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_ + dataSpaceSize_ + BATCH_WRITE_ITEM_OFFSET));
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    expandXRows_ = localMoeExpertNum_ * axisBS_ * worldSize_;
    rankSizeOnWin_ = static_cast<uint64_t>(dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE);
    statusSpaceGm_ = windowInGM_ + dataSpaceSize_;
    statusSpaceGlobal_.SetGlobalBuffer((__gm__ int32_t *)statusSpaceGm_);
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = static_cast<uint64_t>(dataSpaceSize_ + rankId_ * STATE_OFFSET);
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));

    uint64_t winSizeMin =
        serverNum * axisBS_ * (axisHExpandXTypeSize_ + EXTRA_TOKEN_INFO_NUM * axisK_ * sizeof(uint32_t)) +
        IPC_DATA_OFFSET + RDMA_DATA_SIZE + NOTIFY_DATA_SIZE;  // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
    assert(winContext_->winSize >= winSizeMin,
           "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
        epWorldSize:%u, epRankId:%u, moeExpertNum:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
        totalUbSize:%llu, hcclBufferSize:%u\n",
           winContext_->winSize / MB_SIZE, winSizeMin / MB_SIZE, tilingData->moeDistributeCombineInfo.epWorldSize,
           tilingData->moeDistributeCombineInfo.epRankId, tilingData->moeDistributeCombineInfo.moeExpertNum,
           tilingData->moeDistributeCombineInfo.globalBs, tilingData->moeDistributeCombineInfo.bs,
           tilingData->moeDistributeCombineInfo.k, tilingData->moeDistributeCombineInfo.h,
           tilingData->moeDistributeCombineInfo.aivNum, tilingData->moeDistributeCombineInfo.totalUbSize,
           tilingData->moeDistributeCombineInfo.hcclBufferSize);

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    // coreIdx_ < serverNum
    int32_t state = selfStatusTensor(coreIdx_ * UB_ALIGN);
    if (state == 0) {
        sumTarget_ = static_cast<int32_t>(1);
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 1;
        stateValue_ = 1;
    } else {
        sumTarget_ = 0;
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 0;
        stateValue_ = 0;
    }
    BuffInit();
    SplitCoreCal();
    if (coreIdx_ == 0U) {
        readStateGlobal_.SetValue(0, stateValue_);
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            readStateGlobal_);
    }
    send_counts_inner_offset = static_cast<uint64_t>(worldSize_ * localMoeExpertNum_);
    offset_inner_offset = send_counts_inner_offset + static_cast<uint64_t>(globalBs * serverNum);
    send_counts_outer_offset = offset_inner_offset + static_cast<uint64_t>(globalBs * axisK_ * serverNum);
    offset_outer_offset = send_counts_outer_offset + static_cast<uint64_t>(axisBS_);
    share_offset = offset_outer_offset + static_cast<uint64_t>(axisBS_ * serverNum);

    shareAddrGM_ = sendCount + share_offset;
    offsetInnerGM_ = sendCount + offset_inner_offset;
    countInnerGM_ = sendCount + send_counts_inner_offset;
    offsetOuterGM_ = sendCount + offset_outer_offset;
    countOuterGM_ = sendCount + send_counts_outer_offset;

    shareAddrGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + share_offset);
    PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < 8; i++) {
        shareAddreRank[i] = reinterpret_cast<uint64_t>(
            RDMA_DATA_SIZE + hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i));
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::BuffInit()
{
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, (axisHExpandXTypeSize_ + 32U));  // 7168 * 2 * 2 = 28672
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(rowTmpFloatBuf_, axisHFloatSize_);
    tpipe_->InitBuffer(mulBuf_, axisHFloatSize_);       //  // 7168 * 4 = 28672
    tpipe_->InitBuffer(sumFloatBuf_, axisHFloatSize_);  //  // 7168 * 4 = 28672
    tpipe_->InitBuffer(sendCountBuf_, RoundUp(moeExpertNum_ * worldSize_, B32_PER_BLOCK) *
                                          sizeof(int32_t));  // 全局sendCount，为每个专家从不同rank接收的token个数
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, (axisHExpandXTypeSize_ + 32U));
    tpipe_->InitBuffer(statusSumOutBuf_, sizeof(float));
    tpipe_->InitBuffer(batchWriteItemBuf_, BATCH_WRITE_ITEM_SIZE * worldSize_);
    tpipe_->InitBuffer(innerOffsetBuf_, localMoeExpertNum_ * SERVER_RANK_SIZE * sizeof(int32_t));
    batchWriteItemLocalB64 = batchWriteItemBuf_.Get<uint64_t>();
    batchWriteItemLocalB32 = batchWriteItemLocalB64.template ReinterpretCast<uint32_t>();
}
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SplitCoreCal()
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
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AlltoAllDispatch()
{
    rowTmpFloatLocal_ = rowTmpFloatBuf_.Get<float>();
    ipcSliceSize = IPC_DATA_SIZE / worldSize_;
    ipcSliceNodeSize = ipcSliceSize * SERVER_RANK_SIZE;
    LocalTensor<ExpandIdxType> sendCountLocal = sendCountBuf_.Get<int32_t>();
    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_ * worldSize_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        // dstRankId 在本机上的同号卡
        uint32_t targetRank = dstRankId % SERVER_RANK_SIZE;
        // 计算要发往的目标IPC的地址，不考虑flag偏移
        uint64_t targetRankShareAddr = shareAddreRank[targetRank];
        uint64_t targetRankAddr =
            targetRankShareAddr +
            static_cast<uint64_t>(dstRankId / SERVER_RANK_SIZE * ipcSliceNodeSize + IPC_DATA_OFFSET);

        dstshareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)(targetRankAddr));
        shareFlagGlobal_.SetGlobalBuffer((__gm__ int64_t *)(targetRankShareAddr + IPC_DISPATCH_DATA_OFFSET));
        // 计算要发送的token数量
        uint32_t rankTokenNum = 0U;
        uint32_t serverStartExpId = rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE * localMoeExpertNum_;
        for (uint32_t expertId = 0U; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0U;
            if (expertId != 0U || dstRankId != 0U) {
                for (int i = 0; i <= expertId; i++) {
                    for (int j = 0; j < worldSize_; j++) {
                        if ((i == expertId) && j >= dstRankId) {
                            break;
                        }
                        // 对epSendCount的前startExpId到expertId - 1的行求和，和第expertId行的前dstRankId - 1求和
                        preCount += sendCountLocal.GetValue((i + rankId_ * localMoeExpertNum_) * worldSize_ + j);
                    }
                }
            }

            uint32_t tokenNum =
                sendCountLocal.GetValue((expertId + rankId_ * localMoeExpertNum_) * worldSize_ + dstRankId);
            uint32_t startTokenAddr = preCount * axisH_;
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t tokenOffset = 0;
            // 从本server的专家起始id到当前expertId
            for (int i = serverStartExpId; i < rankId_ * localMoeExpertNum_ + expertId; i++) {
                tokenOffset += sendCountLocal.GetValue(i * worldSize_ + dstRankId);
            }
            for (uint32_t tokenId = 0U; tokenId < tokenNum; ++tokenId) {
                float scaleVal = expandScalesGlobal_[preCount].GetValue(tokenId);
                LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
                LocalTensor<float> InUbTemp = InUb[axisH_].template ReinterpretCast<float>();
                InUbTemp(0) = scaleVal;
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopy(InUb, expandXGlobal_[startTokenAddr], axisH_);
                moeQueue_.EnQue(InUb);
                LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
                DataCopy(dstshareMemGlobal_[(tokenOffset + tokenId) * (axisH_ + 16U)], OutUb, axisH_ + 16U);
                moeQueue_.FreeTensor<ExpandXType>(OutUb);
                startTokenAddr += axisH_;
                rankTokenNum++;
                PipeBarrier<PIPE_ALL>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int64_t> InUb = statusBuf_.AllocTensor<int64_t>();
        InUb.SetValue(0, 12345);
        uint32_t flagOffset = rankId_ % SERVER_RANK_SIZE + dstRankId / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
        DataCopy(shareFlagGlobal_[flagOffset * 4], InUb, 4);  // *4是因为单次拷贝256byte = 4*int64
        statusBuf_.FreeTensor<int64_t>(InUb);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToWindow()
{
    // 当前假设一个core处理一个rank的数据累加，因为已经只剩下同号卡，所以只有serverNum个rank
    if (coreIdx_ < serverNum) {
        shareFlagGlobal_.SetGlobalBuffer(
            (__gm__ int64_t *)(shareAddreRank[rankId_ % SERVER_RANK_SIZE] + IPC_DISPATCH_DATA_OFFSET));
        LocalTensor<int64_t> InUb = statusBuf_.AllocTensor<int64_t>();
        for (uint32_t i = 0U; i < SERVER_RANK_SIZE; i++) {
            uint32_t waitFlagAddr = coreIdx_ * SERVER_RANK_SIZE + i;
            while (true) {
                DataCopy(InUb, shareFlagGlobal_[waitFlagAddr * 4], 4);
                PipeBarrier<PIPE_ALL>();
                if (InUb.GetValue(0) == 12345) {
                    break;
                }
            }
        }
        InUb.SetValue(0, 0);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0U; i < SERVER_RANK_SIZE; i++) {
            DataCopy(shareFlagGlobal_[(coreIdx_ * SERVER_RANK_SIZE + i) * 4], InUb,
                     4);  // *4是因为单次拷贝256byte = 4*int64
            PipeBarrier<PIPE_V>();
        }

        statusBuf_.FreeTensor<int64_t>(InUb);
    }
    SyncAll();
    LocalTensor<int32_t> offsetReduceLt =
        innerOffsetBuf_.GetWithOffset<int32_t>(localMoeExpertNum_ * SERVER_RANK_SIZE, 0);
    uint32_t corePerServer = aivNum_ / serverNum;
    if (coreIdx_ >= corePerServer * serverNum) {
        SyncAll<true>();
        return;
    }
    uint32_t BSPerCore = MAX_BS / corePerServer;
    uint32_t remainBS = MAX_BS % corePerServer;
    uint32_t localBlockIdx = coreIdx_ / serverNum;
    uint32_t currentServerIdx = coreIdx_ % serverNum;

    uint32_t startTokenId = localBlockIdx * BSPerCore;
    if (localBlockIdx < remainBS) {
        startTokenId += localBlockIdx;
        BSPerCore += 1;
    } else {
        startTokenId += remainBS;
    }

    uint32_t endTokenId = startTokenId + BSPerCore;

    int32_t targetRankId = currentServerIdx * SERVER_RANK_SIZE + rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int32_t> offsetReduceGt = offsetInnerGlobal_[MAX_BS * moeExpertNum_ * currentServerIdx];
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint64_t copyAddr = shareAddreRank[rankId_ % SERVER_RANK_SIZE] +
                        static_cast<uint64_t>(IPC_DATA_OFFSET + currentServerIdx * ipcSliceNodeSize);
    shareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)copyAddr);
    uint64_t rdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                   currentServerIdx * rankSizeOnWin_ * SERVER_RANK_SIZE);
    localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)rdmaAddr);
    sumFloatLocal_ = sumFloatBuf_.Get<float>();
    countReL = 0;
#ifdef COMBINE_2SERVER_VERSION
    if (coreIdx_ < serverNum) {
        countReL = MAX_BS;
    }
#else
    int32_t countRels[MAX_BS] = {0};
    for (uint32_t i = 0U; i < MAX_BS; i++) {
        bool isTokenInServer = false;
        DataCopy(offsetReduceLt,
                 offsetReduceGt[i * moeExpertNum_ + rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE * localMoeExpertNum_],
                 localMoeExpertNum_ * SERVER_RANK_SIZE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t j = 0U; j < static_cast<uint32_t>(localMoeExpertNum_ * SERVER_RANK_SIZE); j++) {
            int32_t expId = j;
            int32_t offsetValue = offsetReduceLt.GetValue(expId);
            if (offsetValue == -1) continue;
            isTokenInServer = true;
            break;
        }
        if (i == MAX_BS - 1) {
            if (coreIdx_ < serverNum) {
                countReL = isTokenInServer ? countRels[i] + 1 : countRels[i];
            }
            continue;
        }
        if (isTokenInServer) {
            countRels[i + 1] = countRels[i] + 1;
        } else {
            countRels[i + 1] = countRels[i];
        }
    }
#endif

    for (uint32_t i = startTokenId; i < endTokenId; i++) {
        bool isTokenInServer = false;
        Duplicate(sumFloatLocal_, 0.0f, axisH_);
        DataCopy(offsetReduceLt,
                 offsetReduceGt[i * moeExpertNum_ + rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE * localMoeExpertNum_],
                 localMoeExpertNum_ * SERVER_RANK_SIZE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t j = 0U; j < static_cast<uint32_t>(localMoeExpertNum_ * SERVER_RANK_SIZE); j++) {
            int32_t offsetValue = offsetReduceLt.GetValue(j);
            if (offsetValue < 0) continue;
            isTokenInServer = true;
            tmpUb_ = moeSumQueue_.AllocTensor<ExpandXType>();
            uint32_t offsetOnIpc = (offsetValue * (axisH_ + 16U) * sizeof(ExpandXType)) / sizeof(ExpandXType);
            DataCopy(tmpUb_, shareMemGlobal_[offsetOnIpc], axisH_ + 16U);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            LocalTensor<float> InUbTemp = tmpUb_[axisH_].template ReinterpretCast<float>();
            float scaleVal = InUbTemp(0);
            SyncFunc<AscendC::HardEvent::S_V>();
            moeSumQueue_.EnQue(tmpUb_);
            LocalTensor<ExpandXType> tmpOtherUb_ = moeSumQueue_.DeQue<ExpandXType>();
            Cast(rowTmpFloatLocal_, tmpOtherUb_, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(rowTmpFloatLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatLocal_, sumFloatLocal_, rowTmpFloatLocal_, axisH_);
            moeSumQueue_.FreeTensor<ExpandXType>(tmpOtherUb_);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        if (!isTokenInServer) {
            continue;
        }
        LocalTensor<ExpandXType> castUbIn = mulBuf_.Get<ExpandXType>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(castUbIn, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
#ifdef COMBINE_2SERVER_VERSION
        DataCopy(localOutWindow_[i * axisH_], castUbIn, axisH_);
#else
        DataCopy(localOutWindow_[countRels[i] * axisH_], castUbIn, axisH_);
#endif
        PipeBarrier<PIPE_V>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AlltoAllServerDispatch()
{
    uint64_t selfTotalNum = 0U;
    if (coreIdx_ < serverNum) {
        uint32_t tragRankId = rankId_ % SERVER_RANK_SIZE + coreIdx_ * SERVER_RANK_SIZE;
        // 目标卡 GetWindowsOutAddr 地址
        uint64_t dstrdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(tragRankId) + halfWinSize_ * bufferId_ +
                                          (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);
        uint64_t srcrdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                          coreIdx_ * rankSizeOnWin_ * SERVER_RANK_SIZE);

        // countReL
        batchWriteItemLocalB64(0) = srcrdmaAddr;
        batchWriteItemLocalB64(0 + 1) = dstrdmaAddr;
        if (coreIdx_ == (rankId_ / SERVER_RANK_SIZE)) {
            batchWriteItemLocalB64(0 + 2) = 0;
        } else {
            batchWriteItemLocalB64(0 + 2) = countReL * axisH_;
        }
        batchWriteItemLocalB32(0 + 6) = HcclDataType::HCCL_DATA_TYPE_FP16;
        batchWriteItemLocalB32(0 + 7) = tragRankId;

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(workspaceGlobal_[coreIdx_ * 4], batchWriteItemLocalB64, 4);
    }
    SyncAll<true>();
    if (coreIdx_ == 0U) {
        HcclHandle handleId = hccl_.BatchWrite<true>((GM_ADDR)(workspaceGlobal_.GetPhyAddr()), serverNum);
        bufferIdGlobal_(0) = bufferId_ ^ 1;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            bufferIdGlobal_);
    }
    if (coreIdx_ == (rankId_ / SERVER_RANK_SIZE)) {
        uint64_t srcrdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) +
                                          halfWinSize_ * bufferId_ +  //(rankId_ % SERVER_RANK_SIZE) * rankSizeOnWin_);
                                          (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);
        uint64_t dstrdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_) +
                                          halfWinSize_ * bufferId_ +  //(rankId_ % SERVER_RANK_SIZE) * rankSizeOnWin_);
                                          (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);

        localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(dstrdmaAddr));
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(srcrdmaAddr));

        for (uint32_t tokenId = 0U; tokenId < countReL; ++tokenId) {
            LocalTensor<ExpandXType> InUb = moeQueue_.AllocTensor<ExpandXType>();
            DataCopy(InUb, localOutWindow_[tokenId * axisH_], axisH_);
            moeQueue_.EnQue(InUb);
            LocalTensor<ExpandXType> OutUb = moeQueue_.DeQue<ExpandXType>();
            DataCopy(localInWindow_[tokenId * axisH_], OutUb, axisH_);
            moeQueue_.FreeTensor<ExpandXType>(OutUb);
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SetStatus()
{
    if (coreIdx_ != 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t selfServerID = rankId_ / SERVER_RANK_SIZE;
    for (uint32_t serverID = 0U; serverID < serverNum; serverID++) {
        uint32_t targetRank = rankId_ % SERVER_RANK_SIZE + serverID * SERVER_RANK_SIZE;
        batchWriteItemLocalB64(serverID * 4) = (uint64_t)(readStateGlobal_.GetPhyAddr());
        batchWriteItemLocalB64(serverID * 4 + 1) =
            (uint64_t)(hccl_.GetWindowsInAddr(targetRank) + halfWinSize_ * bufferId_ + dataSpaceSize_ +
                       selfServerID * STATE_OFFSET);
        batchWriteItemLocalB64(serverID * 4 + 2) = 8;
        batchWriteItemLocalB32(serverID * 8 + 6) = HcclDataType::HCCL_DATA_TYPE_INT32;
        batchWriteItemLocalB32(serverID * 8 + 7) = targetRank;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(workspaceGlobal_[serverNum * 4], batchWriteItemLocalB64, 4 * (serverNum));
    GlobalTensor<int32_t> localStateGlobal;
    localStateGlobal.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + dataSpaceSize_ + selfServerID * STATE_OFFSET));
    localStateGlobal.SetValue(0, stateValue_);
    DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        localStateGlobal);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    if ASCEND_IS_AIV {
        HcclHandle handleId =
            hccl_.BatchWrite<true>((GM_ADDR)(workspaceGlobal_[serverNum * 4].GetPhyAddr()), serverNum);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitDispatch()
{
    if (coreIdx_ < serverNum) {
        uint32_t targetRank = rankId_ % SERVER_RANK_SIZE + (coreIdx_)*SERVER_RANK_SIZE;
        LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
        uint32_t readNum = 1U;
        DataCopyParams intriParams{static_cast<uint16_t>(readNum), 1, 15, 0};  // srcStride为15个block
        while (true) {
            DataCopy(statusTensor, statusSpaceGlobal_[(coreIdx_)*STATE_OFFSET / sizeof(int32_t)], intriParams);
            PipeBarrier<PIPE_ALL>();
            int32_t sumOfFlag = statusTensor.GetValue(0);

            if (sumOfFlag == sumTarget_) {
                break;
            }
        }
    }

    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Preload()
{
    if (coreIdx_ >= 8U) {
        return;
    }
    processNum = axisBS_ / 8U;
    resNum = axisBS_ - processNum * 8U;
    resLen = (resNum == 0U) ? 0U : 1U;
    startBs = 0U;
    endBs = 0U;
    if (coreIdx_ < resNum) {
        processNum += 1U;
        startBs = coreIdx_ * processNum;
        endBs = startBs + processNum;
    } else {
        startBs = coreIdx_ * processNum + resNum;
        endBs = startBs + processNum;
    }
    uint64_t selfRankAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_) + halfWinSize_ * bufferId_);
    localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(selfRankAddr));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    offsetIndex = 0U;
    sumFloatLocal_ = sumFloatBuf_.Get<float>();

    if (startBs != 0U) {
        offsetIndex = countOuterGlobal_.GetValue(startBs - 1U);
    }
}
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToServer()
{
    if (coreIdx_ >= 8U) {
        SyncAll<true>();
        return;
    }
    uint32_t count = startBs;
    for (uint32_t i = startBs; i < endBs; i++) {
        int flag = 0;
        Duplicate(sumFloatLocal_, 0.0f, axisH_);
        for (int j = 0; j < serverNum; j++) {
            int cntOuter = offsetOuterGlobal_.GetValue(i * serverNum + j);
            if (cntOuter < 0) {
                continue;
            }
            tmpUb_ = moeSumQueue_.AllocTensor<ExpandXType>();
            flag = 1;
#ifdef COMBINE_2SERVER_VERSION
            int offsetOnIpc = (i * axisH_ * sizeof(ExpandXType)) / sizeof(ExpandXType);
#else
            int offsetOnIpc = (cntOuter * axisH_ * sizeof(ExpandXType)) / sizeof(ExpandXType);
#endif
            uint64_t selfRankAddr = (uint64_t)(hccl_.GetWindowsInAddr(rankId_) + halfWinSize_ * bufferId_ +
                                               j * rankSizeOnWin_ * SERVER_RANK_SIZE);
            localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(selfRankAddr));
            DataCopy(tmpUb_, localInWindow_[offsetOnIpc], axisH_);
            moeSumQueue_.EnQue(tmpUb_);
            LocalTensor<ExpandXType> tmpOtherUb_ = moeSumQueue_.DeQue<ExpandXType>();
            // cast before muls
            Cast(rowTmpFloatLocal_, tmpOtherUb_, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            // add mulBufLocal to sumFloatBufLocal
            AscendC::Add(sumFloatLocal_, sumFloatLocal_, rowTmpFloatLocal_, axisH_);
            moeSumQueue_.FreeTensor<ExpandXType>(tmpOtherUb_);
        }
        PipeBarrier<PIPE_V>();
        if (!flag) {
            continue;
        }
        LocalTensor<ExpandXType> castUbIn = mulBuf_.Get<ExpandXType>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(castUbIn, sumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[i * axisH_], castUbIn, axisH_);
        count++;
        PipeBarrier<PIPE_V>();
    }

    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        AlltoAllDispatch();        // 所有核执行
        SumToWindow();             // 前serverNum个核执行
        AlltoAllServerDispatch();  // 前serverNum个核执行
        SetStatus();               // 0核执行
        Preload();                 // 前8核执行
        WaitDispatch();            // 前serverNum个核执行
        SumToServer();             // 前8核执行
        hccl_.Finalize();
    }
}

}  // namespace MoeDistributeCombineA2Impl
#endif  // MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
