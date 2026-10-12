#ifndef MOE_DISTRIBUTE_COMBINE_A5_LAYERED_H
#define MOE_DISTRIBUTE_COMBINE_A5_LAYERED_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_base.h"

namespace MoeDistributeCombineA5LayeredImpl {

constexpr int UB_ALIGN_SIZE = 32;
constexpr uint64_t CACHELINE_SIZE = 64;

#define TemplateMC2TypeA5LayeredClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA5LayeredFunc ExpandXType, ExpandIdxType

using namespace AscendC;
template <TemplateMC2TypeA5LayeredClass>
class MoeDistributeCombineA5Layered
{
public:
    constexpr static uint32_t BUFFER_NUM = 2U;
    constexpr static uint32_t STATE_OFFSET = 512U;
    constexpr static uint32_t STATE_SPACE_SIZE = 1024U * 1024U;
    constexpr static uint32_t UB_ALIGN = 32U;
    constexpr static uint32_t SELF_STATE_OFFSET = 512U * 1024U;
    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B16_PER_BLOCK = 16U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static uint32_t SERVER_RANK_SIZE = 8U;
    constexpr static uint32_t IPC_DATA_OFFSET = 4U * 1024U * 1024U;
    constexpr static uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;
    constexpr static uint32_t VEC_LEN = 256U;
    constexpr static uint32_t MAGIC_OFFSET = 2U * 1024U * 1024U - 32U * 32U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U;
    constexpr static uint64_t MB_SIZE = 1024UL * 1024UL;
    constexpr static uint32_t TBUF_SIZE = 185U * 1024U;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 0U;
    constexpr static uint32_t IPC_REDUCE_USED_CORE_NUM = 32U;
    constexpr static uint32_t WEIGHT_VALUE_NUM = 16U;
    constexpr static uint64_t GM2IPC_SYNC_FLAG = 12345ULL;
    constexpr static uint64_t RDMA_TOKEN_ARRIVED_FLAG = 123ULL;
    constexpr static uint64_t RDMA_TOKEN_END_FLAG = 321ULL;
    constexpr static uint32_t MAX_BS_NUM = 512U;
    constexpr static uint32_t FLAG_SINGLE_CNT = 4;
    constexpr static uint32_t FLAG_TOTAL_SIZE =
        MAX_BS_NUM * FLAG_SINGLE_CNT > IPC_DATA_OFFSET
            ? IPC_DATA_OFFSET
            : MAX_BS_NUM * FLAG_SINGLE_CNT;

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
    template <typename T>
    inline __aicore__ T CeilDiv(const T dividend, const T divisor)
    {
        return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
    }

    __aicore__ inline MoeDistributeCombineA5Layered(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                GM_ADDR tiling, GM_ADDR contextGM);
    __aicore__ inline void Process();
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                           uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo);

private:
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void GM2IPC();
    __aicore__ inline void WaitIPC();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void SumToServer();
    __aicore__ inline void Preload();

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;

    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> bufferIdGlobal_;
    GlobalTensor<int32_t> statusSpaceGlobal_;
    GlobalTensor<int32_t> readStateGlobal_;

    uint64_t shareAddreRank[8];

    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR expandXGM_;
    GM_ADDR expertIdsGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR sendCountGM_;
    GM_ADDR scalesGM_;
    GM_ADDR XOutGM_;
    __gm__ HcclAiRMAInfo *qp_info_{nullptr};

    GM_ADDR shareAddrGM_;
    GM_ADDR offsetInnerGM_;
    GM_ADDR countInnerGM_;
    GM_ADDR offsetOuterGM_;
    GM_ADDR countOuterGM_;

    GlobalTensor<int32_t> shareAddrGlobal_;
    GlobalTensor<uint64_t> shareFlagGlobal_;
    GlobalTensor<ExpandXType> shareMemGlobal_;
    GlobalTensor<ExpandXType> dstshareMemGlobal_;
    GlobalTensor<uint64_t> magicGlobal_;
    GlobalTensor<int32_t> offsetInnerGlobal_;
    GlobalTensor<int16_t> countInnerGlobal_;
    GlobalTensor<int32_t> offsetOuterGlobal_;
    GlobalTensor<int32_t> countOuterGlobal_;

    uint32_t countReL{0};
    uint32_t axisBS_{0};
    uint32_t globalBs{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
    uint32_t moeExpertNum_{0};
    uint32_t localMoeExpertNum_{0};
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
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    TBuf<> statusBuf_;

    int32_t sumTarget_{0};
    int32_t stateValue_{0};
    uint32_t startBs{0};
    uint32_t endBs{0};
    uint32_t processNum{0};
    uint32_t resNum{0};
    uint32_t resLen{0};
    uint32_t offsetIndex{0};
    uint32_t maxLocalBs{0};
    uint32_t stepCoreNum{0};
    uint64_t magicValue{0};
    LocalTensor<int32_t> offsetReduceLocal_;
    LocalTensor<int32_t> countReduceLocal_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
};

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo *)QpInfo)->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ *)(((__gm__ HcclAiRMAInfo *)QpInfo)->sqPtr +
                               destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo *)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAMem;
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t *)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t *)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;

    PipeBarrier<PIPE_ALL>();

    while (1) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);
    byte_4 |= 1U << 8U;

    *(__gm__ uint32_t *)(wqeAddr) = byte_4;
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen;
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U;
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)destDmaAddr;

    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) =
        (uint64_t)srcDmaAddr;

    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;
    doorBellInfo |= 0UL << 24UL;
    doorBellInfo |= (curHead % 65536UL) << 32UL;
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL;

    __gm__ uint64_t *doorBellAddr = (__gm__ uint64_t *)(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR XOut,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tiling, GM_ADDR contextGM)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tiling);
    expandXGM_ = expandX;
    expertIdsGM_ = expertIds;
    expandIdxGM_ = expandIdx;
    sendCountGM_ = sendCount;
    scalesGM_ = scales;
    XOutGM_ = XOut;
    rankId_ = tilingData.moeDistributeCombineV2Info.epRankId;
    axisBS_ = tilingData.moeDistributeCombineV2Info.bs;
    axisH_ = tilingData.moeDistributeCombineV2Info.h;
    axisK_ = tilingData.moeDistributeCombineV2Info.k;
    aivNum_ = tilingData.moeDistributeCombineV2Info.aivNum;
    moeExpertNum_ = tilingData.moeDistributeCombineV2Info.moeExpertNum;
    worldSize_ = tilingData.moeDistributeCombineV2Info.epWorldSize;

    globalBs = tilingData.moeDistributeCombineV2Info.globalBs;
    if (globalBs >= MAX_BS_NUM) {
        maxLocalBs = MAX_BS_NUM;
    } else {
        maxLocalBs = globalBs;
    }

    winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM;
    hccl_.InitV2(contextGM, &tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineV2TilingData, mc2CcTiling1));
    qp_info_ = (__gm__ HcclAiRMAInfo *)(((__gm__ HcclA2CombineOpParam *)contextGM)->aiRMAInfo);

    halfWinSize_ = RDMA_DATA_SIZE / 2U;
    IPC_DATA_SIZE = winContext_->winSize - RDMA_DATA_SIZE - IPC_DATA_OFFSET;
    dataSpaceSize_ = halfWinSize_ - STATE_SPACE_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataSpaceSize_ + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferIdGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();

    serverNum = worldSize_ / SERVER_RANK_SIZE;
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    readStateGlobal_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + dataSpaceSize_));
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    expandXRows_ = localMoeExpertNum_ * axisBS_ * worldSize_;
    rankSizeOnWin_ = static_cast<uint64_t>(dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE);
    statusSpaceGm_ = windowInGM_ + dataSpaceSize_;
    statusSpaceGlobal_.SetGlobalBuffer((__gm__ int32_t *)statusSpaceGm_);
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    stateOffsetOnWin_ = static_cast<uint64_t>(dataSpaceSize_ + rankId_ * STATE_OFFSET);
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));

    uint64_t winSizeMin = static_cast<uint64_t>(moeExpertNum_) * axisBS_ *
                              (static_cast<uint64_t>(axisHExpandXTypeSize_) +
                               static_cast<uint64_t>(EXTRA_TOKEN_INFO_NUM) * axisK_ * sizeof(uint32_t)) +
                          IPC_DATA_OFFSET + RDMA_DATA_SIZE;

    assert(winContext_->winSize >= winSizeMin,
           "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB.\n",
           winContext_->winSize / MB_SIZE, winSizeMin / MB_SIZE);

    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
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
    offset_inner_offset = send_counts_inner_offset + static_cast<uint64_t>(globalBs * serverNum / 2);
    send_counts_outer_offset = offset_inner_offset + static_cast<uint64_t>(globalBs * axisK_ * serverNum);
    offset_outer_offset = send_counts_outer_offset + static_cast<uint64_t>(axisBS_);
    share_offset = offset_outer_offset + static_cast<uint64_t>(axisBS_ * serverNum);

    shareAddrGM_ = sendCount + share_offset;
    offsetInnerGM_ = sendCount + offset_inner_offset;
    countInnerGM_ = sendCount + send_counts_inner_offset;
    offsetOuterGM_ = sendCount + offset_outer_offset;
    countOuterGM_ = sendCount + send_counts_outer_offset;

    shareAddrGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + share_offset);
    countInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(sendCount) + send_counts_inner_offset * 2);
    offsetInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_inner_offset);
    countOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + send_counts_outer_offset);
    offsetOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_outer_offset);

    PipeBarrier<PIPE_ALL>();
    for (int i = 0; i < 8; i++) {
        shareAddreRank[i] = reinterpret_cast<uint64_t>(
            RDMA_DATA_SIZE + hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i));
    }
    magicGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(shareAddreRank[rankId_ % SERVER_RANK_SIZE]));
    magicValue = magicGlobal_.GetValue(MAGIC_OFFSET / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::BuffInit()
{
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN_SIZE);
    ubLocal = rdmaInBuf_.Get<uint64_t>();
    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN_SIZE);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();
    tpipe_->InitBuffer(tBuf, TBUF_SIZE);
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::SplitCoreCal()
{
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

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::GM2IPC()
{
    ipcSliceSize = IPC_DATA_SIZE / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    ipcSliceNodeSize = ipcSliceSize * SERVER_RANK_SIZE;

    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    LocalTensor<ExpandIdxType> sendCountLocal =
        tBuf.GetWithOffset<int32_t>(RoundUp(moeExpertNum_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, B32_PER_BLOCK);

    LocalTensor<float> expandScalesLocal =
        tBuf.GetWithOffset<float>((maxLocalBs + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN, baseBuffOffset);
    baseBuffOffset += sizeof(float) * ((maxLocalBs + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);

    LocalTensor<ExpandXType> inUb = tBuf.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    LocalTensor<float> inUbTemp = inUb[axisH_].template ReinterpretCast<float>();

    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t localShareAddr = shareAddreRank[rankId_ % SERVER_RANK_SIZE];
    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
        uint64_t targetRankAddr = localShareAddr + static_cast<uint64_t>(dstRankId * ipcSliceSize + IPC_DATA_OFFSET);
        dstshareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)(targetRankAddr));
        uint32_t rankTokenNum = 0U;
        for (uint32_t expertId = 0U; expertId < localMoeExpertNum_; ++expertId) {
            uint32_t preCount = 0U;
            if (expertId != 0U || dstRankId != 0U) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            uint32_t tokenNum = sendCountLocal.GetValue(expertId * worldSize_ + dstRankId) - preCount;
            uint32_t startTokenAddr = preCount * axisH_;
            PipeBarrier<PIPE_ALL>();
            DataCopy(expandScalesLocal, expandScalesGlobal_[preCount], (tokenNum + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            for (uint32_t tokenId = 0U; tokenId < tokenNum; ++tokenId) {
                float scaleVal = expandScalesLocal.GetValue(tokenId);
                inUbTemp(0) = scaleVal;
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopy(inUb, expandXGlobal_[startTokenAddr], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopy(dstshareMemGlobal_[rankTokenNum * (axisH_ + WEIGHT_VALUE_NUM)], inUb,
                         axisH_ + WEIGHT_VALUE_NUM);
                startTokenAddr += axisH_;
                rankTokenNum++;
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    SyncAll<true>();
    if (coreIdx_ < SERVER_RANK_SIZE) {
        uint64_t targetAddr = shareAddreRank[coreIdx_ % SERVER_RANK_SIZE];
        shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)targetAddr);
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        inUb(0) = GM2IPC_SYNC_FLAG + magicValue;
        uint32_t flagOffset = rankId_ % SERVER_RANK_SIZE;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[flagOffset * FLAG_SINGLE_CNT], inUb, FLAG_SINGLE_CNT);
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::WaitIPC()
{
    uint32_t stepCoreNum_ = SERVER_RANK_SIZE;
    shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)shareAddreRank[rankId_ % SERVER_RANK_SIZE]);
    if (coreIdx_ < stepCoreNum_) {
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        uint32_t waitFlagAddr = coreIdx_ % stepCoreNum_;
        while (true) {
            DataCopy(inUb, shareFlagGlobal_[waitFlagAddr * FLAG_SINGLE_CNT], FLAG_SINGLE_CNT);
            PipeBarrier<PIPE_ALL>();
            if (inUb(0) >= (GM2IPC_SYNC_FLAG + magicValue)) {
                break;
            }
        }
        inUb(0) = 0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_[waitFlagAddr * FLAG_SINGLE_CNT], inUb, FLAG_SINGLE_CNT);
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::SumToWindow()
{
    uint32_t coreNumPerServer = stepCoreNum / serverNum;
    uint32_t serverId_ = coreIdx_ / coreNumPerServer;
    uint32_t targetRankId_ = rankId_ % SERVER_RANK_SIZE + serverId_ * SERVER_RANK_SIZE;

    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    LocalTensor<int16_t> countReduceLocal =
        tBuf.GetWithOffset<int16_t>(RoundUp(maxLocalBs, B16_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int16_t) * RoundUp(maxLocalBs, B16_PER_BLOCK);

    LocalTensor<int32_t> offsetReduceLocal =
        tBuf.GetWithOffset<int32_t>(RoundUp(maxLocalBs * axisK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(maxLocalBs * axisK_, B32_PER_BLOCK);

    LocalTensor<ExpandXType> dataInLocal = tBuf.GetWithOffset<ExpandXType>(
        axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    baseBuffOffset += sizeof(ExpandXType) * (axisH_ + WEIGHT_VALUE_NUM);

    LocalTensor<float> castInFloatLocal =
        tBuf.GetWithOffset<float>(RoundUp(axisHFloatSize_, VEC_LEN) / sizeof(float), baseBuffOffset);
    baseBuffOffset += RoundUp(axisHFloatSize_, VEC_LEN);

    LocalTensor<float> sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);
    LocalTensor<float> inUbTemp = dataInLocal[axisH_].template ReinterpretCast<float>();

    DataCopy(countReduceLocal, countInnerGlobal_[globalBs * serverId_], RoundUp(maxLocalBs, B16_PER_BLOCK));
    DataCopy(offsetReduceLocal, offsetInnerGlobal_[globalBs * axisK_ * serverId_],
             RoundUp(maxLocalBs * axisK_, B32_PER_BLOCK));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t rdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                   serverId_ * rankSizeOnWin_ * SERVER_RANK_SIZE);
    localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)rdmaAddr);
    LocalTensor<uint64_t> rdmaFlagLocal = statusBuf_.Get<uint64_t>();
    rdmaFlagLocal(0) = RDMA_TOKEN_ARRIVED_FLAG + magicValue;
    PipeBarrier<PIPE_ALL>();
    int offsetPre = 0;
    offsetIndex = 0U;

    uint32_t offsetIndexs[65];
    uint32_t copyNums[65];
    uint32_t dataOffsets[65];
    uint32_t totalCopyLen = 0;
    uint32_t processNum_ = 0;
    for (uint32_t i = 0U; i < maxLocalBs; i++) {
        if ((i % coreNumPerServer) == (coreIdx_ % coreNumPerServer)) {
            int offsetCur = static_cast<int32_t>(countReduceLocal.GetValue(i));
            uint32_t dataOffset = i * axisH_;
            if (i != 0U) {
                offsetPre = static_cast<int32_t>(countReduceLocal.GetValue(i - 1));
            }
            int copyNum = offsetCur - offsetPre;
            if (copyNum <= 0) {
                break;
            }
            offsetIndex = static_cast<uint32_t>(offsetPre);
            offsetIndexs[processNum_] = offsetIndex;
            copyNums[processNum_] = static_cast<uint32_t>(copyNum);
            dataOffsets[processNum_] = dataOffset;
            totalCopyLen += static_cast<uint32_t>(copyNum);
            processNum_++;
        }
    }

    uint32_t processTokenNum = 0;
    uint32_t offsetIndexStart = offsetIndexs[processTokenNum];
    offsetIndex = offsetIndexs[processTokenNum];
    uint32_t copyNum = copyNums[processTokenNum];
    uint32_t dataOffset = dataOffsets[processTokenNum];
    uint32_t tokenOffset = 0;

    for (uint32_t i = 0U; i < totalCopyLen; i++) {
        uint32_t targetIpcRank = offsetReduceLocal.GetValue(offsetIndex) / (globalBs * axisK_);
        uint32_t targetIpcOffset =
            offsetReduceLocal.GetValue(offsetIndex) % (globalBs * axisK_) * (axisH_ + WEIGHT_VALUE_NUM);
        uint64_t copyAddr = shareAddreRank[targetIpcRank % SERVER_RANK_SIZE] +
                            static_cast<uint64_t>(targetRankId_ * ipcSliceSize) +
                            static_cast<uint64_t>(IPC_DATA_OFFSET);
        shareMemGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)copyAddr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(dataInLocal, shareMemGlobal_[targetIpcOffset], axisH_ + WEIGHT_VALUE_NUM);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        float scaleVal = inUbTemp(0);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(castInFloatLocal, dataInLocal, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        if ((offsetIndex - offsetIndexStart) == 0U) {
            Muls(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);
        } else {
            Axpy(sumFloatLocal, castInFloatLocal, scaleVal, axisH_);
        }
        offsetIndex += 1U;
        PipeBarrier<PIPE_V>();
        if ((offsetIndex - offsetIndexStart) == copyNum) {
            tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
            PipeBarrier<PIPE_V>();
            Cast(dataInLocal, sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(localOutWindow_[tokenOffset * axisH_], dataInLocal, axisH_);
            PipeBarrier<PIPE_MTE3>();
            DataCopy(shareFlagGlobal_[(serverId_ + 1) * FLAG_TOTAL_SIZE + tokenOffset * FLAG_SINGLE_CNT],
                     rdmaFlagLocal, FLAG_SINGLE_CNT);
            processTokenNum++;
            offsetIndex = offsetIndexs[processTokenNum];
            copyNum = copyNums[processTokenNum];
            dataOffset = dataOffsets[processTokenNum];
            offsetIndexStart = offsetIndex;
        }
    }
    PipeBarrier<PIPE_ALL>();
    rdmaFlagLocal(0) = RDMA_TOKEN_END_FLAG + magicValue;
    tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
    DataCopy(shareFlagGlobal_[(serverId_ + 1) * FLAG_TOTAL_SIZE + tokenOffset * FLAG_SINGLE_CNT], rdmaFlagLocal,
             FLAG_SINGLE_CNT);
    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::AlltoAllServerDispatch()
{
    LocalTensor<uint64_t> checkRdmaLocal = statusBuf_.Get<uint64_t>();
    LocalTensor<ExpandXType> tmpLowUb_ = tBuf.Get<ExpandXType>();
    uint32_t checkServer = coreIdx_ - stepCoreNum;
    GlobalTensor<ExpandXType> aivSrcGlobal;
    GlobalTensor<ExpandXType> aivDstGlobal;
    uint32_t tragRankId = rankId_ % SERVER_RANK_SIZE + SERVER_RANK_SIZE * checkServer;
    uint32_t copySum = 0;
    uint32_t copyOnceNum = 1;
    uint32_t copyLen_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
    uint32_t copyLenAlign_ = copyLen_;
    uint32_t selfServerID = rankId_ / SERVER_RANK_SIZE;
    bool stopFlag = false;
    uint32_t cpNum = 0;

    uint64_t srcrdmaAddr = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                      checkServer * rankSizeOnWin_ * SERVER_RANK_SIZE);
    uint64_t dstrdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(tragRankId) + halfWinSize_ * bufferId_ +
                                      (rankId_ / SERVER_RANK_SIZE) * rankSizeOnWin_ * SERVER_RANK_SIZE);
    while (!stopFlag) {
        for (uint32_t i = 0U; i < copyOnceNum; i++) {
            while (true) {
                DataCopy(checkRdmaLocal[64],
                         shareFlagGlobal_[(checkServer + 1) * FLAG_TOTAL_SIZE + copySum * FLAG_SINGLE_CNT],
                         FLAG_SINGLE_CNT);
                PipeBarrier<PIPE_ALL>();
                if (checkRdmaLocal.GetValue(64) == (RDMA_TOKEN_ARRIVED_FLAG + magicValue)) {
                    copySum++;
                    break;
                } else if (checkRdmaLocal.GetValue(64) == (RDMA_TOKEN_END_FLAG + magicValue) || copySum == maxLocalBs) {
                    stopFlag = true;
                    break;
                }
            }
            PipeBarrier<PIPE_ALL>();
            if (stopFlag) {
                break;
            }
        }
        if (copySum > 0U) {
            if (rankId_ != tragRankId) {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(dstrdmaAddr));
                AIVRDMAPostSend((GM_ADDR)(srcrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)),
                                (GM_ADDR)(dstrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)), tragRankId,
                                copyLen_ * copyOnceNum, qp_info_);
            } else {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(dstrdmaAddr));
                cpNum = axisH_;
                for (uint32_t k = 0U; k < copyOnceNum; k++) {
                    DataCopy(tmpLowUb_,
                             aivSrcGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXType)],
                             cpNum);
                    PipeBarrier<PIPE_ALL>();
                    DataCopy(aivDstGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXType)],
                             tmpLowUb_, cpNum);
                }
            }
        }
    }
    if (rankId_ != tragRankId) {
        AIVRDMAPostSend((GM_ADDR)((uint64_t)(readStateGlobal_.GetPhyAddr())),
                        (GM_ADDR)((uint64_t)(hccl_.GetWindowsInAddr(tragRankId) + halfWinSize_ * bufferId_ +
                                             dataSpaceSize_ + selfServerID * STATE_OFFSET)),
                        tragRankId, 32, qp_info_);
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::WaitDispatch()
{
    if ((coreIdx_ < serverNum) && (coreIdx_ != (rankId_ / SERVER_RANK_SIZE))) {
        LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
        uint32_t readNum = 1U;
        DataCopyParams intriParams{static_cast<uint16_t>(readNum), 1, 15, 0};
        while (true) {
            DataCopy(statusTensor, statusSpaceGlobal_[(coreIdx_)*STATE_OFFSET / sizeof(int32_t)], intriParams);
            PipeBarrier<PIPE_ALL>();
            int32_t sumOfFlag = statusTensor.GetValue(0);
            if (sumOfFlag == sumTarget_) {
                break;
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::Preload()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        return;
    }
    processNum = axisBS_ / reduceCore;
    resNum = axisBS_ - processNum * reduceCore;
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

    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    offsetReduceLocal_ = tBuf.GetWithOffset<int32_t>(
        RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);
    baseBuffOffset += sizeof(uint32_t) * RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t)));

    countReduceLocal_ =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);

    DataCopy(offsetReduceLocal_, offsetOuterGlobal_,
             RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t))));
    DataCopy(countReduceLocal_, countOuterGlobal_, RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t))));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    offsetIndex = 0U;
    if (startBs != 0U) {
        offsetIndex = countReduceLocal_.GetValue(startBs - 1U);
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::SumToServer()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        SyncAll<true>();
        return;
    }

    uint32_t baseBuffOffset = sizeof(uint32_t) * RoundUp(axisBS_ * serverNum, (uint32_t)(UB_ALIGN / sizeof(int32_t))) +
                              sizeof(int32_t) * RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));

    LocalTensor<float> sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);
    LocalTensor<ExpandXType> sumFpAndBfLocal = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    LocalTensor<ExpandXType> dataIn = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(ExpandXType);
    LocalTensor<float> castFp32 = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);

    for (uint32_t i = startBs; i < endBs; i++) {
        int offsetPre = 0;
        int offsetCur = countReduceLocal_.GetValue(i);
        if (i != 0U) {
            offsetPre = countReduceLocal_.GetValue(i - 1);
        }
        PipeBarrier<PIPE_ALL>();
        int copyNum = offsetCur - offsetPre;
        if (!copyNum) {
            break;
        }
        Duplicate(sumFloatLocal, 0.0f, axisH_);
        for (int j = 0; j < copyNum; j++) {
            int offsetOnIpc =
                (offsetReduceLocal_.GetValue(offsetIndex) / axisBS_ * rankSizeOnWin_ * SERVER_RANK_SIZE +
                 offsetReduceLocal_.GetValue(offsetIndex) % axisBS_ * axisH_ * sizeof(ExpandXType)) /
                sizeof(ExpandXType);
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            DataCopy(dataIn, localInWindow_[offsetOnIpc], axisH_);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(castFp32, dataIn, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatLocal, sumFloatLocal, castFp32, axisH_);
            offsetIndex++;
        }
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Cast(sumFpAndBfLocal, sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[i * axisH_], sumFpAndBfLocal, axisH_);
        PipeBarrier<PIPE_V>();
    }

    SyncAll<true>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeCombineA5Layered<TemplateMC2TypeA5LayeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        GM2IPC();
        WaitIPC();
        stepCoreNum = IPC_REDUCE_USED_CORE_NUM;
        if (coreIdx_ < stepCoreNum) {
            SumToWindow();
        } else if (coreIdx_ < (stepCoreNum + serverNum)) {
            AlltoAllServerDispatch();
        } else {
            SyncAll<true>();
        }
        if (coreIdx_ == 0U) {
            magicGlobal_.SetValue(MAGIC_OFFSET / sizeof(uint64_t), magicValue + 1);
            PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(
                magicGlobal_[MAGIC_OFFSET / sizeof(uint64_t)]);
            bufferIdGlobal_(0) = bufferId_ ^ 1;
            PipeBarrier<PIPE_ALL>();
            AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(bufferIdGlobal_[0]);
        }
        Preload();
        WaitDispatch();
        SumToServer();
        hccl_.Finalize();
    }
}

}  // namespace MoeDistributeCombineA5LayeredImpl
#endif  // MOE_DISTRIBUTE_COMBINE_A5_LAYERED_H
