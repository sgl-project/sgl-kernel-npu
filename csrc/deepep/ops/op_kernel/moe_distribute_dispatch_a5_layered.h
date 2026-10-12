#ifndef MOE_DISTRIBUTE_DISPATCH_A5_LAYERED_H
#define MOE_DISTRIBUTE_DISPATCH_A5_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_base.h"
#ifdef __DAV_C310__
#include "quantize_functions.h"
#endif

namespace MoeDistributeDispatchA5LayeredImpl {
#define TemplateMC2TypeA5LayeredClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsMxQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA5LayeredFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsMxQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeA5LayeredClass>
class MoeDistributeDispatchA5Layered
{
public:
    constexpr static uint32_t STATE_OFFSET = 512;
    constexpr static uint32_t STATUS_SIZE_LAYERED = 1024 * 1024;
    constexpr static uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
    constexpr static uint32_t SERVER_RANK_SIZE = 8;
    constexpr static uint32_t B64_PER_BLOCK = 4;
    constexpr static uint32_t B16_PER_BLOCK = 16;
    constexpr static uint32_t UB_32B_ALIGN = 32;
    constexpr static uint32_t BITS32_PER_BLOCK = 8U;
    constexpr static uint32_t FP4_ELEMS_PER_BYTE = 2U;
    constexpr static uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint32_t IPC_MAGIC_OFFSET = 2 * 1024 * 1024 - 128 * 32;
    constexpr static uint32_t IPC_FLAG_OFFSET = 1 * 1024 * 1024;
    constexpr static uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
    constexpr static uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
    constexpr static uint32_t MTU_SIZE = 4 * 1024;
    constexpr static uint32_t IPC_BUFF_ALIGN = 512;
    constexpr static int32_t IPC_FLAG_STEP_1 = 0x0d0d0d0d;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
    constexpr static uint32_t MAX_BS_NUM = 512U;
    constexpr static uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
    constexpr static uint64_t SHOULD_SEND_FLAG_VALUE = 0x0f0f0f0f;
    constexpr static uint64_t END_OF_WRITE_FLAG_VALUE = 0xffffffff;
    constexpr static uint32_t FLAG_SIZE = 64;
    constexpr static uint32_t FINISH_STATUS = 0;
    constexpr static uint32_t WAIT_STATUS = 1;
    constexpr static uint32_t ARRIVAL_STATUS = 2;
    constexpr static uint32_t SKIP_STATUS = 3;
    constexpr static uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U;

    template <typename T>
    static __aicore__ inline T CeilDiv(const T dividend, const T divisor)
    {
        return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
    }
    template <typename T>
    static __aicore__ inline T Ceil(const T val, const T align)
    {
        return (val + align - 1) / align;
    }
    template <typename T>
    static __aicore__ inline T Ceil32(const T val)
    {
        return (val + (T)31) / (T)32;
    }
    template <typename T>
    static __aicore__ inline T Align256(const T val)
    {
        return Ceil(val, (T)256) * 256;
    }
    template <typename T>
    static __aicore__ inline T Align128(const T val)
    {
        return Ceil(val, (T)128) * 128;
    }
    template <typename T>
    static __aicore__ inline T Align32(const T val)
    {
        return Ceil(val, (T)32) * 32;
    }
    static __aicore__ inline uint32_t Align2(const uint32_t val)
    {
        return (val + 1) & ~1U;
    }

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

public:
    __aicore__ inline MoeDistributeDispatchA5Layered(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                GM_ADDR tilingGM, GM_ADDR contextGM0);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void SendDataToServer(uint32_t destServerId);
    __aicore__ inline void CreateInnerReduceInfo(uint32_t serverIdx);
    __aicore__ inline void CreateOuterReduceInfo();
    __aicore__ inline void Win2Ipc();
    __aicore__ inline void Ipc2Out();
    __aicore__ inline void WaitIpcFlag(int32_t flagVal = 1);
    __aicore__ inline void SetIpcFlag(int32_t flagVal = 1);
    __aicore__ inline void CleanUp();

    __aicore__ inline uint32_t GetExpRank(uint32_t expertId);
    __aicore__ inline void QuantProcess(uint32_t sendTokenNum, LocalTensor<XType> xTokenLt,
                                        LocalTensor<float> tokenCastLt);
    __aicore__ inline uint32_t GetArrivedTokenInfo(uint32_t serverIdx, uint32_t tokenIdx, bool justExpInfo,
                                                   LocalTensor<uint8_t> localUB_U8);
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                           uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo);
    __aicore__ inline uint32_t GetSelfServerTokenInfo(uint32_t tokenIdx, bool justExpInfo,
                                                      LocalTensor<uint8_t> localUB_U8);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<float> weightsOutGt;
    GlobalTensor<uint64_t> sendStatusTensor_;
    GlobalTensor<uint8_t> sendTokensU8Tensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<uint32_t> expertToServerGlobalTensor_;
    GlobalTensor<uint64_t> readStatusTensor_;
    GlobalTensor<uint64_t> tokenAddrFlagStructGlobalU64Tensor_;

    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int16_t> expertIdsI16Tensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertToServerCntTensor_;
    LocalTensor<uint32_t> expertToServerIdxTensor_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;

    TBuf<> statusBuf_;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;

    __gm__ HcclAiRMAInfo *qp_info_;
    GM_ADDR expandXGM_;
    GM_ADDR expandIdxGM_;
    GM_ADDR weightsGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR dataBatchWriteInfo_;
    GM_ADDR expertToServerCntGM_;
    GM_ADDR shareAddrs[8];
    GM_ADDR tokenAddrFlagStructGM_;

    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t alignK_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t serverId_{0};
    uint32_t aivId_{0};
    uint32_t moeExpertNum_{0};
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
    uint32_t tokenUbSize_{0};

    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
    uint32_t weightLenInStruct_{0};
    uint32_t realLenInStruct_{0};
    uint32_t cntLenInStruct_{0};
    uint32_t tokenOffsetInStruct_{0};
    uint32_t expOffsetInStruct_{0};
    uint32_t weightOffsetInStruct_{0};
    uint32_t cntOffsetInStruct_{0};
    uint32_t scaleOffsetInStruct_{0};
    uint32_t scaleLenInStruct_{0};
    uint32_t flagLenInStruct_{0};
    uint32_t flagOffsetInStruct_{0};
    uint64_t magicVal_{0};

    uint64_t combineInnerCntOffset;
    uint64_t combineInnerCntIndexOffset;
    uint64_t combineOuterCntOffset;
    uint64_t combineOuterCntIndexOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
};

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM, GM_ADDR contextGM0)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);

    hccl_.InitV2(contextGM0, &tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchV2TilingData, mc2CcTiling1));

    winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchV2Info.epRankId;
    serverId_ = rankId_ / SERVER_RANK_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);
    qp_info_ = (__gm__ HcclAiRMAInfo *)(((__gm__ HcclA2CombineOpParam *)contextGM0)->aiRMAInfo);

    axisBS_ = tilingData.moeDistributeDispatchV2Info.bs;
    globalBs_ = tilingData.moeDistributeDispatchV2Info.globalBs;
    axisH_ = tilingData.moeDistributeDispatchV2Info.h;
    axisK_ = tilingData.moeDistributeDispatchV2Info.k;
    alignK_ = RoundUp(axisK_, BITS32_PER_BLOCK);
    aivNum_ = tilingData.moeDistributeDispatchV2Info.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchV2Info.epWorldSize;
    moeExpertNum_ = tilingData.moeDistributeDispatchV2Info.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    totalSize_ = winContext_->winSize;
    totalWinSize_ = RDMA_DATA_SIZE;
    shareMemOffset_ = totalWinSize_;
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - STATUS_SIZE_LAYERED;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchV2Info.expertTokenNumsType;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;

    uint64_t winSizeMin =
        moeExpertNum_ * axisBS_ * (axisH_ * sizeof(XType) + EXTRA_TOKEN_INFO_NUM * alignK_ * sizeof(uint32_t)) +
        IPC_DATA_OFFSET + RDMA_DATA_SIZE;
    assert(winContext_->winSize >= winSizeMin,
           "The HCCL_BUFFSIZE is %lluMB, the min value should be %lluMB. \
        epWorldSize:%u, epRankId:%u, moeExpertNum:%u, quantMode:%u, globalBs:%u, bs:%u, k:%u, h:%u, aivNum:%u, \
        isQuant:%d, totalUbSize:%llu, expertTokenNumsType:%u\n",
           winContext_->winSize / MB_SIZE, winSizeMin / MB_SIZE, tilingData.moeDistributeDispatchV2Info.epWorldSize,
           tilingData.moeDistributeDispatchV2Info.epRankId, tilingData.moeDistributeDispatchV2Info.moeExpertNum,
           tilingData.moeDistributeDispatchV2Info.quantMode, tilingData.moeDistributeDispatchV2Info.globalBs,
           tilingData.moeDistributeDispatchV2Info.bs, tilingData.moeDistributeDispatchV2Info.k,
           tilingData.moeDistributeDispatchV2Info.h, tilingData.moeDistributeDispatchV2Info.aivNum,
           tilingData.moeDistributeDispatchV2Info.isQuant, tilingData.moeDistributeDispatchV2Info.totalUbSize,
           tilingData.moeDistributeDispatchV2Info.expertTokenNumsType);

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;
    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;

    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_));
    }
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;

    if constexpr (IsMxQuant) {
        uint32_t copyOutAxisH = axisH_;
        if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                      Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
            copyOutAxisH = CeilDiv(axisH_, 2U);
        }
        tokenLenInStruct_ = Align256(copyOutAxisH) * sizeof(ExpandXOutType);
        scaleLenInStruct_ = Align2(Ceil32(axisH_)) * sizeof(uint8_t);
    } else {
        tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);
        scaleLenInStruct_ = UB_32B_ALIGN;
    }
    expLenInStruct_ = alignK_ * sizeof(uint32_t);
    weightLenInStruct_ = alignK_ * sizeof(uint32_t);
    realLenInStruct_ = axisK_ * sizeof(uint32_t);
    flagLenInStruct_ = UB_32B_ALIGN;
    tokenStructLen_ = flagLenInStruct_ + tokenLenInStruct_ + expLenInStruct_ + weightLenInStruct_ + scaleLenInStruct_;

    flagOffsetInStruct_ = 0;
    tokenOffsetInStruct_ = flagLenInStruct_;
    expOffsetInStruct_ = tokenOffsetInStruct_ + tokenLenInStruct_;
    weightOffsetInStruct_ = expOffsetInStruct_ + expLenInStruct_;
    scaleOffsetInStruct_ = weightOffsetInStruct_ + weightLenInStruct_;

    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    weightsOutGt.SetGlobalBuffer((__gm__ float *)(expandIdxOut));
    expertTokenNumsOutGM_ = expertTokenNumsOut;
    epRecvCountsGM_ = epRecvCountsOut;
    expandXGM_ = x;
    expandIdxGM_ = expertIds;
    weightsGM_ = scales;

    sendTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + WIN_SIZE));
    readStatusTensor_.SetGlobalBuffer((__gm__ uint64_t *)(windowInGM_ + WIN_SIZE));

    tokenAddrFlagStructGM_ = workspaceGM;
    tokenAddrFlagStructGlobalU64Tensor_.SetGlobalBuffer((__gm__ uint64_t *)(tokenAddrFlagStructGM_),
                                                        axisBS_ * FLAG_SIZE);

    combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int16_t);
    combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t);
    combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;

    tpipe_->InitBuffer(statusBuf_, FLAG_SIZE);

    tpipe_->InitBuffer(rdmaInBuf_, UB_32B_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_32B_ALIGN);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();

    tpipe_->InitBuffer(tBuf, TBUF_SIZE);

    uint32_t expertIdsSize = RoundUp(expertIdsCnt_ * static_cast<uint32_t>(sizeof(int16_t)), UB_32B_ALIGN);
    tokenUbSize_ = TBUF_SIZE - TBUF_TEMP_OFFSET - expertIdsSize;
    expertIdsI16Tensor_ =
        tBuf.GetWithOffset<int16_t>(axisBS_ * alignK_, tokenUbSize_ + TBUF_TEMP_OFFSET);

    if (aivId_ == 0) {
        sendStatusTensor_.SetValue(0, END_OF_WRITE_FLAG_VALUE);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            sendStatusTensor_);
    }

    LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
    GlobalTensor<uint64_t> magicGt;
    magicGt.SetGlobalBuffer((__gm__ uint64_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_MAGIC_OFFSET) +
                            aivId_ * UB_32B_ALIGN / sizeof(uint64_t));
    DataCopy(tempLocal, magicGt, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    tempLocal(0) += 1;
    magicVal_ = tempLocal(0);
    DataCopy(magicGt, tempLocal, UB_32B_ALIGN / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo *)QpInfo)->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ *)(((__gm__ HcclAiRMAInfo *)QpInfo)->sqPtr +
                               destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAWQ));
    auto mem_info_table = ((__gm__ HcclAiRMAInfo *)QpInfo)->memPtr;
    auto sizeof_memdetail = ((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAMem;
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo *)QpInfo)->curRankId);
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
__aicore__ inline void
MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::CreateInnerReduceInfo(uint32_t serverIdx)
{
    uint32_t curServerId = serverIdx;
    uint32_t currServerExpBegin = rankId_ / 8 * moeExpertNumInServer_;
    uint32_t currServerExpEnd = currServerExpBegin + moeExpertNumInServer_;
    uint32_t tokenOccurNum = 0;
    uint32_t expOccurNum = 0;
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;

    uint32_t tokenStatus = WAIT_STATUS;
    uint32_t selfTokenIdx = 0;
    LocalTensor<uint8_t> localUB_U8 =
        tBuf.GetWithOffset<uint8_t>(MAX_BS_NUM * alignK_ * sizeof(int32_t), IPC_BUFF_ALIGN);
    LocalTensor<int32_t> localUB_32 = localUB_U8.template ReinterpretCast<int32_t>();

    uint32_t tokenIdx = 0;
    while (tokenStatus != FINISH_STATUS) {
        if (serverId_ == serverIdx) {
            tokenStatus = GetSelfServerTokenInfo(selfTokenIdx, true, localUB_U8[tokenIdx * expLenInStruct_]);
            if (tokenStatus == SKIP_STATUS || tokenStatus == ARRIVAL_STATUS) selfTokenIdx++;
        } else {
            tokenStatus = GetArrivedTokenInfo(curServerId, tokenIdx, true, localUB_U8[tokenIdx * expLenInStruct_]);
        }
        PipeBarrier<PIPE_ALL>();
        if (tokenStatus != ARRIVAL_STATUS) {
            continue;
        } else {
            tokenIdx += 1;
        }
    }

    uint32_t realBS = tokenIdx;
    if (realBS == 0) {
        uint32_t copyTokenNum = aivNum_ < globalBs_ ? aivNum_ : globalBs_;
        LocalTensor<int16_t> zeroTemp = tBuf.GetWithOffset<int16_t>(copyTokenNum * sizeof(int16_t), 0);
        Duplicate<int16_t>(zeroTemp, 0, RoundUp(copyTokenNum, B16_PER_BLOCK));
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<int16_t> combineInnerCnt;
        combineInnerCnt.SetGlobalBuffer(
            (__gm__ int16_t *)(epRecvCountsGM_ + combineInnerCntOffset + globalBs_ * curServerId * sizeof(int16_t)));
        DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint32_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(combineInnerCnt, zeroTemp, innerCntWriteCountsParams);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    LocalTensor<int32_t> localUB =
        tBuf.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, BITS32_PER_BLOCK), baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int16_t> combineReduceInfo =
        tBuf.GetWithOffset<int16_t>(moeExpertNumInServer_ * realBS, baseBuffOffset);

    baseBuffOffset += sizeof(int16_t) * RoundUp(moeExpertNumInServer_ * realBS, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> expCntMap = tBuf.GetWithOffset<int32_t>(moeExpertNumInServer_, baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNumInServer_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> tokenOffset =
        tBuf.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, BITS32_PER_BLOCK), baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int32_t> innerOffsetLt =
        tBuf.GetWithOffset<int32_t>(RoundUp(realBS * alignK_, BITS32_PER_BLOCK), baseBuffOffset);

    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * alignK_, TBUF_OFFSET_ALIGN_B32_CNT);
    LocalTensor<int16_t> innerCntLt =
        tBuf.GetWithOffset<int16_t>(RoundUp(realBS + aivNum_, B16_PER_BLOCK), baseBuffOffset);

    Duplicate<int16_t>(combineReduceInfo, int16_t(-1), moeExpertNumInServer_ * realBS);
    Duplicate<int32_t>(expCntMap, int32_t(0), moeExpertNumInServer_);
    Duplicate<int32_t>(tokenOffset, int32_t(0), realBS * alignK_);
    Duplicate<int16_t>(innerCntLt, 0, RoundUp(realBS + aivNum_, B16_PER_BLOCK));
    Duplicate<int32_t>(innerOffsetLt, 0, (realBS)*alignK_);

    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIdx = 0; tokenIdx < realBS; tokenIdx++) {
        for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
            int32_t expId = localUB_32(tokenIdx * alignK_ + expIdx);
            if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                int32_t expIdInServer = expId % moeExpertNumInServer_;
                uint32_t offsetInExp = expCntMap(expIdInServer);
                expCntMap(expIdInServer) += 1;
                combineReduceInfo(expIdInServer * realBS + offsetInExp) = static_cast<uint16_t>(tokenIdx);
                tokenOffset(tokenIdx * axisK_ + expIdx) = offsetInExp;
            }
        }
    }

    for (uint32_t expIdx = 0; expIdx < moeExpertNumInServer_; expIdx++) {
        if (expIdx % localMoeExpertNum_ == 0) {
            continue;
        }
        expCntMap(expIdx) += expCntMap(expIdx - 1);
    }

    for (uint32_t expBlockId = 0; expBlockId < moeExpertNumInServer_; expBlockId++) {
        uint32_t validCnt = (expBlockId % localMoeExpertNum_ == 0)
                                ? expCntMap(expBlockId)
                                : (expCntMap(expBlockId) - expCntMap(expBlockId - 1));
        for (uint32_t tokenIdx = 0; tokenIdx < validCnt; tokenIdx++) {
            uint32_t tokenId = static_cast<uint32_t>(combineReduceInfo(expBlockId * realBS + tokenIdx));
            if (tokenId == -1) {
                continue;
            }
            for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
                uint32_t expId = localUB_32(tokenId * alignK_ + expIdx);
                if (expId >= currServerExpBegin && expId < currServerExpEnd) {
                    uint32_t expIdInServer = expId % moeExpertNumInServer_;
                    uint32_t rankIdInServer = expIdInServer / localMoeExpertNum_;
                    combineReduceInfo(expIdInServer * realBS + tokenOffset(tokenId * axisK_ + expIdx)) = -1;
                    innerCntLt(tokenOccurNum) += 1;
                    innerOffsetLt(expOccurNum) =
                        (expIdInServer % localMoeExpertNum_ == 0) ? 0 : expCntMap(expIdInServer - 1);
                    innerOffsetLt(expOccurNum) += rankIdInServer * globalBs_ * axisK_;
                    innerOffsetLt(expOccurNum) += tokenOffset(tokenId * axisK_ + expIdx);
                    expOccurNum += 1;
                }
            }
            tokenOccurNum += 1;
        }
    }
    for (uint32_t tokenIdx = 1; tokenIdx < realBS; ++tokenIdx) {
        innerCntLt(tokenIdx) += innerCntLt(tokenIdx - 1);
    }
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<int16_t> combineInnerCnt;

    combineInnerCnt.SetGlobalBuffer(
        (__gm__ int16_t *)(epRecvCountsGM_ + combineInnerCntOffset + globalBs_ * curServerId * sizeof(int16_t)));
    uint32_t copyTokenNum = (realBS + aivNum_) < globalBs_ ? (realBS + aivNum_) : globalBs_;
    DataCopyExtParams innerCntWriteCountsParams{1, static_cast<uint16_t>(copyTokenNum * sizeof(int16_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineInnerCnt, innerCntLt, innerCntWriteCountsParams);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<int32_t> combineInnerOffset;
    combineInnerOffset.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineInnerCntIndexOffset +
                                                          globalBs_ * axisK_ * curServerId * sizeof(int32_t)));

    DataCopyExtParams innerOffsetWriteCountsParams{1, static_cast<uint32_t>(realBS * axisK_ * sizeof(int32_t)), 0, 0,
                                                   0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineInnerOffset, innerOffsetLt, innerOffsetWriteCountsParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::CreateOuterReduceInfo()
{
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;

    LocalTensor<int32_t> miniExpIds = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> miniServerExpIds =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineCnt_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineCntIdx_ =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffset_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffsetIdx_ =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerCntLt = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerOffsetLt =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * axisK_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> expertIdsI32Tensor =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK), baseBuffOffset);

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor, expertIdsGMTensor_, expCopyParams, expPadParams);

    Duplicate<int32_t>(miniExpIds, int32_t(moeExpertNum_), RoundUp(axisBS_, BITS32_PER_BLOCK));
    Duplicate<int32_t>(miniServerExpIds, int32_t(moeExpertNum_), RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK));
    Duplicate<int32_t>(combineCnt_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(combineOffset_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(outerCntLt, 0, RoundUp(axisBS_, BITS32_PER_BLOCK));
    Duplicate<int32_t>(outerOffsetLt, 0, RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK));

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t expertIndex = 0; expertIndex < expertIdsCnt_; ++expertIndex) {
        uint32_t tokenIdx = expertIndex / axisK_;
        uint32_t expId = expertIdsI32Tensor(expertIndex);
        uint32_t expServerId = expId / moeExpertNumInServer_;

        uint32_t miniExpId = miniExpIds(tokenIdx);
        miniExpIds(tokenIdx) = (expId < miniExpId) ? expId : miniExpId;

        if (miniServerExpIds(tokenIdx * serverNum + expServerId) > expId) {
            miniServerExpIds(tokenIdx * serverNum + expServerId) = expId;
        }

        if (expertIndex % axisK_ != axisK_ - 1) {
            continue;
        }
        combineCntIdx_(tokenIdx) = combineCnt_(miniExpId);
        combineCnt_(miniExpId) += 1;

        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            uint32_t miniServerExpId = miniServerExpIds(tokenIdx * serverNum + serverIdx);
            if (miniServerExpId != moeExpertNum_) {
                combineOffsetIdx_(tokenIdx * serverNum + serverIdx) = combineOffset_(miniServerExpId);
                combineOffset_(miniServerExpId) += 1;
            }
        }
    }
    for (uint32_t expertIndex = 1; expertIndex < moeExpertNum_; ++expertIndex) {
        combineCnt_(expertIndex) += combineCnt_(expertIndex - 1);
        combineOffset_(expertIndex) += combineOffset_(expertIndex - 1);
    }

    uint32_t outerOffsetIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < axisBS_; ++tokenIdx) {
        uint32_t miniExpId = miniExpIds(tokenIdx);
        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            uint32_t miniServerExpId = miniServerExpIds(tokenIdx * serverNum + serverIdx);
            if (miniServerExpId == moeExpertNum_) {
                continue;
            }
            outerCntLt(tokenIdx) += 1;
            uint32_t preServerCnt = (serverIdx == 0) ? 0 : combineOffset_(serverIdx * moeExpertNumInServer_ - 1);
            uint32_t serverBaseCnt = serverIdx * axisBS_;
            uint32_t preTokenCnt = (miniServerExpId == 0) ? 0 : combineOffset_(miniServerExpId - 1);
            uint32_t tokenOffset =
                preTokenCnt - preServerCnt + combineOffsetIdx_(tokenIdx * serverNum + serverIdx) + serverBaseCnt;
            outerOffsetLt(outerOffsetIdx) = tokenOffset;
            outerOffsetIdx++;
        }
    }

    for (uint32_t tokenIdx = 1; tokenIdx < axisBS_; ++tokenIdx) {
        outerCntLt(tokenIdx) += outerCntLt(tokenIdx - 1);
    }

    GlobalTensor<int32_t> combineOuterCnt;
    combineOuterCnt.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineOuterCntOffset));

    DataCopyExtParams outerCntWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterCnt, outerCntLt, outerCntWriteCountsParams);

    GlobalTensor<int32_t> combineOuterOffset;

    PipeBarrier<PIPE_ALL>();
    combineOuterOffset.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineOuterCntIndexOffset));

    DataCopyExtParams outerOffsetWriteCountsParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0,
                                                   0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterOffset, outerOffsetLt, outerOffsetWriteCountsParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::ReorderTokens()
{
    uint32_t sendTokenNum = axisBS_ / aivNum_;
    uint32_t remainderTokenNum = axisBS_ % aivNum_;
    uint32_t startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;

    if (sendTokenNum == 0) {
        return;
    }

    LocalTensor<int32_t> expertIdsI32Tensor = tBuf.Get<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK));

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor, expertIdsGMTensor_, expCopyParams, expPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(expertIdsI16Tensor_, expertIdsI32Tensor, RoundMode::CAST_NONE, axisBS_ * axisK_);
    SyncFunc<AscendC::HardEvent::V_MTE2>();

    uint32_t singleTokenUBSize = tokenStructLen_;
    uint32_t quantTokenUBSize = 0;
    if constexpr (DynamicQuant || IsMxQuant) {
        quantTokenUBSize = tokenStructLen_ > axisH_ * sizeof(XType) ? tokenStructLen_ : axisH_ * sizeof(XType);
        singleTokenUBSize = quantTokenUBSize + axisH_ * sizeof(float);
    }
    uint32_t maxTokenNumInUB = tokenUbSize_ / singleTokenUBSize;
    uint32_t batchNum = (sendTokenNum + maxTokenNumInUB - 1) / maxTokenNumInUB;

    LocalTensor<uint8_t> tokenTensorU8_ =
        tBuf.GetWithOffset<uint8_t>(maxTokenNumInUB * tokenStructLen_, TBUF_TEMP_OFFSET);
    LocalTensor<uint64_t> tokenTempTensorU64_ = tokenTensorU8_.template ReinterpretCast<uint64_t>();
    LocalTensor<XType> tokenLt = tokenTensorU8_.template ReinterpretCast<XType>();
    LocalTensor<float> tokenCastLt;
    GlobalTensor<uint8_t> expertIdsGMTensorU8_;
    GlobalTensor<uint8_t> weightGt;
    GlobalTensor<uint8_t> xGMtU8;
    xGMtU8.SetGlobalBuffer((__gm__ uint8_t *)expandXGM_);
    weightGt.SetGlobalBuffer((__gm__ uint8_t *)weightsGM_);
    expertIdsGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandIdxGM_);

    if constexpr (DynamicQuant || IsMxQuant) {
        uint32_t tokenCastLtOffset = RoundUp(TBUF_TEMP_OFFSET + quantTokenUBSize * maxTokenNumInUB, UB_32B_ALIGN);
        tokenCastLt = tBuf.GetWithOffset<float>(axisH_ * maxTokenNumInUB, tokenCastLtOffset);
    }

    for (uint32_t batchIndex = 0; batchIndex < batchNum; batchIndex++) {
        uint32_t currentTokenNum = sendTokenNum > maxTokenNumInUB ? maxTokenNumInUB : sendTokenNum;
        if constexpr (DynamicQuant || IsMxQuant) {
            DataCopy(tokenTensorU8_, xGMtU8[startTokenId * axisH_ * sizeof(XType)],
                     currentTokenNum * axisH_ * sizeof(XType));
            PipeBarrier<PIPE_ALL>();
            QuantProcess(currentTokenNum, tokenLt, tokenCastLt);
        } else {
            DataCopyExtParams tokenCopyParams{static_cast<uint16_t>(currentTokenNum),
                                              static_cast<uint32_t>(tokenLenInStruct_), 0,
                                              static_cast<uint32_t>(tokenGapInStruct_), 0};
            DataCopyPadExtParams<uint8_t> tokenPadParams;
            DataCopyPad(tokenTensorU8_[tokenOffsetInStruct_], xGMtU8[startTokenId * tokenLenInStruct_], tokenCopyParams,
                        tokenPadParams);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams expCopyParams{static_cast<uint16_t>(currentTokenNum), static_cast<uint32_t>(realLenInStruct_),
                                        0, static_cast<uint32_t>(infoGapInStruct_), 0};
        DataCopyPadExtParams<uint8_t> expPadParams;
        DataCopyPad(tokenTensorU8_[expOffsetInStruct_], expertIdsGMTensorU8_[startTokenId * realLenInStruct_],
                    expCopyParams, expPadParams);
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams weightCopyParams{static_cast<uint16_t>(currentTokenNum),
                                           static_cast<uint32_t>(realLenInStruct_), 0,
                                           static_cast<uint32_t>(infoGapInStruct_), 0};
        DataCopyPadExtParams<uint8_t> weightPadParams;
        DataCopyPad(tokenTensorU8_[weightOffsetInStruct_], weightGt[startTokenId * realLenInStruct_], weightCopyParams,
                    weightPadParams);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t tokenIndex = 0; tokenIndex < currentTokenNum; ++tokenIndex) {
            uint32_t tokenId = startTokenId + tokenIndex;
            uint32_t startExpId = tokenId * axisK_;
            uint32_t flagOffset = (tokenIndex * tokenStructLen_ + flagOffsetInStruct_) / sizeof(uint64_t);
            tokenTempTensorU64_(flagOffset) = SHOULD_SEND_FLAG_VALUE;
            uint64_t sendServerInfo = 0;
            for (uint32_t i = 0; i < axisK_; i++) {
                uint32_t expertId = static_cast<uint32_t>(expertIdsI16Tensor_(startExpId + i));
                uint32_t dstServerId = expertId / moeExpertNumInServer_;
                sendServerInfo |= (1UL << dstServerId);
            }
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<uint64_t> sendServerInfoTemp =
                tokenAddrFlagStructGlobalU64Tensor_[(FLAG_SIZE * tokenId) / sizeof(uint64_t)];
            sendServerInfoTemp.SetValue(0, sendServerInfo);
            DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                sendServerInfoTemp);
            PipeBarrier<PIPE_ALL>();
        }
        uint32_t tokenWinOutOffset = startTokenId * tokenStructLen_;
        DataCopy(sendTokensU8Tensor_[tokenWinOutOffset], tokenTensorU8_, currentTokenNum * tokenStructLen_);
        PipeBarrier<PIPE_ALL>();
        startTokenId += currentTokenNum;
        sendTokenNum -= currentTokenNum;
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::QuantProcess(
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

    if constexpr (!IsMxQuant) {
        Cast(tokenCastLt, xTokenLt, RoundMode::CAST_NONE, sendTokenNum * axisH_);
    }
    for (int32_t i = 0; i < sendTokenNum; ++i) {
#ifdef __DAV_C310__
        if constexpr (IsMxQuant) {
            uint32_t mxScaleNum = Align2(Ceil32(axisH_));
            LocalTensor<uint8_t> tokenUnitLt = xTokenLt.template ReinterpretCast<uint8_t>();
            __ubuf__ XType *srcAddr = (__ubuf__ XType *)tokenUnitLt[i * axisH_ * sizeof(XType)].GetPhyAddr();
            LocalTensor<uint16_t> tokenCastU16 = tokenCastLt.template ReinterpretCast<uint16_t>();
            __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)tokenCastU16.GetPhyAddr();
            __ubuf__ uint16_t *halfScaleLocalAddr =
                (__ubuf__ uint16_t *)tokenCastU16[Align32(mxScaleNum)].GetPhyAddr();
            __ubuf__ int8_t *outLocalAddr =
                (__ubuf__ int8_t *)tokenUnitLt[i * tokenStructLen_ + tokenOffsetInStruct_].GetPhyAddr();
            __ubuf__ uint16_t *mxScaleLocalAddr =
                (__ubuf__ uint16_t *)tokenUnitLt[i * tokenStructLen_ + scaleOffsetInStruct_].GetPhyAddr();

            quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_);
            quant::ComputeScale<ExpandXOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
            if constexpr (Std::IsSame<ExpandXOutType, fp8_e4m3fn_t>::value ||
                          Std::IsSame<ExpandXOutType, fp8_e5m2_t>::value) {
                quant::ComputeFp8Data<XType, ExpandXOutType, AscendC::RoundMode::CAST_TRUNC,
                                      AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
            } else {
                quant::ComputeFp4Data<XType, ExpandXOutType, AscendC::RoundMode::CAST_TRUNC,
                                      AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_);
            }
        } else
#endif
        {
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
            xOutTensor = tokenUnitLt[i * tokenStructLen_ + tokenOffsetInStruct_].template ReinterpretCast<ExpandXOutType>();
            Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

            LocalTensor<float> scaleTensor =
                tokenUnitLt[i * tokenStructLen_ + scaleOffsetInStruct_].template ReinterpretCast<float>();
            scaleTensor.SetValue(0, float(1.0) / dynamicScale);
        }
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void
MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::SendDataToServer(uint32_t destServerId)
{
    uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + destServerId * SERVER_RANK_SIZE;
    uint64_t destServerMask = (1UL << destServerId);

    uint64_t dstRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + (halfWinSize_ * bufferId_ * 1UL) +
                                      (serverId_ * SERVER_SIZE_ON_WIN * 1UL));
    uint64_t srcRdmaAddrBase = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + (halfWinSize_ * bufferId_ * 1UL));
    LocalTensor<uint64_t> sendTokenInfoLocalTensor =
        tBuf.GetWithOffset<uint64_t>((axisBS_ * FLAG_SIZE) / sizeof(uint64_t), 0);
    DataCopy(sendTokenInfoLocalTensor, tokenAddrFlagStructGlobalU64Tensor_, (axisBS_ * FLAG_SIZE) / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();

    for (uint32_t tokenIdx = 0; tokenIdx < axisBS_; ++tokenIdx) {
        uint64_t destServerInfo = sendTokenInfoLocalTensor(tokenIdx * FLAG_SIZE / sizeof(uint64_t));
        if ((destServerInfo & destServerMask) != 0) {
            uint64_t srcRdmaAddr = (uint64_t)(srcRdmaAddrBase + (tokenStructLen_ * tokenIdx * 1UL));
            AIVRDMAPostSend((GM_ADDR)srcRdmaAddr, (GM_ADDR)dstRdmaAddr, dstRankId, tokenStructLen_, qp_info_);
            dstRdmaAddr += tokenStructLen_;
            PipeBarrier<PIPE_ALL>();
        }
    }

    uint64_t srcFlagRdmaAddr = (uint64_t)(sendStatusTensor_.GetPhyAddr());
    uint64_t dstFlagRdmaAddr =
        (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + WIN_SIZE + serverId_ * STATE_OFFSET);
    AIVRDMAPostSend((GM_ADDR)srcFlagRdmaAddr, (GM_ADDR)dstFlagRdmaAddr, dstRankId, FLAG_SIZE, qp_info_);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::GetExpRank(uint32_t expertId)
{
    return expertId / localMoeExpertNum_;
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::SetIpcFlag(int32_t flagVal)
{
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> globalSet;
    globalSet.SetGlobalBuffer((__gm__ uint64_t *)(shareAddrs[destRankIdx] + IPC_FLAG_OFFSET) +
                              localRankId * B64_PER_BLOCK);
    LocalTensor<uint64_t> localSet = tBuf.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    uint64_t setVal = magicVal_;
    localSet.SetValue(0, setVal);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(globalSet, localSet, B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::WaitIpcFlag(int32_t flagVal)
{
    uint64_t waitVal = magicVal_;
    if (aivId_ >= SERVER_RANK_SIZE) {
        return;
    }
    LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(B64_PER_BLOCK, 0);
    bool isSync = true;
    uint32_t destRankIdx = aivId_;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<uint64_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ uint64_t *)(shareAddrs[localRankId] + IPC_FLAG_OFFSET) +
                              destRankIdx * B64_PER_BLOCK);
    PipeBarrier<PIPE_ALL>();
    do {
        DataCopy(localWait, flagIpcGt, B64_PER_BLOCK);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint64_t tempVal = localWait.GetValue(0);
        if (tempVal >= waitVal) {
            break;
        }
    } while (isSync);
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::GetArrivedTokenInfo(
    uint32_t serverIdx, uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> localUB_U8)
{
    GlobalTensor<uint64_t> TokenFlagGtU64;
    GlobalTensor<uint8_t> TokensGtU8;

    TokenFlagGtU64.SetGlobalBuffer((__gm__ uint64_t *)(windowInGM_));
    TokensGtU8.SetGlobalBuffer((__gm__ uint8_t *)(windowInGM_));

    LocalTensor<uint64_t> statusTensor = statusBuf_.Get<uint64_t>();
    DataCopy(statusTensor, readStatusTensor_[(serverIdx)*STATE_OFFSET / sizeof(uint64_t)],
             FLAG_SIZE / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    uint64_t endFlagValue = statusTensor.GetValue(0);

    uint32_t TokenOffset = serverIdx * SERVER_SIZE_ON_WIN + tokenIdx * tokenStructLen_;
    DataCopy(statusTensor, TokenFlagGtU64[(TokenOffset + flagOffsetInStruct_) / sizeof(uint64_t)],
             FLAG_SIZE / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    uint64_t tokenFlagValue = statusTensor.GetValue(0);

    uint32_t nextTokenOffset = serverIdx * SERVER_SIZE_ON_WIN + (tokenIdx + 1) * tokenStructLen_;
    DataCopy(statusTensor, TokenFlagGtU64[(nextTokenOffset + flagOffsetInStruct_) / sizeof(uint64_t)],
             FLAG_SIZE / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();
    uint64_t nextTokenFlagValue = statusTensor.GetValue(0);

    if (nextTokenFlagValue == SHOULD_SEND_FLAG_VALUE) {
        if (justExpInfo) {
            DataCopy(localUB_U8, TokensGtU8[TokenOffset + expOffsetInStruct_], expLenInStruct_);
        } else {
            DataCopy(localUB_U8, TokensGtU8[TokenOffset], tokenStructLen_);
        }
        PipeBarrier<PIPE_ALL>();
        return ARRIVAL_STATUS;
    }

    if (endFlagValue != END_OF_WRITE_FLAG_VALUE) {
        PipeBarrier<PIPE_ALL>();
        return WAIT_STATUS;
    } else {
        if (tokenFlagValue == SHOULD_SEND_FLAG_VALUE) {
            if (justExpInfo) {
                DataCopy(localUB_U8, TokensGtU8[TokenOffset + expOffsetInStruct_], expLenInStruct_);
            } else {
                DataCopy(localUB_U8, TokensGtU8[TokenOffset], tokenStructLen_);
            }
            PipeBarrier<PIPE_ALL>();
            return ARRIVAL_STATUS;
        } else {
            PipeBarrier<PIPE_ALL>();
            return FINISH_STATUS;
        }
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::GetSelfServerTokenInfo(
    uint32_t tokenIdx, bool justExpInfo, LocalTensor<uint8_t> localUB_U8)
{
    if (tokenIdx >= axisBS_) {
        return FINISH_STATUS;
    }

    LocalTensor<uint64_t> sendTokenInfoLocalTensor = statusBuf_.Get<uint64_t>();
    DataCopy(sendTokenInfoLocalTensor, tokenAddrFlagStructGlobalU64Tensor_[tokenIdx * FLAG_SIZE / sizeof(uint64_t)],
             FLAG_SIZE / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();

    uint64_t sendFlag = sendTokenInfoLocalTensor(0);

    uint64_t destServerMask = (1UL << serverId_);
    if ((sendFlag & destServerMask) == 0) {
        return SKIP_STATUS;
    } else {
        GlobalTensor<uint8_t> TokensGtU8;
        TokensGtU8.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
        if (justExpInfo) {
            DataCopy(localUB_U8, TokensGtU8[tokenIdx * tokenStructLen_ + expOffsetInStruct_], expLenInStruct_);
        } else {
            DataCopy(localUB_U8, TokensGtU8[tokenIdx * tokenStructLen_], tokenStructLen_);
        }
        PipeBarrier<PIPE_ALL>();
        return ARRIVAL_STATUS;
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::Win2Ipc()
{
    uint32_t coresPerServer = (aivNum_ - serverNum - 1) / serverNum;
    uint32_t logicAivId = aivId_ - serverNum - 1;
    if (logicAivId >= coresPerServer * serverNum) {
        return;
    }
    uint32_t formServerId = logicAivId / coresPerServer;

    uint32_t expStartId = serverId_ * SERVER_RANK_SIZE * localMoeExpertNum_;
    uint32_t expEndId = expStartId + SERVER_RANK_SIZE * localMoeExpertNum_;
    uint32_t tokenIdx = 0;
    uint32_t selfTokenIdx = 0;

    uint32_t tokenStatus = WAIT_STATUS;
    bool justExpInfo = (tokenIdx % coresPerServer != logicAivId % coresPerServer);
    uint32_t tokenNumPerExpInfoSize =
        SERVER_RANK_SIZE * localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT * sizeof(int32_t);

    GlobalTensor<uint8_t> targetTokenIpcGt;
    GlobalTensor<int32_t> targetCntIpcGt;

    LocalTensor<int32_t> tokenNumPerExp =
        tBuf.GetWithOffset<int32_t>(SERVER_RANK_SIZE * localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT, TBUF_TEMP_OFFSET);
    LocalTensor<uint8_t> localUB_U8 = tBuf.GetWithOffset<uint8_t>(
        tokenStructLen_ / sizeof(uint8_t), RoundUp(tokenNumPerExpInfoSize + TBUF_TEMP_OFFSET, IPC_BUFF_ALIGN));
    LocalTensor<int32_t> localUB_32 = tBuf.GetWithOffset<int32_t>(
        tokenStructLen_ / sizeof(int32_t), RoundUp(tokenNumPerExpInfoSize + TBUF_TEMP_OFFSET, IPC_BUFF_ALIGN));

    Duplicate<int32_t>(tokenNumPerExp, 0, SERVER_RANK_SIZE * localMoeExpertNum_ * EXP_TOKEN_COUNT_FLAG_CNT);
    PipeBarrier<PIPE_ALL>();
    while (tokenStatus != FINISH_STATUS) {
        if (formServerId == serverId_) {
            tokenStatus = GetSelfServerTokenInfo(selfTokenIdx, justExpInfo, localUB_U8);
            if (tokenStatus == SKIP_STATUS || tokenStatus == ARRIVAL_STATUS) {
                selfTokenIdx++;
            }
        } else {
            tokenStatus = GetArrivedTokenInfo(formServerId, tokenIdx, justExpInfo, localUB_U8);
        }

        if (tokenStatus != ARRIVAL_STATUS) {
            continue;
        }
        LocalTensor<int32_t> expInfoTensor;
        if (justExpInfo) {
            expInfoTensor = localUB_32;
        } else {
            expInfoTensor = localUB_32[expOffsetInStruct_ / sizeof(int32_t)];
        }

        for (int32_t expIndex = 0; expIndex < axisK_; ++expIndex) {
            uint32_t targetExpId = (uint32_t)(expInfoTensor(expIndex));
            if (targetExpId < expStartId || targetExpId >= expEndId) {
                continue;
            }

            uint32_t targetRankId = GetExpRank(targetExpId);
            uint32_t localExpIdx = targetExpId % (localMoeExpertNum_ * SERVER_RANK_SIZE);
            uint32_t targetTokenIdx = (uint32_t)(tokenNumPerExp(localExpIdx * EXP_TOKEN_COUNT_FLAG_CNT));
            tokenNumPerExp(localExpIdx * EXP_TOKEN_COUNT_FLAG_CNT) += 1;
            if (justExpInfo) {
                continue;
            }

            uint32_t targetExpOffset =
                (targetExpId % localMoeExpertNum_) * worldSize_ * RANK_SIZE_ON_IPC;
            uint32_t targetServerOffset = formServerId * SERVER_RANK_SIZE * RANK_SIZE_ON_IPC;
            uint32_t targetRankOffset = (rankId_ % SERVER_RANK_SIZE) * RANK_SIZE_ON_IPC;
            uint32_t targetTokenOffset = tokenStructLen_ * targetTokenIdx;
            uint32_t targetOffset =
                targetExpOffset + targetServerOffset + targetRankOffset + targetTokenOffset;
            targetTokenIpcGt.SetGlobalBuffer(
                (__gm__ uint8_t *)(shareAddrs[targetRankId % SERVER_RANK_SIZE] + IPC_DATA_OFFSET + targetOffset));
            PipeBarrier<PIPE_ALL>();
            DataCopy(targetTokenIpcGt, localUB_U8, tokenStructLen_);
            PipeBarrier<PIPE_ALL>();
        }
        tokenIdx += 1;
        justExpInfo = (tokenIdx % coresPerServer != logicAivId % coresPerServer);
    }
    uint32_t batchNum = (SERVER_RANK_SIZE * localMoeExpertNum_ + coresPerServer - 1) / coresPerServer;
    for (uint32_t batch = 0; batch < batchNum; batch++) {
        uint32_t targetExpId = expStartId + batch * coresPerServer + logicAivId % coresPerServer;
        uint32_t targetRankId = GetExpRank(targetExpId);
        if (targetExpId >= expEndId) {
            return;
        }
        uint32_t localExpOffset = targetExpId % (localMoeExpertNum_ * SERVER_RANK_SIZE) * EXP_TOKEN_COUNT_FLAG_CNT;
        uint32_t targetCntOffset = ((targetExpId % localMoeExpertNum_) * worldSize_ + formServerId * SERVER_RANK_SIZE +
                                    (rankId_ % SERVER_RANK_SIZE)) *
                                   EXP_TOKEN_COUNT_FLAG_CNT;
        targetCntIpcGt.SetGlobalBuffer(
            (__gm__ int32_t *)(shareAddrs[targetRankId % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));
        PipeBarrier<PIPE_ALL>();
        DataCopy(targetCntIpcGt[targetCntOffset], tokenNumPerExp[localExpOffset], EXP_TOKEN_COUNT_FLAG_CNT);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::Ipc2Out()
{
    uint32_t coresPerExp = aivNum_ / localMoeExpertNum_;
    if (aivId_ >= coresPerExp * localMoeExpertNum_) {
        return;
    }
    uint32_t coresPerServer = aivNum_ / serverNum;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int32_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE]));
    uint32_t curExpIdx = aivId_ / coresPerExp;
    uint32_t localAivId = aivId_ % coresPerExp;
    uint32_t srCntPerExp = serverNum * SERVER_RANK_SIZE;
    uint32_t srCntPerCore = srCntPerExp / coresPerExp;
    uint32_t srCntRemain = srCntPerExp % coresPerExp;
    uint32_t srCntPreRemain = (localAivId < srCntRemain) ? localAivId : srCntRemain;
    uint32_t srCntCurCore = (localAivId < srCntRemain) ? (srCntPerCore + 1) : srCntPerCore;

    GlobalTensor<int32_t> tokenCntIpcGt;
    tokenCntIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));

    LocalTensor<int64_t> tokenCntByExpUB = tBuf.GetWithOffset<int64_t>(2 * 1024 / sizeof(int64_t), 4 * 1024);
    LocalTensor<int32_t> tokenCntUB = tBuf.GetWithOffset<int32_t>(2 * 1024 / sizeof(int32_t), 6 * 1024);
    LocalTensor<float> weightLt = tBuf.GetWithOffset<float>(2 * 1024 / sizeof(float), 2 * 1024);

    DataCopyExtParams copyExpertIdsParams{1,
                                          static_cast<uint32_t>(serverNum * SERVER_RANK_SIZE * localMoeExpertNum_ *
                                                                EXP_TOKEN_COUNT_FLAG_CNT * sizeof(int32_t)),
                                          0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams;
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(tokenCntUB, tokenCntIpcGt, copyExpertIdsParams, padParams);

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    int32_t cntSum = 0;
    const int tempSize = serverNum * SERVER_RANK_SIZE * localMoeExpertNum_;
    int log2WorldSize = ScalarGetSFFValue<1>(worldSize_);
#pragma unroll 8
    for (uint32_t i = 0; i < tempSize; ++i) {
        cntSum += tokenCntUB(i << 3);
        tokenCntUB(i) = cntSum;
    }

    for (uint32_t i = 0; i < localMoeExpertNum_; ++i) {
        if (expertTokenNumsType_ == 1) {
            int32_t preValue = (i == 0) ? 0 : tokenCntUB(i * worldSize_ - 1);
            tokenCntByExpUB(i) = static_cast<int64_t>(tokenCntUB(i * worldSize_ + worldSize_ - 1) - preValue);
        } else {
            tokenCntByExpUB(i) = static_cast<int64_t>(tokenCntUB(i * worldSize_ + worldSize_ - 1));
        }
    }

    uint32_t srPreCnt = curExpIdx * srCntPerExp + localAivId * srCntPerCore + srCntPreRemain;
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint8_t> srcIpcGt;
    srcIpcGt.SetGlobalBuffer((__gm__ uint8_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_DATA_OFFSET));

    LocalTensor<uint8_t> localUB = tBuf.GetWithOffset<uint8_t>(tokenUbSize_ / sizeof(uint8_t), TBUF_TEMP_OFFSET);
    LocalTensor<float> localUBfloat = tBuf.GetWithOffset<float>(tokenUbSize_ / sizeof(float), TBUF_TEMP_OFFSET);
    LocalTensor<int32_t> localUBint32 = tBuf.GetWithOffset<int32_t>(tokenUbSize_ / sizeof(int32_t), TBUF_TEMP_OFFSET);

    int32_t sumTokenCnt = (0 == srPreCnt) ? 0 : tokenCntUB(srPreCnt - 1);
    for (uint32_t idx = 0; idx < srCntCurCore; ++idx) {
        uint32_t srIdx = srPreCnt + idx;
        int32_t curSrTokenCnt = tokenCntUB(srIdx) - (srIdx == 0 ? 0 : tokenCntUB(srIdx - 1));
        if (curSrTokenCnt == 0) {
            continue;
        }
        uint32_t tokenCntInUB = tokenUbSize_ / tokenStructLen_;
        uint32_t batchCnt = (curSrTokenCnt + tokenCntInUB - 1) / tokenCntInUB;
        for (uint32_t batchIdx = 0; batchIdx < batchCnt; ++batchIdx) {
            uint32_t tokenCntInBatch = tokenCntInUB;
            if (batchIdx == batchCnt - 1) {
                tokenCntInBatch = curSrTokenCnt - (batchCnt - 1) * tokenCntInUB;
            }
            DataCopyExtParams copyTokenParams{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(tokenCntInBatch * tokenStructLen_), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams;
            uint32_t srcIpcOffset = srIdx * RANK_SIZE_ON_IPC + batchIdx * tokenCntInUB * tokenStructLen_;
            DataCopyPad(localUB, srcIpcGt[srcIpcOffset], copyTokenParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyExtParams writeTokenParams{static_cast<uint16_t>(tokenCntInBatch),
                                               static_cast<uint32_t>(sizeof(ExpandXOutType) * axisH_),
                                               static_cast<uint32_t>(tokenGapInStruct_), 0, 0};
            LocalTensor<ExpandXOutType> outUB = localUB.ReinterpretCast<ExpandXOutType>();
            DataCopyPad(expandXOutGMTensor_[(sumTokenCnt + batchIdx * tokenCntInUB) * axisH_],
                        outUB[tokenOffsetInStruct_ / sizeof(ExpandXOutType)], writeTokenParams);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t tokenIdx = 0; tokenIdx < tokenCntInBatch; tokenIdx++) {
                for (uint32_t expIdx = 0; expIdx < axisK_; expIdx++) {
                    uint32_t expOffset = (tokenIdx * tokenStructLen_ + expOffsetInStruct_) / sizeof(int32_t) + expIdx;
                    if (curExpIdx + rankId_ * localMoeExpertNum_ == localUBint32(expOffset)) {
                        uint32_t weightOffset = expOffset + alignK_;
                        weightLt(tokenIdx) = localUBfloat(weightOffset);
                        break;
                    }
                }
                LocalTensor<float> pintfLt =
                    localUBfloat[(tokenIdx * tokenStructLen_ + weightOffsetInStruct_) / sizeof(float)];
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams weightTokenParams{static_cast<uint16_t>(1),
                                                static_cast<uint32_t>(tokenCntInBatch * sizeof(float)), 0, 0, 0};
            DataCopyPad(weightsOutGt[(sumTokenCnt + batchIdx * tokenCntInUB)], weightLt, weightTokenParams);
            PipeBarrier<PIPE_ALL>();
            if constexpr (DynamicQuant || IsMxQuant) {
                if constexpr (IsMxQuant) {
                    DataCopyExtParams quantTokenParams{
                        static_cast<uint16_t>(tokenCntInBatch), scaleLenInStruct_,
                        static_cast<uint32_t>((tokenStructLen_ - scaleLenInStruct_) / UB_32B_ALIGN), 0, 0};
                    LocalTensor<uint8_t> quantTempUB = localUB[scaleOffsetInStruct_].template ReinterpretCast<uint8_t>();
                    auto dynamicScalesOutU8GT = dynamicScalesOutGMTensor_.template ReinterpretCast<uint8_t>();
                    DataCopyPad(dynamicScalesOutU8GT[(sumTokenCnt + batchIdx * tokenCntInUB) * scaleLenInStruct_],
                                quantTempUB, quantTokenParams);
                } else {
                    DataCopyExtParams quantTokenParams{
                        static_cast<uint16_t>(tokenCntInBatch), static_cast<uint32_t>(sizeof(float)),
                        static_cast<uint32_t>((tokenStructLen_ - UB_32B_ALIGN) / UB_32B_ALIGN), 0, 0};
                    LocalTensor<float> quantTempUB = localUB[scaleOffsetInStruct_].template ReinterpretCast<float>();
                    DataCopyPad(dynamicScalesOutGMTensor_[(sumTokenCnt + batchIdx * tokenCntInUB)], quantTempUB,
                                quantTokenParams);
                }
            }
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
        sumTokenCnt += curSrTokenCnt;
    }
    if (aivId_ == 0) {
        GlobalTensor<int32_t> tokenNumsGlobal;
        tokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_));
        DataCopyExtParams countsParams{
            1, static_cast<uint32_t>(localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(tokenNumsGlobal, tokenCntUB, countsParams);

        GlobalTensor<int64_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int64_t *)(expertTokenNumsOutGM_));
        DataCopyExtParams writeCountsParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsGlobal, tokenCntByExpUB, writeCountsParams);
    }
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::CleanUp()
{
    if (aivId_ == 0) {
        bufferChosenGlobal_(0) = bufferId_ ^ 1;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            bufferChosenGlobal_);
    }

    uint32_t tokenEndFlagCleanSize = MAX_BS_NUM * FLAG_SIZE;
    uint32_t writeEndFlagCleanSize = serverNum * STATE_OFFSET;
    uint32_t maxCleanSize =
        tokenEndFlagCleanSize > writeEndFlagCleanSize ? tokenEndFlagCleanSize : writeEndFlagCleanSize;
    LocalTensor<int32_t> cleanTempLt_ = tBuf.GetWithOffset<int32_t>(maxCleanSize / sizeof(int32_t), TBUF_TEMP_OFFSET);
    Duplicate<int32_t>(cleanTempLt_, 0, maxCleanSize / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>();
    if (aivId_ == serverNum - 1) {
        GlobalTensor<int32_t> readStatusTensorU32;
        readStatusTensorU32.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + WIN_SIZE));
        DataCopy(readStatusTensorU32, cleanTempLt_, writeEndFlagCleanSize / sizeof(uint32_t));
    }

    GlobalTensor<int32_t> tokenEndFlagCleanTensor;
    tokenEndFlagCleanTensor.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + aivId_ * SERVER_SIZE_ON_WIN));
    DataCopyExtParams cleanTokenEndFlagParams{uint16_t(MAX_BS_NUM), uint32_t(flagLenInStruct_), 0,
                                              uint32_t(tokenStructLen_ - flagLenInStruct_), 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(tokenEndFlagCleanTensor[flagOffsetInStruct_ / sizeof(int32_t)], cleanTempLt_, cleanTokenEndFlagParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA5LayeredClass>
__aicore__ inline void MoeDistributeDispatchA5Layered<TemplateMC2TypeA5LayeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        ReorderTokens();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if (aivId_ < serverNum) {
            if (aivId_ != serverId_) {
                SendDataToServer(aivId_);
            }
            CreateInnerReduceInfo(aivId_);
        } else if (aivId_ == serverNum) {
            CreateOuterReduceInfo();
        } else {
            Win2Ipc();
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetIpcFlag(IPC_FLAG_STEP_1);
        WaitIpcFlag(IPC_FLAG_STEP_1);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        Ipc2Out();
        if (aivId_ < serverNum) {
            PipeBarrier<PIPE_ALL>();
            CleanUp();
        }

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA5LayeredImpl
#endif  // MOE_DISTRIBUTE_DISPATCH_A5_LAYERED_H
