/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_a2_layered.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
#define MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_base.h"

namespace MoeDistributeDispatchA2Impl {
#define TemplateMC2TypeA2layeredClass \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist
#define TemplateMC2TypeA2layeredFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist

using namespace AscendC;
template <TemplateMC2TypeA2layeredClass>
class MoeDistributeDispatchV2Layered
{
public:
    constexpr static uint32_t STATE_OFFSET = 512;                 // 状态空间偏移地址
    constexpr static uint32_t STATUS_SIZE_LAYERED = 1024 * 1024;  // 1M
    constexpr static uint32_t RDMA_BUFFER_ALIGN = 4 * 1024;
    constexpr static uint32_t SERVER_RANK_SIZE = 8;
    constexpr static uint32_t B64_PER_BLOCK = 4;
    constexpr static uint32_t B16_PER_BLOCK = 16;
    constexpr static uint32_t UB_32B_ALIGN = 32;
    constexpr static uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
    constexpr static uint32_t TBUF_SIZE = 190 * 1024;
    constexpr static uint32_t IPC_DATA_OFFSET = 4 * 1024 * 1024;
    constexpr static uint32_t RDMA_DATA_SIZE = 800U * 1024U * 1024U;
    constexpr static uint32_t IPC_FLAG_OFFSET = 1 * 1024 * 1024;
    constexpr static uint32_t IPC_TOKEN_CNT_OFFSET = 2 * 1024 * 1024;
    constexpr static uint32_t IPC_MAGIC_OFFSET = IPC_TOKEN_CNT_OFFSET - 128 * 32;
    constexpr static uint32_t MTU_SIZE = 4 * 1024;
    constexpr static uint32_t IPC_BUFF_ALIGN = 512;
    constexpr static int32_t IPC_FLAG_STEP_1 = 0x0d0d0d0d;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 8 * 1024;
    constexpr static uint32_t MAX_BS_NUM = 512U;  // 适配bs=512
    constexpr static uint32_t TBUF_OFFSET_ALIGN_B32_CNT = 2 * 1024 / sizeof(int32_t);
    constexpr static uint64_t SHOULD_SEND_FLAG_VALUE = 0x0f0f0f0f;
    constexpr static uint64_t END_OF_WRITE_FLAG_VALUE = 0xffffffff;
    constexpr static uint32_t FLAG_SIZE = 64;
    constexpr static uint32_t FINISH_STATUS = 0;
    constexpr static uint32_t WAIT_STATUS = 1;
    constexpr static uint32_t ARRIVAL_STATUS = 2;
    constexpr static uint32_t SKIP_STATUS = 3;
    constexpr static uint32_t EXTRA_TOKEN_INFO_NUM = 4U;  // 专家信息 权重信息 量化Scale 到达标志位

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
    __aicore__ inline MoeDistributeDispatchV2Layered(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM,
                                GM_ADDR contextGM0);
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
    // GlobalTensor<float> weightsOutGt;
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

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};   // 真实的K值
    uint32_t alignK_{0};  // axisK_与 BITS32_PER_BLOCK 对齐
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t serverId_{0};
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
    uint32_t tokenUbSize_{0};

    // TokenStruck
    uint32_t tokenGapInStruct_{0};
    uint32_t infoGapInStruct_{0};
    uint32_t tokenStructLen_{0};
    uint32_t tokenLenInStruct_{0};
    uint32_t expLenInStruct_{0};
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
    uint64_t combineOuterCntPerServerOffset;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclA2CombineOpParam *winContext_{nullptr};
};

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut, GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM,
    GM_ADDR contextGM0)
{
    tpipe_ = pipe;
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    auto tiling = (__gm__ MoeDistributeDispatchV2TilingData *)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling->mc2CcTiling));
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);

    hccl_.Init(contextGM0, mc2InitTiling);
    hccl_.SetCcTiling(mc2CcTiling);

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
    totalWinSize_ = RDMA_DATA_SIZE;  // 800 MB for RDMA
    shareMemOffset_ = totalWinSize_;
    halfWinSize_ = totalWinSize_ / 2;
    WIN_SIZE = halfWinSize_ - STATUS_SIZE_LAYERED;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchV2Info.expertTokenNumsType;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    serverNum = worldSize_ / SERVER_RANK_SIZE;

    uint64_t winSizeMin =
        moeExpertNum_ * axisBS_ * (axisH_ * sizeof(XType) + EXTRA_TOKEN_INFO_NUM * alignK_ * sizeof(uint32_t)) +
        IPC_DATA_OFFSET + RDMA_DATA_SIZE;  // 考虑负载极其不均衡时，HCCL BUFFSIZE需要开的大小
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

    // RDMA buffer init
    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + WIN_SIZE + worldSize_ * STATE_OFFSET));
    bufferId_ = bufferChosenGlobal_(0);
    windowInGM_ = windowInGM_ + halfWinSize_ * bufferId_;
    windowOutGM_ = windowOutGM_ + halfWinSize_ * bufferId_;
    RANK_SIZE_ON_IPC = (totalSize_ - totalWinSize_ - IPC_DATA_OFFSET) / (localMoeExpertNum_ * worldSize_);
    RANK_SIZE_ON_IPC = (RANK_SIZE_ON_IPC / IPC_BUFF_ALIGN) * IPC_BUFF_ALIGN;

    // IPC buffer init
    for (int i = 0; i < SERVER_RANK_SIZE; i++) {
        shareAddrs[i] = (__gm__ uint8_t *)(reinterpret_cast<uint64_t>(
            hccl_.GetWindowsInAddr(rankId_ / SERVER_RANK_SIZE * SERVER_RANK_SIZE + i) + shareMemOffset_));
    }
    SERVER_SIZE_ON_WIN = WIN_SIZE / serverNum;
    SERVER_SIZE_ON_WIN = (SERVER_SIZE_ON_WIN / RDMA_BUFFER_ALIGN) * RDMA_BUFFER_ALIGN;

    // TokenStruct info init
    tokenLenInStruct_ = axisH_ * sizeof(ExpandXOutType);
    expLenInStruct_ = alignK_ * sizeof(uint32_t);  // 为了对齐，使用 alignK_ 计算tokenStruct中的内存
    realLenInStruct_ = axisK_ * sizeof(uint32_t);  // 内存中实际有效部分，跟 axisK_ 有关
    scaleLenInStruct_ = UB_32B_ALIGN;
    flagLenInStruct_ = UB_32B_ALIGN;
    tokenStructLen_ = flagLenInStruct_ + tokenLenInStruct_ + expLenInStruct_ + scaleLenInStruct_;

    /* 注意：flag必须放置在整个token struct的最前端，而且token和token之间不能连续发送。
       原因：两条ROCE消息通过PCIE总线写到GM内存时，只有第二条消息的第一个分片的写操作和上一条消息保证是保序的，其余分片可能比第一条消息更早写入。
            后续需要通过下一个token的flag到达来校验第一个token是否收到。
       满足条件：寄存器默认配置保证消息第一个分片写操作保序 */

    /* struct结构如下：
    //    | flag: 32B | token(data): H * dtype | exp: alignK * uint32  | weight: alignK * uint32 | scale: 32B |
       | flag: 32B | token(data): H * dtype | exp: alignK * uint32 | scale: 32B |
    */
    flagOffsetInStruct_ = 0;
    tokenOffsetInStruct_ = flagLenInStruct_;
    expOffsetInStruct_ = tokenOffsetInStruct_ + tokenLenInStruct_;
    scaleOffsetInStruct_ = expOffsetInStruct_ + expLenInStruct_;

    tokenGapInStruct_ = (tokenStructLen_ - tokenLenInStruct_) / UB_32B_ALIGN;
    infoGapInStruct_ = (tokenStructLen_ - expLenInStruct_) / UB_32B_ALIGN;

    // Input/Output global tensor init
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    expertTokenNumsOutGM_ = expertTokenNumsOut;  // 无GlobalTensor
    epRecvCountsGM_ = epRecvCountsOut;           // 无GlobalTensor
    expandXGM_ = x;
    expandIdxGM_ = expertIds;

    // RDMA send/recv global tensor init
    sendTokensU8Tensor_.SetGlobalBuffer((__gm__ uint8_t *)(windowOutGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ uint64_t *)(windowOutGM_ + WIN_SIZE));
    readStatusTensor_.SetGlobalBuffer((__gm__ uint64_t *)(windowInGM_ + WIN_SIZE));

    // Global work space init
    tokenAddrFlagStructGM_ = workspaceGM;
    tokenAddrFlagStructGlobalU64Tensor_.SetGlobalBuffer(
        (__gm__ uint64_t *)(tokenAddrFlagStructGM_),
        axisBS_ * FLAG_SIZE);  // 存放bs个token要发送的serverId，每个token用一个int64表示，没饿过bit位表示一个serverId

    // Combine info offset init
    combineInnerCntOffset = localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t);
    combineInnerCntIndexOffset = combineInnerCntOffset + globalBs_ * serverNum * sizeof(int16_t);
    combineOuterCntOffset = combineInnerCntIndexOffset + globalBs_ * axisK_ * serverNum * sizeof(int32_t);
    combineOuterCntIndexOffset = combineOuterCntOffset + axisBS_ * sizeof(int32_t);
    combineOuterCntPerServerOffset = combineOuterCntIndexOffset + axisBS_ * serverNum * sizeof(int32_t);
    moeExpertNumInServer_ = SERVER_RANK_SIZE * localMoeExpertNum_;

    // UB init
    tpipe_->InitBuffer(statusBuf_, FLAG_SIZE);

    tpipe_->InitBuffer(rdmaInBuf_, UB_32B_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_32B_ALIGN);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();

    tpipe_->InitBuffer(tBuf, TBUF_SIZE);

    // The maximum value of expertIdsCnt_ is 512 * 16, so there is no integer wrap.
    uint32_t expertIdsSize = RoundUp(expertIdsCnt_ * static_cast<uint32_t>(sizeof(int16_t)), UB_32B_ALIGN);
    tokenUbSize_ = TBUF_SIZE - TBUF_TEMP_OFFSET - expertIdsSize;  // = 190KB-8KB-16KB=166KB
    expertIdsI16Tensor_ =
        tBuf.GetWithOffset<int16_t>(axisBS_ * alignK_, tokenUbSize_ + TBUF_TEMP_OFFSET);  // 512 * 16 * 2 = 16KB

    // RDMA发送完成标志初始化
    if (aivId_ == 0) {
        sendStatusTensor_.SetValue(0, END_OF_WRITE_FLAG_VALUE);
        DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            sendStatusTensor_);
    }

    // 每次调用magic++,用来区分不同轮次
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::AIVRDMAPostSend(
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

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while (1) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();  // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                      // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);  // [7] owner_bit
    byte_4 |= 1U << 8U;                        // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t *)(wqeAddr) = byte_4;          // Control set by local parameter see above lines
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen;  // message size
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;           // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U;  // [120:127] num_sge = 1
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;          // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)destDmaAddr;  // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;  // L_Key
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) =
        (uint64_t)srcDmaAddr;  // src VA addr memory registered by RNIC

    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;                     // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL;                           // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL;           // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL;  // [48:50] DB_SL = qp.sl

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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void
MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::CreateInnerReduceInfo(uint32_t serverIdx)
{
    // serverNum个Core加入本函数
    uint32_t curServerId = serverIdx;
    uint32_t currServerExpBegin = rankId_ / 8 * moeExpertNumInServer_;       // 目标Server的起始专家
    uint32_t currServerExpEnd = currServerExpBegin + moeExpertNumInServer_;  // 目标Server的结束专家
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
        if (serverId_ == serverIdx) {  // 如果是本机的token，只拷贝expertIds
            tokenStatus = GetSelfServerTokenInfo(selfTokenIdx, true, localUB_U8[tokenIdx * expLenInStruct_]);
            if (tokenStatus == SKIP_STATUS || tokenStatus == ARRIVAL_STATUS) {
                selfTokenIdx++;
            }
        } else {  // 如果不是本机的token，收到flag后，只拷贝expertIds
            tokenStatus = GetArrivedTokenInfo(curServerId, tokenIdx, true, localUB_U8[tokenIdx * expLenInStruct_]);
        }
        PipeBarrier<PIPE_ALL>();
        if (tokenStatus != ARRIVAL_STATUS) {
            continue;
        } else {
            tokenIdx += 1;
        }
    }

    uint32_t realBS = tokenIdx;  // rankId_ 实际收到的token数
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

    Duplicate<int16_t>(combineReduceInfo, int16_t(-1),
                       moeExpertNumInServer_ * realBS);  // 记录当前server的每个expert接收的tokenIndex,
                                                         // 从小到大（极端情况下，每个专家可能接收realBs个token）
    Duplicate<int32_t>(expCntMap, int32_t(0), moeExpertNumInServer_);  // 记录每个专家接收的token数的前缀和
    Duplicate<int32_t>(tokenOffset, int32_t(0),
                       realBS * alignK_);  // 记录 每个token 在每个expert的偏移，下标对应于 exprtiIds 数组下标
    Duplicate<int16_t>(
        innerCntLt, 0,
        RoundUp(realBS + aivNum_, B16_PER_BLOCK));  // 记录每个token被当前server的每个expert接收的次数的前缀和
    Duplicate<int32_t>(innerOffsetLt, 0, realBS * alignK_);

    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 根据 tokenStruct 的 expertIds 计算offset
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

    // 计算当前server内每个专家（0~expertNumInServer-1）接收token的前缀和
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
                    uint32_t expIdInServer = expId % moeExpertNumInServer_;        // server id
                    uint32_t rankIdInServer = expIdInServer / localMoeExpertNum_;  // rank id
                    combineReduceInfo(expIdInServer * realBS + tokenOffset(tokenId * axisK_ + expIdx)) = -1;
                    innerCntLt(tokenOccurNum) += 1;
                    innerOffsetLt(expOccurNum) = (expIdInServer % localMoeExpertNum_ == 0)
                                                     ? 0
                                                     : expCntMap(expIdInServer - 1);       // token在rank内的偏移
                    innerOffsetLt(expOccurNum) += rankIdInServer * globalBs_ * axisK_;     // token在rank间的偏移
                    innerOffsetLt(expOccurNum) += tokenOffset(tokenId * axisK_ + expIdx);  // token在专家内的偏移
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::CreateOuterReduceInfo()
{
    // 仅一个核进去该逻辑
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;

    LocalTensor<int32_t> miniServerExpIds =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> tokenInServerNums =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffset_ = tBuf.GetWithOffset<int32_t>(moeExpertNum_, baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> combineOffsetIdx_ =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerCntLt = tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerOffsetLt =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> outerCntPerServerLt =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(axisBS_ * serverNum, TBUF_OFFSET_ALIGN_B32_CNT);

    LocalTensor<int32_t> expertIdsI32Tensor =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK), baseBuffOffset);

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor, expertIdsGMTensor_, expCopyParams, expPadParams);

    Duplicate<int32_t>(miniServerExpIds, int32_t(moeExpertNum_),
                       RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK));  // 记录 每个token发给每个server内的最小expertId
    Duplicate<int32_t>(combineOffset_, int32_t(0), moeExpertNum_);
    Duplicate<int32_t>(tokenInServerNums, 0, RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK));
    Duplicate<int32_t>(outerCntLt, 0, RoundUp(axisBS_, BITS32_PER_BLOCK));
    Duplicate<int32_t>(outerOffsetLt, 0, RoundUp(axisBS_ * serverNum, BITS32_PER_BLOCK));
    Duplicate<int32_t>(outerCntPerServerLt, 0,
                       RoundUp(axisBS_ * serverNum,
                               BITS32_PER_BLOCK));  // 记录每个token在每个server上被接收的个数，不记录接收数量为0的数据

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();

    // ServerIdx，统计token去往了哪些server，以及在server上的偏移，统计目的专家信息
    for (uint32_t expertIndex = 0; expertIndex < expertIdsCnt_; ++expertIndex) {
        uint32_t tokenIdx = expertIndex / axisK_;
        uint32_t expId = expertIdsI32Tensor(expertIndex);
        uint32_t expServerId = expId / moeExpertNumInServer_;            // 专家在第几个serve
        uint32_t tokenInServerIdx = tokenIdx * serverNum + expServerId;  // 计算tokenIdx要发送多少个到每个serverr

        tokenInServerNums(tokenInServerIdx) += 1;

        // 当前token每个目的server,统计其最小expId
        if (miniServerExpIds(tokenInServerIdx) > expId) {
            miniServerExpIds(tokenInServerIdx) = expId;
        }

        if (expertIndex % axisK_ != axisK_ - 1) {
            continue;
        }

        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            tokenInServerIdx = tokenIdx * serverNum + serverIdx;
            uint32_t miniServerExpId = miniServerExpIds(tokenInServerIdx);
            if (miniServerExpId != moeExpertNum_) {
                combineOffsetIdx_(tokenInServerIdx) = combineOffset_(miniServerExpId);
                combineOffset_(miniServerExpId) += tokenInServerNums(tokenInServerIdx);  //
            }
        }
    }
    // 计算前序和
    for (uint32_t expertIndex = 1; expertIndex < moeExpertNum_; ++expertIndex) {
        combineOffset_(expertIndex) += combineOffset_(expertIndex - 1);
    }

    // 第三次遍历，填充bs个token的Reduceinfo
    uint32_t outerOffsetIdx = 0;
    for (uint32_t tokenIdx = 0; tokenIdx < axisBS_; ++tokenIdx) {
        // 将cnt,offset填写到InfoTensor对应的位置
        for (uint32_t serverIdx = 0; serverIdx < serverNum; ++serverIdx) {
            // 对于无效server跳过
            uint32_t tokenInServerIdx = tokenIdx * serverNum + serverIdx;
            uint32_t miniServerExpId = miniServerExpIds(tokenInServerIdx);
            if (miniServerExpId == moeExpertNum_) {
                continue;
            }
            outerCntLt(tokenIdx) += 1;
            outerCntPerServerLt(outerOffsetIdx) = tokenInServerNums(tokenInServerIdx);
            uint32_t preServerCnt = (serverIdx == 0) ? 0 : combineOffset_(serverIdx * moeExpertNumInServer_ - 1);
            uint32_t serverBaseCnt = serverIdx * axisBS_ * axisK_;
            uint32_t preTokenCnt = (miniServerExpId == 0) ? 0 : combineOffset_(miniServerExpId - 1);
            uint32_t tokenOffset = preTokenCnt - preServerCnt + combineOffsetIdx_(tokenInServerIdx) + serverBaseCnt;
            outerOffsetLt(outerOffsetIdx) = tokenOffset;
            outerOffsetIdx++;
        }
    }

    // 第四次遍历获取累加和
    for (uint32_t tokenIdx = 1; tokenIdx < axisBS_; ++tokenIdx) {
        outerCntLt(tokenIdx) += outerCntLt(tokenIdx - 1);
    }

    GlobalTensor<int32_t> combineOuterCntGt;
    GlobalTensor<int32_t> combineOuterOffsetGt;
    GlobalTensor<int32_t> combineOuterCntPerServerGt;

    combineOuterCntGt.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineOuterCntOffset));
    DataCopyExtParams outerCntParams{1, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterCntGt, outerCntLt, outerCntParams);
    PipeBarrier<PIPE_ALL>();

    combineOuterOffsetGt.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineOuterCntIndexOffset));
    DataCopyExtParams outerOffsetParams{1, static_cast<uint32_t>(axisBS_ * serverNum * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterOffsetGt, outerOffsetLt, outerOffsetParams);
    PipeBarrier<PIPE_ALL>();

    combineOuterCntPerServerGt.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_ + combineOuterCntPerServerOffset));
    DataCopyExtParams outerCntPerServerParams{1, static_cast<uint32_t>(axisBS_ * serverNum * sizeof(int32_t)), 0, 0, 0};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(combineOuterCntPerServerGt, outerCntPerServerLt, outerCntPerServerParams);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::ReorderTokens()
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

    LocalTensor<int32_t> expertIdsI32Tensor = tBuf.Get<int32_t>(RoundUp(axisBS_ * axisK_, BITS32_PER_BLOCK));

    DataCopyExtParams expCopyParams{1, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expPadParams;
    DataCopyPad(expertIdsI32Tensor, expertIdsGMTensor_, expCopyParams, expPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(expertIdsI16Tensor_, expertIdsI32Tensor, RoundMode::CAST_NONE, axisBS_ * axisK_);
    SyncFunc<AscendC::HardEvent::V_MTE2>();

    // 计算单个token在ub中占用buffer大小，量化情况下还包含量化所学workspace
    uint32_t singleTokenUBSize = tokenStructLen_;
    uint32_t quantTokenUBSize = 0;
    if constexpr (DynamicQuant) {
        quantTokenUBSize = tokenStructLen_ > axisH_ * sizeof(XType) ? tokenStructLen_ : axisH_ * sizeof(XType);
        singleTokenUBSize = quantTokenUBSize + axisH_ * sizeof(float);
    }
    uint32_t maxTokenNumInUB = tokenUbSize_ / singleTokenUBSize;
    uint32_t batchNum = (sendTokenNum + maxTokenNumInUB - 1) / maxTokenNumInUB;

    LocalTensor<uint8_t> tokenTensorU8_ = tBuf.GetWithOffset<uint8_t>(
        maxTokenNumInUB * tokenStructLen_, TBUF_TEMP_OFFSET);  // 存放tokenStruct的tensor，uint8_t类型
    LocalTensor<uint64_t> tokenTempTensorU64_ =
        tokenTensorU8_.template ReinterpretCast<uint64_t>();  // 存放tokenStruct的tensor，uint64_t类型
    LocalTensor<XType> tokenLt = tokenTensorU8_.template ReinterpretCast<XType>();  // 存放tokenStruct的tensor，fp16类型
    LocalTensor<float> tokenCastLt;                                                 // 仅量化使用
    GlobalTensor<uint8_t> expertIdsGMTensorU8_;
    GlobalTensor<uint8_t> xGMtU8;
    xGMtU8.SetGlobalBuffer((__gm__ uint8_t *)expandXGM_);
    expertIdsGMTensorU8_.SetGlobalBuffer((__gm__ uint8_t *)expandIdxGM_);

    if constexpr (DynamicQuant) {
        uint32_t tokenCastLtOffset = RoundUp(TBUF_TEMP_OFFSET + quantTokenUBSize * maxTokenNumInUB, UB_32B_ALIGN);
        tokenCastLt =
            tBuf.GetWithOffset<float>(axisH_ * maxTokenNumInUB, tokenCastLtOffset);  // 临时存放float类型token的tensor
    }

    // 每次向OutGM中拷贝batchNum个tokenStruct
    for (uint32_t batchIndex = 0; batchIndex < batchNum; batchIndex++) {
        // token 进行拷贝
        uint32_t currentTokenNum = sendTokenNum > maxTokenNumInUB ? maxTokenNumInUB : sendTokenNum;
        if constexpr (DynamicQuant) {
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
        // ExpertId进行拷贝
        DataCopyExtParams expCopyParams{static_cast<uint16_t>(currentTokenNum), static_cast<uint32_t>(realLenInStruct_),
                                        0, static_cast<uint32_t>(infoGapInStruct_), 0};
        DataCopyPadExtParams<uint8_t> expPadParams;
        DataCopyPad(tokenTensorU8_[expOffsetInStruct_], expertIdsGMTensorU8_[startTokenId * realLenInStruct_],
                    expCopyParams, expPadParams);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t tokenIndex = 0; tokenIndex < currentTokenNum; ++tokenIndex) {
            // 获取token在WinOut的地址
            // flag 进行 写入
            uint32_t tokenId = startTokenId + tokenIndex;
            uint32_t startExpId = tokenId * axisK_;
            uint32_t flagOffset = (tokenIndex * tokenStructLen_ + flagOffsetInStruct_) / sizeof(uint64_t);
            tokenTempTensorU64_(flagOffset) = SHOULD_SEND_FLAG_VALUE;

            uint64_t sendServerInfo = 0;  // 用64个bit位表示 每个serverId是否需要发送，最多支持64个server
            for (uint32_t i = 0; i < axisK_; i++) {
                uint32_t expertId = static_cast<uint32_t>(expertIdsI16Tensor_(startExpId + i));  // 读取expId
                uint32_t dstServerId = expertId / moeExpertNumInServer_;  // 计算出 该expId所在的serverId
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::QuantProcess(
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
    LocalTensor<float> maxLt = tBuf.GetWithOffset<float>(maxArrLen, maxArrUbOffset);  // max|min|max(abs)

    Cast(tokenCastLt, xTokenLt, RoundMode::CAST_NONE, sendTokenNum * axisH_);  // bf16->float32
    for (int32_t i = 0; i < sendTokenNum; ++i) {
        PipeBarrier<PIPE_V>();
        if constexpr (DynamicQuant) {
            ReduceMax(maxLt[maxValOffset], tokenCastLt[i * axisH_], workLt, axisH_,
                      false);  // token tensor中hidden个元素 的最大值
            SyncFunc<AscendC::HardEvent::V_S>();
            PipeBarrier<PIPE_V>();
            ReduceMin(maxLt[minValOffset], tokenCastLt[i * axisH_], workLt, axisH_,
                      false);  // token tensor中hidden个元素 的最小值
            PipeBarrier<PIPE_V>();
            Abs(maxLt, maxLt, maxArrLen - 1);  // 对token 的 最大值和最小值取绝对值
            PipeBarrier<PIPE_V>();
            ReduceMax(maxLt[resValOffset], maxLt, workLt, maxArrLen - 1, false);  // token绝对值的最大值

            SyncFunc<AscendC::HardEvent::V_S>();
            float maxVal = maxLt(resValOffset);
            dynamicScale = float(quantMax) / float(maxVal);  // 根据 token绝对值的最大值 计算出动态量化scale值
            SyncFunc<AscendC::HardEvent::S_V>();
            Muls(tokenCastLt[i * axisH_], tokenCastLt[i * axisH_], dynamicScale, axisH_);  // 对token量化
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<half> halfLocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = tokenCastLt[i * axisH_].template ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, tokenCastLt[i * axisH_], RoundMode::CAST_RINT,
             axisH_);  // 将量化后的token从float类型 转为int32类型
        PipeBarrier<PIPE_V>();
        SetDeqScale(deqScale);
        PipeBarrier<PIPE_V>();

        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);  // 将量化后的token从int32类型 转为half类型

        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXOutType> xOutTensor;
        LocalTensor<uint8_t> tokenUnitLt;
        tokenUnitLt = xTokenLt.template ReinterpretCast<uint8_t>();
        xOutTensor = tokenUnitLt[i * tokenStructLen_ + tokenOffsetInStruct_].template ReinterpretCast<ExpandXOutType>();
        Cast(xOutTensor, halfLocalTemp, RoundMode::CAST_TRUNC,
             axisH_);  // 将量化后的token从half类型 转为int8类型,放到xTokenLt中

        LocalTensor<float> scaleTensor =
            tokenUnitLt[i * tokenStructLen_ + scaleOffsetInStruct_].template ReinterpretCast<float>();
        scaleTensor.SetValue(0, float(1.0) / dynamicScale);  // 将 动态量化scale值 放到xTokenLt中token的后面
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void
MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::SendDataToServer(uint32_t destServerId)
{
    uint32_t dstRankId =
        rankId_ % SERVER_RANK_SIZE + destServerId * SERVER_RANK_SIZE;  // 根据自己的rank计算其他节点的同号rank
    uint64_t destServerMask = (1UL << destServerId);

    // 根据BufferID选择对应WindowBuffer -> 根据对应本机的Server选择Dst对应预留区域
    uint64_t dstRdmaAddr = (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + (halfWinSize_ * bufferId_ * 1UL) +
                                      (serverId_ * SERVER_SIZE_ON_WIN * 1UL));
    uint64_t srcRdmaAddrBase = (uint64_t)(hccl_.GetWindowsOutAddr(rankId_) + (halfWinSize_ * bufferId_ * 1UL));
    LocalTensor<uint64_t> sendTokenInfoLocalTensor =
        tBuf.GetWithOffset<uint64_t>((axisBS_ * FLAG_SIZE) / sizeof(uint64_t), 0);
    DataCopy(sendTokenInfoLocalTensor, tokenAddrFlagStructGlobalU64Tensor_, (axisBS_ * FLAG_SIZE) / sizeof(uint64_t));
    PipeBarrier<PIPE_ALL>();

    for (uint32_t tokenIdx = 0; tokenIdx < axisBS_; ++tokenIdx) {
        uint64_t destServerInfo = sendTokenInfoLocalTensor(tokenIdx * FLAG_SIZE / sizeof(uint64_t));
        if ((destServerInfo & destServerMask) !=
            0) {  // 当前有需要发送的token立即发送，每个token只会发一份给同一个server
            uint64_t srcRdmaAddr = (uint64_t)(srcRdmaAddrBase + (tokenStructLen_ * tokenIdx * 1UL));
            AIVRDMAPostSend((GM_ADDR)srcRdmaAddr, (GM_ADDR)dstRdmaAddr, dstRankId, tokenStructLen_, qp_info_);
            dstRdmaAddr += tokenStructLen_;
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 发送完成标志到 dstServer
    uint64_t srcFlagRdmaAddr = (uint64_t)(sendStatusTensor_.GetPhyAddr());
    uint64_t dstFlagRdmaAddr =
        (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + halfWinSize_ * bufferId_ + WIN_SIZE + serverId_ * STATE_OFFSET);
    AIVRDMAPostSend((GM_ADDR)srcFlagRdmaAddr, (GM_ADDR)dstFlagRdmaAddr, dstRankId, FLAG_SIZE, qp_info_);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::GetExpRank(uint32_t expertId)
{
    return expertId / localMoeExpertNum_;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::SetIpcFlag(int32_t flagVal)
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::WaitIpcFlag(int32_t flagVal)
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
        // 当有core未达到checkValue的阶段时，继续等待
        uint64_t tempVal = localWait.GetValue(0);
        if (tempVal >= waitVal) {
            break;
        }
    } while (isSync);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::GetArrivedTokenInfo(
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

    // 等到发送结束信号，没等到token结束信号，则返回结束等待状态
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
        // 等待 token 或者 endOfWrite
        PipeBarrier<PIPE_ALL>();
        return WAIT_STATUS;
    } else {  // 得到上个token->可以处理
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::GetSelfServerTokenInfo(
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::Win2Ipc()
{
    uint32_t coresPerServer = (aivNum_ - serverNum - 1) / serverNum;
    uint32_t logicAivId = aivId_ - serverNum - 1;
    if (logicAivId >= coresPerServer * serverNum) {
        return;
    }
    // 计算本core需要处理的ServerId
    uint32_t formServerId = logicAivId / coresPerServer;

    uint32_t expStartId = serverId_ * SERVER_RANK_SIZE * localMoeExpertNum_;
    uint32_t expEndId = expStartId + SERVER_RANK_SIZE * localMoeExpertNum_;
    // 获取到达的Token，统计专家信息，并且完成Ipc发送
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

            // 本卡需要发送
            uint32_t targetExpOffset =
                (targetExpId % localMoeExpertNum_) * worldSize_ * RANK_SIZE_ON_IPC;            // 第几个Exp段
            uint32_t targetServerOffset = formServerId * SERVER_RANK_SIZE * RANK_SIZE_ON_IPC;  // 第几个Server段
            uint32_t targetRankOffset = (rankId_ % SERVER_RANK_SIZE) * RANK_SIZE_ON_IPC;       // 第几个Rank段
            uint32_t targetTokenOffset = tokenStructLen_ * targetTokenIdx;                     // 第几个Token位
            uint32_t targetOffset =
                targetExpOffset + targetServerOffset + targetRankOffset + targetTokenOffset;  // 总偏移
            targetTokenIpcGt.SetGlobalBuffer(
                (__gm__ uint8_t *)(shareAddrs[targetRankId % SERVER_RANK_SIZE] + IPC_DATA_OFFSET + targetOffset));
            PipeBarrier<PIPE_ALL>();
            DataCopy(targetTokenIpcGt, localUB_U8, tokenStructLen_);
            PipeBarrier<PIPE_ALL>();
        }
        tokenIdx += 1;
        justExpInfo = (tokenIdx % coresPerServer != logicAivId % coresPerServer);
    }
    // 数据发送结束，填写tokenNum到对端Ipc，每轮填写coresPerServer个，总共要填写 SERVER_RANK_SIZE * localMoeExpertNum_个
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::Ipc2Out()
{
    uint32_t coresPerExp = aivNum_ / localMoeExpertNum_;
    if (aivId_ >= coresPerExp * localMoeExpertNum_) {
        return;
    }
    uint32_t coresPerServer = aivNum_ / serverNum;
    uint32_t localRankId = rankId_ % SERVER_RANK_SIZE;
    GlobalTensor<int32_t> flagIpcGt;
    flagIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE]));
    // PipeBarrier<PIPE_ALL>();
    uint32_t curExpIdx = aivId_ / coresPerExp;   // 当前处理的专家在本卡上的Idx
    uint32_t localAivId = aivId_ % coresPerExp;  // 处理本专家的同一批Core中，本Core的Idx
    // 每个exp对应ranksize行
    uint32_t srCntPerExp = serverNum * SERVER_RANK_SIZE;
    // 平均每个核处理多少行
    uint32_t srCntPerCore = srCntPerExp / coresPerExp;
    // 平分后还剩多少行
    uint32_t srCntRemain = srCntPerExp % coresPerExp;
    // 前面的核共分到了多少剩余
    uint32_t srCntPreRemain = (localAivId < srCntRemain) ? localAivId : srCntRemain;
    // 当前核分到多少行
    uint32_t srCntCurCore = (localAivId < srCntRemain) ? (srCntPerCore + 1) : srCntPerCore;

    GlobalTensor<int32_t> tokenCntIpcGt;
    tokenCntIpcGt.SetGlobalBuffer((__gm__ int32_t *)(shareAddrs[rankId_ % SERVER_RANK_SIZE] + IPC_TOKEN_CNT_OFFSET));

    // tBuf 内存分配
    // 2k ~ 4k 保存按expert统计的token个数信息
    LocalTensor<int64_t> tokenCntByExpUB = tBuf.GetWithOffset<int64_t>(2 * 1024 / sizeof(int64_t), 2 * 1024);
    // 4k ~ 6k 保存token个数统计信息
    LocalTensor<int32_t> tokenCntUB = tBuf.GetWithOffset<int32_t>(2 * 1024 / sizeof(int32_t), 4 * 1024);

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
        // 循环本Core需要处理的Rank数
        uint32_t srIdx = srPreCnt + idx;
        int32_t curSrTokenCnt = tokenCntUB(srIdx) - (srIdx == 0 ? 0 : tokenCntUB(srIdx - 1));
        if (curSrTokenCnt == 0) {
            continue;
            // 目标Rank没Token发来则跳过
        }
        uint32_t tokenCntInUB = tokenUbSize_ / tokenStructLen_;
        // 单次能搬移的token数据量
        uint32_t batchCnt = (curSrTokenCnt + tokenCntInUB - 1) / tokenCntInUB;
        // 循环搬运次数
        // 分批逻辑待修改，应该是先收集所有待处理Rank的Token，再写out
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
            // dynamic scales to output
            if constexpr (DynamicQuant) {
                DataCopyExtParams quantTokenParams{
                    static_cast<uint16_t>(tokenCntInBatch), static_cast<uint32_t>(sizeof(float)),
                    static_cast<uint32_t>((tokenStructLen_ - UB_32B_ALIGN) / UB_32B_ALIGN), 0, 0};

                LocalTensor<float> quantTempUB = localUB[scaleOffsetInStruct_].ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[(sumTokenCnt + batchIdx * tokenCntInUB)], quantTempUB,
                            quantTokenParams);
            }
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
        sumTokenCnt += curSrTokenCnt;
    }
    if (aivId_ == 0) {
        // 搬运token统计信息到output
        GlobalTensor<int32_t> tokenNumsGlobal;
        tokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_));
        DataCopyExtParams countsParams{
            1, static_cast<uint32_t>(localMoeExpertNum_ * serverNum * SERVER_RANK_SIZE * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(tokenNumsGlobal, tokenCntUB, countsParams);

        // 搬运按expert的token信息到output
        GlobalTensor<int64_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int64_t *)(expertTokenNumsOutGM_));
        DataCopyExtParams writeCountsParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expertTokenNumsGlobal, tokenCntByExpUB, writeCountsParams);
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::CleanUp()
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

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeDispatchV2Layered<TemplateMC2TypeA2layeredFunc>::Process()
{
    if ASCEND_IS_AIV {    // 全aiv处理
        ReorderTokens();  // 前axisBS_个核处理，重排token，将tokenStruct放在一起，计算token索引，以及每个token要发给哪些server
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        if (aivId_ < serverNum) {  // 前serverNum个核做跨机通信，比如2机就是0-1核做跨机通信
            if (aivId_ !=
                serverId_) {  // 与serverId_不相同的核将数据发送到其他server，每个token只会发一份到同一个server
                SendDataToServer(aivId_);
            }
            CreateInnerReduceInfo(aivId_);  // 前serverNum个核 计算 recvCount 机内通信的count和index
        } else if (aivId_ == serverNum) {
            CreateOuterReduceInfo();  // 第serverNum个核 计算 recvCount 机间通信的count和index
        } else {
            Win2Ipc();  // 剩余核 做IPC通信，将token发送给对应专家所在的rank
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetIpcFlag(IPC_FLAG_STEP_1);
        WaitIpcFlag(IPC_FLAG_STEP_1);
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        Ipc2Out();  // 接收IPC通信发来的token，并拷贝到OutGM
        if (aivId_ < serverNum) {
            PipeBarrier<PIPE_ALL>();
            CleanUp();
        }

        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        hccl_.Finalize();
    }
}
}  // namespace MoeDistributeDispatchA2Impl
#endif  // MOE_DISTRIBUTE_DISPATCH_A2_LAYERED_H
