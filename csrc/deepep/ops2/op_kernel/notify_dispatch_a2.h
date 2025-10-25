#ifndef NOTIFY_DISPATCH_A2_H
#define NOTIFY_DISPATCH_A2_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "sync_collectives.h"
#include "moe_distribute_base.h"
#include "notify_dispatch_tiling_a2.h"

using namespace AscendC;
using namespace Moe;

#define KERNELS_ARGS_FUN_A2_ALL2ALL()                                                                                  \
    GM_ADDR sendDataInput, GM_ADDR tokenPerExpertDataInput, GM_ADDR tmpDataInput, GM_ADDR sendDataOffsetOutput,        \
        GM_ADDR recvDataOutput, int64_t len, int64_t numTokens, int64_t topkNum, int64_t numExperts, int op, int root, \
        int cycleCount, GM_ADDR scale, int64_t scaleCount, GM_ADDR offset, int localRank, int localRankSize,           \
        GM_ADDR commArgs, GM_ADDR tokenServerIdxOutput, GM_ADDR tokensUniquePerServerOutput,                           \
        GM_ADDR epRankTokenCntOutput, GM_ADDR localEpTokenCntOutput, GM_ADDR srcOffsetRankTokenIdxOutput,              \
        GM_ADDR dstOffsetRankTokenIdxOutput, GM_ADDR offsetInnerOutput, GM_ADDR countOuterOutput,                      \
        GM_ADDR expandIdxOutput, GM_ADDR workspace, GM_ADDR tiling

#define KERNELS_ARGS_CALL_A2_ALL2ALL()                                                                            \
    sendDataInput, tokenPerExpertDataInput, tmpDataInput, sendDataOffsetOutput, recvDataOutput, len, numTokens,   \
        topkNum, numExperts, op, root, cycleCount, scale, scaleCount, offset, localRank, localRankSize, commArgs, \
        tokenServerIdxOutput, tokensUniquePerServerOutput, epRankTokenCntOutput, localEpTokenCntOutput,           \
        srcOffsetRankTokenIdxOutput, dstOffsetRankTokenIdxOutput, offsetInnerOutput, countOuterOutput,            \
        expandIdxOutput, workspace, tiling

#define printflag(ss)                                                   \
    if (blockIdx < coreNumBetween) {                                    \
        printf("========rank:%d coreIdx:%d " #ss "\n", rank, blockIdx); \
    }

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <typename T>
class NotifyDispatchA2
{
    constexpr static int32_t MAX_CORE_NUM = 20;
    constexpr static int64_t MULTI_RANK_SIZE = 4;  // 每个core最多往4个rank发送数据，64卡场景
    constexpr static int64_t MAX_RANK_SIZE = 64;   // 910B设备本算子最大支持的rank数，64卡场景
    constexpr static int32_t INVALID_RANK = -1;
    constexpr static uint32_t TEMP_BUF_LEN = 128 * 1024;  // tuf注册长度为128K，剩余部分注册为其他buffer
    constexpr static uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;  // 对齐tiling

    constexpr static uint32_t BW_ITEM_SIZE = 32;                               // = sizeof(BatchWriteItem)
    constexpr static uint32_t U64_PER_ITEM = BW_ITEM_SIZE / sizeof(uint64_t);  // 每个BatchWriteItem占多少个unit64
    constexpr static uint32_t U32_PER_ITEM = BW_ITEM_SIZE / sizeof(uint32_t);  // 每个BatchWriteItem占多少个unit32
    constexpr static uint32_t BW_MEB_OFFSET64_LOCAL_GM = 0;  // BatchWriteItem成员变量offset，按照sizeof(unit64)计算
    constexpr static uint32_t BW_MEB_OFFSET64_REMOTE_GM = 1;  // BatchWriteItem成员变量offset，按照sizeof(unit64)计算
    constexpr static uint32_t BW_MEB_OFFSET64_DATA_SIZE = 2;  // BatchWriteItem成员变量offset，按照sizeof(unit64)计算
    constexpr static uint32_t BW_MEB_OFFSET32_DATA_TYPE = 6;  // BatchWriteItem成员变量offset，按照sizeof(unit32)计算
    constexpr static uint32_t BW_MEB_OFFSET32_TARGET_RANK = 7;  // BatchWriteItem成员变量offset，按照sizeof(unit32)计算

    constexpr static int32_t FLAG_VALUE = 0xFFFFFFFF;
    constexpr static uint32_t STATUS_ENTRY_SIZE = 32;  // 每个status entry占用的空间大小, bytes
    constexpr static uint32_t U32_STATUS_ENTRY = STATUS_ENTRY_SIZE / sizeof(int32_t);
    constexpr static uint32_t FLAG_OFFSET = 8;          // status_flag 在 statusTensor中的offset, bytes
    constexpr static uint32_t SOURCE_RANK_OFFSET = 16;  // sourceRankId 在 statusTensor中的offset, bytes
    constexpr static uint32_t DEST_RANK_OFFSET = 20;    // destRankId 在 statusTensor中的offset, bytes
    constexpr static uint32_t DATALEN_OFFSET = 24;      // dataLen 在 statusTensor中的offset, bytes
    constexpr static uint32_t UB_ALIGN = 32;            // UB按32字节对齐
    constexpr static uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_ALIGN / sizeof(int32_t);  // 8
    constexpr static uint32_t GM_ALIGN = 64;                                          // GM按64字节对齐

    constexpr static uint32_t MAX_BS = 4096;  // 每卡支持的最大bs

public:
    __aicore__ inline NotifyDispatchA2(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_A2_ALL2ALL())
    {
        InitAll2AllLayeredRdma(KERNELS_ARGS_CALL_A2_ALL2ALL());

        tokenPerExpertDataAlignLen = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataOffsetAlignLen = Ceil(numExperts * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataAlignLen = Ceil(len * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;  // TODO: 数据长度
        perRankDataNum = len;                                                     // 发送所有数据

        InitTensorLen();

        InitShare();
        // 初始化核分组, 需要外部调用保证所有的server的localRankSize均相同
        serverNum = CeilDiv(rankSize, localRankSize);
        serverId = rank / localRankSize;
        // printf("rank:%d coreIdx:%d rankSize:%d localRankSize:%d serverNum:%d serverId:%d\n", rank, blockIdx,
        // rankSize, localRankSize, serverNum, serverId);
        InitCoreGroup();
        // 初始化目标rank列表
        InitTargetRank();
        // 初始化数据切片
        InitDataSlice();

        this->sendDataInput = (__gm__ T *)sendDataInput;
        this->tokenPerExpertDataInput = (__gm__ int32_t *)tokenPerExpertDataInput;
        this->tmpDataInput = (__gm__ int32_t *)tmpDataInput;
        this->sendDataOffsetOutput = (__gm__ T *)sendDataOffsetOutput;
        this->recvDataOutput = (__gm__ T *)recvDataOutput;
        this->epRankTokenCntOutputGM_ = (__gm__ int32_t *)epRankTokenCntOutput;

        sendDataInputGt.SetGlobalBuffer((__gm__ T *)sendDataInput);
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        tmpDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tmpDataInput);
        sendDataOffsetOutputGt.SetGlobalBuffer((__gm__ T *)sendDataOffsetOutput);
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);

        tokenServerIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)tokenServerIdxOutput);
        tokensUniquePerServerOutputGT_.SetGlobalBuffer((__gm__ int32_t *)tokensUniquePerServerOutput);
        epRankTokenCntOutputGT_.SetGlobalBuffer((__gm__ int32_t *)epRankTokenCntOutput);
        localEpTokenCntOutputGT_.SetGlobalBuffer((__gm__ int32_t *)localEpTokenCntOutput);
        srcOffsetRankTokenIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)srcOffsetRankTokenIdxOutput);
        dstOffsetRankTokenIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)dstOffsetRankTokenIdxOutput);
        offsetInnerOutputGT_.SetGlobalBuffer((__gm__ int32_t *)offsetInnerOutput);
        countOuterOutputGT_.SetGlobalBuffer((__gm__ int32_t *)countOuterOutput);
        expandIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutput);

        // 初始化RDMA相关变量
        // dataSpaceGT_ = workspace; // 需要预留大一些空间供存放交换后拆分出来的数据
        windowInGM_ = this->shareAddrs[rank];
        windowOutGM_ = hccl_.GetWindowsOutAddr(rank) + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE;
        batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t *)(workspace), rankSize * U32_PER_ITEM);
        // batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t*)(epRankTokenCntOutputGM_), rankSize * U32_PER_ITEM);
        // // 出参地址临时使用
        windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_ + IPC_DATA_OFFSET));
        windowInTensor_.SetGlobalBuffer((__gm__ T *)(windowInGM_ + IPC_DATA_OFFSET));
        windowOutstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_ + IPC_DATA_OFFSET));
        windowOutTensor_.SetGlobalBuffer((__gm__ T *)(windowOutGM_ + IPC_DATA_OFFSET));

        pipe.InitBuffer(batchWriteInfoBuf_, rankSize * BW_ITEM_SIZE);
        pipe.InitBuffer(tempBuf_, UB_ALIGN);                        // 存放临时的立即数
        pipe.InitBuffer(statusBuf_, rankSize * STATUS_ENTRY_SIZE);  // rankSize * 32B
        statusTensor_ = statusBuf_.Get<int32_t>();  // 保存发送数据量及flag，同时用于计算windows中的偏移
        Duplicate<int32_t>(statusTensor_, 0, rankSize * STATUS_ENTRY_SIZE);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            // 第一阶段，处理server间通信
            if (serverNum > 1) {
                ProcessBetweenServer();
            }
            printflag("beforeProcessWithinServer\n");

            // 第二阶段，处理server内通信
            ProcessWithinServer();
            SyncAll<true>();

            printflag("beforeSplitAndCalcData\n");
            // 交换后的数据拆分和计算输出
            SplitAndCalcData();  // TODO: 先验证recv_data
            SyncAll<true>();

            printflag("beforeFinalize\n");
            hccl_.Finalize();
            printflag("AfterFinalize\n");
        }
        // if (blockIdx == 0) {
        //     AscendC::DumpTensor(tokenServerIdxOutputGT_, 166, 16);
        //     AscendC::DumpTensor(countOuterOutputGT_, 167, 16);
        //     AscendC::DumpTensor(expandIdxOutputGT_, 168, 16);
        // }
        PRINTF("[notify] rank:%d, block:%d \n", rank, blockIdx);
    }

private:
    FORCE_INLINE_AICORE void InitAll2AllLayeredRdma(KERNELS_ARGS_FUN_A2_ALL2ALL())
    {
        this->root = 0;
        this->len = len;
        this->numExperts = numExperts;
        this->numTokens = numTokens;
        this->topkNum = topkNum;
        this->scale = nullptr;
        this->magic = 0;
        this->localRank = localRank;
        this->localRankSize = localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        uint8_t ctxIdx;

        ctxIdx = COMM_EP_IDX;

        // 初始化RDMA相关变量
        auto tilingData = (__gm__ NotifyDispatchA2TilingData *)tiling;
        __gm__ void *mc2InitTiling = (__gm__ void *)(&(tilingData->mc2InitTiling));
        __gm__ void *mc2CcTiling = (__gm__ void *)(&(tilingData->mc2CcTiling1));

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

        hccl_.Init(contextGM0, mc2InitTiling);
        hccl_.SetCcTiling(mc2CcTiling);
        this->winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)contextGM0;

        // 设置并自增magic
        magicTensor_.SetGlobalBuffer((__gm__ int32_t *)(hccl_.GetWindowsInAddr(rank) + IPC_DATA_OFFSET -
                                                        blockNum * sizeof(int32_t) * EXP_TOKEN_COUNT_FLAG_CNT));

        pipe.InitBuffer(this->tBuf, TEMP_BUF_LEN);
        LocalTensor<int32_t> tempLocal = tBuf.Get<int32_t>();
        tempLocal(0) = 1;
        // 使用atomic方式实现+1
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);  // 等待SetValue完成
        DataCopy(magicTensor_[blockIdx * EXP_TOKEN_COUNT_FLAG_CNT], tempLocal, EXP_TOKEN_COUNT_FLAG_CNT);
        AscendC::SetAtomicNone();
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);  // 等待DataCopy完成
        magic = magicTensor_.GetValue(blockIdx * EXP_TOKEN_COUNT_FLAG_CNT);
        PipeBarrier<PIPE_ALL>();
        // 初始化目标rank的shareAddrs
        for (int i = 0; i < rankSize; i++) {
            this->shareAddrs[i] = hccl_.GetWindowsInAddr(i) + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE;
        }

        sync.Init(this->rank, this->rankSize, this->shareAddrs, tBuf);
    }

    template <typename K, typename U = K>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U> &sendDataInputGt,
                                             const GlobalTensor<K> &recvDataOutputGT, int op);
    template <typename F>
    FORCE_INLINE_AICORE void SetAtomic(int op);
    FORCE_INLINE_AICORE void UnsetAtomic(int op);
    template <HardEvent eventType>
    FORCE_INLINE_AICORE void SetWaitEvent(event_t eventId);

    __aicore__ inline void InitTensorLen()
    {
        numTokensPerExpertAlignLen = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gNumTokensPerExpertAlignLen = Ceil(rankSize * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        numTokensUniquePerServerAlignLen = Ceil(serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gNumTokensUniquePerServerAlignLen = Ceil(rankSize * serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        numTokensPerServerAlignLen = Ceil(MAX_BS * serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gNumTokensPerServerAlignLen =
            Ceil(rankSize * MAX_BS * serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        tokenServerCntAlignLen = Ceil(MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gTokenServerCntAlignLen = Ceil(rankSize * MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        tokenServerIdxAlignLen = Ceil(MAX_BS * serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gTokenServerIdxAlignLen = Ceil(rankSize * MAX_BS * serverNum * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        tokenExpertIdxAlignLen = Ceil(MAX_BS * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gTokenExpertIdxAlignLen = Ceil(rankSize * MAX_BS * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        expertMaxBsSrcOffsetAlignLen = Ceil(numExperts * MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gExpertMaxBsSrcOffsetAlignLen =
            Ceil(rankSize * numExperts * MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        expertMaxBsOriOffsetAlignLen = Ceil(numExperts * MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        gExpertMaxBsOriOffsetAlignLen =
            Ceil(rankSize * numExperts * MAX_BS * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        /*
        if (blockIdx == 0) {
            PRINTF("[InitTensorLen] rank:%d, blockIdx:%d, send_count:%d, numExperts:%d, rankSize:%d, serverNum:%d,
        MAX_BS:%d, \
                numTokensPerExpertAlignLen:%d, gNumTokensPerExpertAlignLen:%d, \
                numTokensUniquePerServerAlignLen:%d, gNumTokensUniquePerServerAlignLen:%d, \
                numTokensPerServerAlignLen:%d, gNumTokensPerServerAlignLen:%d, \
                tokenServerCntAlignLen:%d, gTokenServerCntAlignLen:%d, \
                tokenServerIdxAlignLen:%d, gTokenServerIdxAlignLen:%d, \
                tokenExpertIdxAlignLen:%d, gTokenExpertIdxAlignLen:%d, \
                expertMaxBsSrcOffsetAlignLen:%d, gExpertMaxBsSrcOffsetAlignLen:%d, \
                expertMaxBsOriOffsetAlignLen:%d, gExpertMaxBsOriOffsetAlignLen:%d \n",
                rank, blockIdx, len, numExperts, rankSize, serverNum, MAX_BS,
                numTokensPerExpertAlignLen, gNumTokensPerExpertAlignLen,
                numTokensUniquePerServerAlignLen, gNumTokensUniquePerServerAlignLen,
                numTokensPerServerAlignLen, gNumTokensPerServerAlignLen,
                tokenServerCntAlignLen, gTokenServerCntAlignLen,
                tokenServerIdxAlignLen, gTokenServerIdxAlignLen,
                tokenExpertIdxAlignLen, gTokenExpertIdxAlignLen,
                expertMaxBsSrcOffsetAlignLen, gExpertMaxBsSrcOffsetAlignLen,
                expertMaxBsOriOffsetAlignLen, gExpertMaxBsOriOffsetAlignLen);
        }
        */
    }

    __aicore__ inline void InitShare()
    {
        int64_t queNum = MAX_CORE_NUM;
        queElemLen = (IPC_BUFF_MAX_SIZE - IPC_DATA_OFFSET) / sizeof(T) / queNum;  // 计算共享队列元素大小
        queSize = (queElemLen * sizeof(T) / GM_ALIGN) * GM_ALIGN;                 // GM 64字节对齐
        queLen = queSize / sizeof(T);  // 一个que的可放入的元素数量
    }

    __aicore__ inline void InitCoreGroup()
    {
        coreNumBetween = (rankSize <= MAX_CORE_NUM) ? rankSize : MAX_CORE_NUM;
        coreNumWithin = (rankSize <= MAX_CORE_NUM) ? rankSize : MAX_CORE_NUM;
        rankNumPerCore = CeilDiv(rankSize, MAX_CORE_NUM);  // 每个核负责的rank数
    }

    // 计算通信目标，分两个阶段：
    // 阶段一：处理Server间通信，Server间的同号卡之间进行Pair-wise的通信，顺序为从小到大的循环的环形
    // 阶段二：处理Server内通信，Server内的卡间进行fullmesh通信，同时需要将阶段一的数据传递给其他设备
    __aicore__ inline void InitTargetRank()
    {
        // 阶段一：server间的target rank, 此处表示数据最终的targetRank，并非直接发送的目标
        int32_t startRankId = blockIdx * rankNumPerCore;
        targetRankNum = (rankSize - startRankId) < rankNumPerCore ? (rankSize - startRankId) : rankNumPerCore;
        if (targetRankNum < 0) {
            targetRankNum = 0;
        }

        for (int i = 0; i < targetRankNum; i++) {
            targetRank[i] = startRankId + i;
        }
        // 其余值设置为 invalid
        for (int i = targetRankNum; i < MULTI_RANK_SIZE; i++) {
            targetRank[i] = INVALID_RANK;
        }
    }

    __aicore__ inline void InitDataSlice()
    {
        // 生产者负责搬运本rank的输入数据至共享内存，input-->share
        if (blockIdx < coreNumWithin) {
            writeGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[rank] + IPC_DATA_OFFSET));
        }
    }

    __aicore__ inline void ProcessWithinServer()
    {
        if (blockIdx < coreNumWithin) {
            InputToShareSlice();
            ShareToShareSlice();
        }
    }

    __aicore__ inline void InputToShareSlice()
    {
        if (blockIdx > 0) {
            return;
        }
        // 将本卡在Server内发送的input数据拷贝到本卡的共享内存对应位置
        int targetRankId = rank;
        int32_t targetServerId = targetRankId / localRankSize;

        int64_t datalen = this->len;
        readGt = sendDataInputGt[0];
        CpGM2GMPingPong<T>(datalen * sizeof(T), readGt, writeGt[queLen * targetRankId + STATUS_ENTRY_SIZE / sizeof(T)],
                           COPYONLY);  // 预留一个flag偏移位置
        // printflag("CpGM2GMPingPong\n");

        for (int i = 0; i < localRankSize; ++i) {
            int32_t curServerRankId = serverId * localRankSize + i;
            // printf("SetInner rank:%d coreIdx:%d magic:%d, curServerRankId:%d\n", rank, blockIdx, magic,
            // curServerRankId);
            sync.SetInnerFlag(magic, 1, curServerRankId, rank);
        }
        // AscendC::DumpTensor(writeGt[queLen * targetRankId + STATUS_ENTRY_SIZE / sizeof(T)], 338, datalen);
    }

    __aicore__ inline void ShareToShareSlice()
    {
        // 从Server内其他卡的共享内存对应位置拷贝数据到本卡的output
        if (blockIdx > 0) {
            return;
        }
        // printflag("ShareToShareSlice\n");
        int64_t recvCount = this->len;
        for (int i = 0; i < localRankSize; ++i) {
            int32_t targetRankId = serverId * localRankSize + i;
            // printf("WaitInner rank:%d coreIdx:%d magic:%d, targetRankId:%d\n", rank, blockIdx, magic, targetRankId);
            sync.WaitInnerFlag(magic, 1, rank, targetRankId);
            for (int j = 0; j < serverNum; ++j) {
                int32_t serverTarRankId = j * localRankSize + i;  // 对应为targetRankId的同号卡
                remoteGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[targetRankId] + IPC_DATA_OFFSET +
                                                      serverTarRankId * queSize +
                                                      STATUS_ENTRY_SIZE));  // 该rank上的第server块
                // PRINTF("[2ShareToShareSlice] rank:%d, blockId:%d, targetRankId:%d, serverTarRankId:%d", rank,
                // blockIdx, targetRankId, serverTarRankId); AscendC::DumpTensor(remoteGt, 362, recvCount);
                CpGM2GMPingPong<T>(recvCount * sizeof(T), remoteGt, recvDataOutputGt[serverTarRankId * this->len],
                                   COPYONLY);
            }
        }
    }

    __aicore__ inline void AssembleSendData()
    {
        pipe.InitBuffer(tokenPerExpertDataBuf, tokenPerExpertDataAlignLen);
        pipe.InitBuffer(sendDataBuf, sendDataAlignLen);
        pipe.InitBuffer(sendDataOffsetBuf, sendDataOffsetAlignLen);

        __ubuf__ int32_t *tokenPerExpertUB = (__ubuf__ int32_t *)get_imm(96);
        CpGM2UB(tokenPerExpertUB, (__gm__ int32_t *)tokenPerExpertDataInputGt.GetPhyAddr(), tokenPerExpertDataAlignLen);
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        __ubuf__ T *sendDataOffsetUB = (__ubuf__ T *)get_imm(96 + tokenPerExpertDataAlignLen);
        __ubuf__ T *sendDataUB = (__ubuf__ T *)get_imm(96 + tokenPerExpertDataAlignLen + sendDataOffsetAlignLen);

        int prefixSum = 0;
        for (int i = 0; i < numExperts; ++i) {
            int numTokensExpert = tokenPerExpertUB[i];
            sendDataUB[i * sendPerGroup] = numTokensExpert;
            sendDataUB[i * sendPerGroup + 1] = prefixSum;
            sendDataUB[i * sendPerGroup + 2] = numTokens;
            sendDataOffsetUB[i] = prefixSum;

            prefixSum += numTokensExpert;
        }
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);

        CpUB2GM((__gm__ T *)sendDataInputGt.GetPhyAddr(), sendDataUB, sendDataAlignLen);
        CpUB2GM((__gm__ T *)sendDataOffsetOutputGt.GetPhyAddr(), sendDataOffsetUB, sendDataOffsetAlignLen);
        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void ProcessBetweenServer()
    {
        InputToWindowOut();
        // printflag("AfterInputToWindowOut\n")
        ConstructBatchWriteInfo();
        // printflag("ConstructBatchWriteInfo\n")
        SyncAll<true>();
        SendRdma();
        // printflag("SendRdma\n")
        WaitRdma();
        // printflag("WaitRdma\n")
        SyncAll<true>();
        // printflag("before WindowInToOutput\n")
        WindowInToOutput();
        // printflag("WindowInToOutput\n")
    }

    // 从input将数据拷贝到windowOutTensor,供RDMA进行发送
    __aicore__ inline void InputToWindowOut()
    {
        /* statusFlag 和 dataFlag 为int32_t，各自占用8B中的前4Bytes
        ---------------------------------------------------------------------------------------------------------------
        |8B pads|flag 8B|source 4B|target 4B|datalen 4B|4B pads|   Data (datalen * sizeof(T))    | flag 8B | 24B pads |
        ---------------------------------------------------------------------------------------------------------------
        */
        if (blockIdx > 1) {
            return;
        }
        int32_t targetRankId = 0;
        if (blockIdx == 0) {
            // targetRankId = rank;
            return;                                                     // 同server的不搬运
        } else {                                                        // blockIdx=1
            targetRankId = (1 - serverId) * localRankSize + localRank;  // 2个server的计算方式，求对端同号卡rankid
        }
        int32_t targetServerId = targetRankId / localRankSize;

        int64_t datalen = this->len;
        readGt = sendDataInputGt[0];  // 读取全部数据

        // 计算各个位置的offset，in bytes
        int64_t statusEntryOffset = queSize * targetRankId;
        int64_t statusFlagOffset = statusEntryOffset + FLAG_OFFSET;
        int64_t sourceRankIdOffset = statusEntryOffset + SOURCE_RANK_OFFSET;
        int64_t destRankIdOffset = statusEntryOffset + DEST_RANK_OFFSET;
        int64_t dataLenOffset = statusEntryOffset + DATALEN_OFFSET;
        int64_t dataOffset = statusEntryOffset + STATUS_ENTRY_SIZE;
        int64_t dataFlagOffset = dataOffset + datalen * sizeof(T);
        CpGM2GMPingPong<T>(datalen * sizeof(T), readGt, windowOutTensor_[dataOffset / sizeof(T)], COPYONLY);
        // printflag("enter2 InputToWindowOut\n")

        windowOutstatusTensor_(statusFlagOffset / sizeof(int32_t)) = FLAG_VALUE;
        windowOutstatusTensor_(sourceRankIdOffset / sizeof(int32_t)) = rank;
        windowOutstatusTensor_(destRankIdOffset / sizeof(int32_t)) = targetRankId;
        windowOutstatusTensor_(dataLenOffset / sizeof(int32_t)) = (int32_t)datalen;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowOutstatusTensor_[(statusEntryOffset / sizeof(int32_t))]);
        windowOutstatusTensor_(dataFlagOffset / sizeof(int32_t)) = FLAG_VALUE;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowOutstatusTensor_[(dataFlagOffset / sizeof(int32_t))]);
        // PRINTF("##### rank:%d, blockId:%d, statusFlagOffset: %ld, windowOutstatusTensor_: %d, winOutAddr: %p \n",
        // rank, blockIdx,
        //     statusFlagOffset, windowOutstatusTensor_.GetValue(statusFlagOffset / sizeof(int32_t)),
        //     windowOutstatusTensor_[statusFlagOffset].GetPhyAddr());
        // PRINTF("#####2 rank:%d, blockId:%d, statusEntryOffset: %ld, statusFlagOffset: %ld, sourceRankIdOffset: %ld,
        // destRankIdOffset: %ld, dataLenOffset:%ld, dataOffset: %ld, dataFlagOffset: %ld\n",
        //     rank, blockIdx, statusEntryOffset, statusFlagOffset, sourceRankIdOffset, sourceRankIdOffset,
        //     destRankIdOffset, dataLenOffset, dataOffset, dataFlagOffset);
        // AscendC::DumpTensor(windowOutstatusTensor_[statusEntryOffset / sizeof(int32_t)], 495, 8);
        // printflag("enter3 InputToWindowOut\n")
    }

    // 创建RDMA使用的batch write信息
    __aicore__ inline void ConstructBatchWriteInfo()
    {
        if (targetRankNum == 0 || blockIdx > 0) {
            return;
        }

        LocalTensor<uint32_t> batchWriteU32Tensor_ = batchWriteInfoBuf_.Get<uint32_t>();
        LocalTensor<uint64_t> batchWriteU64Tensor_ = batchWriteInfoBuf_.Get<uint64_t>();
        uint32_t batchWriteDataType = static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        int32_t targetRankId = (1 - serverId) * localRankSize + localRank;  // 2个server的计算方式

        int32_t targetServerId = targetRankId / localRankSize;
        uint32_t sendToRankId = targetServerId * localRankSize + localRank;  // 数据发送目标Server的同号卡rankId

        // 数据在目标GM中的位置，保证第一轮数据不相互覆盖
        uint32_t sendOffset = serverId * localRankSize + (targetRankId % localRankSize);

        int64_t datalen = this->len;
        GM_ADDR localBuf = (__gm__ uint8_t *)(windowOutGM_ + IPC_DATA_OFFSET + targetRankId * queSize);
        GM_ADDR remoteGM = (__gm__ uint8_t *)(shareAddrs[sendToRankId] + IPC_DATA_OFFSET + rank * queSize);
        uint64_t batchWriteDataSize = datalen * sizeof(T) + 2 * STATUS_ENTRY_SIZE;  // payload加前后共2个flag长度

        batchWriteU64Tensor_(0 * U64_PER_ITEM + BW_MEB_OFFSET64_LOCAL_GM) = (uint64_t)localBuf;
        batchWriteU64Tensor_(0 * U64_PER_ITEM + BW_MEB_OFFSET64_REMOTE_GM) = (uint64_t)remoteGM;
        batchWriteU64Tensor_(0 * U64_PER_ITEM + BW_MEB_OFFSET64_DATA_SIZE) = batchWriteDataSize;
        batchWriteU32Tensor_(0 * U32_PER_ITEM + BW_MEB_OFFSET32_DATA_TYPE) = batchWriteDataType;
        batchWriteU32Tensor_(0 * U32_PER_ITEM + BW_MEB_OFFSET32_TARGET_RANK) = sendToRankId;

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        // AscendC::DumpTensor(batchWriteInfoTensor_, 544, rankSize * U32_PER_ITEM);
        DataCopy(batchWriteInfoTensor_[0], batchWriteU32Tensor_, 1 * U32_PER_ITEM);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SendRdma()
    {
        if (blockIdx == 0) {
            HcclHandle batchWrResult = hccl_.BatchWrite<true>((GM_ADDR)batchWriteInfoTensor_.GetPhyAddr(), 1);
        }
    }

    __aicore__ inline void WaitRdma()
    {
        if (targetRankNum == 0 || blockIdx > 0) {
            return;
        }

        DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        LocalTensor<int32_t> dataFlagLocal = tempBuf_.Get<int32_t>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();

        int32_t targetRankId = (1 - serverId) * localRankSize + localRank;  // 2个server的计算方式
        int32_t targetServerId = targetRankId / localRankSize;

        int64_t statusOffset = targetRankId * queSize + FLAG_OFFSET;
        // PRINTF("===rank:%d, blockId:%d, tarRankId:%d, tarServerId:%d, statusOffset: %ld, value: %d, winIn:%p \n",
        // rank, blockIdx, targetRankId, targetServerId, statusOffset,
        //     windowInstatusTensor_.GetValue(statusOffset / sizeof(int32_t)), windowInstatusTensor_[statusOffset /
        //     sizeof(int32_t)].GetPhyAddr());
        // AscendC::DumpTensor(windowInstatusTensor_[targetRankId * queSize / sizeof(int32_t)], 589, U32_STATUS_ENTRY);

        int64_t datalen = 0;
        int32_t statusFlag = 0;
        int32_t dataFlag = 0;
        // int64_t systemCycleBefore = AscendC::GetSystemCycle(); // 调用Add指令前的cycle数
        while (statusFlag != FLAG_VALUE) {
            DataCopy(statusTensor_[0], windowInstatusTensor_[targetRankId * queSize / sizeof(int32_t)],
                     U32_STATUS_ENTRY);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            statusFlag = statusTensor_(FLAG_OFFSET / sizeof(int32_t));
            datalen = statusTensor_(DATALEN_OFFSET / sizeof(int32_t));
            PipeBarrier<PIPE_MTE2>();

            // int64_t systemCycleAfter = AscendC::GetSystemCycle(); // 调用Add指令后的cycle数
            // if ((systemCycleAfter - systemCycleBefore) / 50 > 1000000) {
            //     PRINTF("[1statusFlag] rank:%d, blockId:%d, tarRankId:%d, tarServerId:%d, statusOffset: %ld, value:
            //     %d, winIn:%p \n", rank, blockIdx, targetRankId, targetServerId, statusOffset,
            //         windowInstatusTensor_.GetValue(statusOffset / sizeof(int32_t)),
            //         windowInstatusTensor_[statusOffset / sizeof(int32_t)].GetPhyAddr());
            //     AscendC::DumpTensor(windowInstatusTensor_[targetRankId * queSize / sizeof(int32_t)], 608,
            //     U32_STATUS_ENTRY * 32); break;
            // }
        }

        // int64_t systemCycleBefore2 = AscendC::GetSystemCycle(); // 调用Add指令前的cycle数
        uint64_t dataFlagOffset = (targetRankId * queSize + datalen * sizeof(T) + STATUS_ENTRY_SIZE) / sizeof(int32_t);
        while (dataFlag != FLAG_VALUE) {
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            dataFlag = dataFlagLocal(0);
            PipeBarrier<PIPE_MTE2>();
            // PRINTF("===[dataFlag]rank:%d, blockId:%d, tarRankId:%d, tarServerId:%d, dataFlag:%d \n",
            //     rank, blockIdx, targetRankId, targetServerId, dataFlag);

            // int64_t systemCycleAfter2 = AscendC::GetSystemCycle(); // 调用Add指令后的cycle数
            // if ((systemCycleAfter2 - systemCycleBefore2) / 50 > 1000000) {
            //     PRINTF("[1dataFlag] rank:%d, blockId:%d, tarRankId:%d, tarServerId:%d, dataFlagOffset: %d, value: %d
            //     \n", rank, blockIdx, targetRankId, targetServerId, dataFlagOffset,
            //         windowInstatusTensor_.GetValue(dataFlagOffset));
            //     AscendC::DumpTensor(windowInstatusTensor_[dataFlagOffset], 628, sizeof(int32_t));
            //     break;
            // }
        }
        windowInstatusTensor_(dataFlagOffset) = 0;
    }

    // 从RDMA收到的windowInTensor将数据拷贝到output
    __aicore__ inline void WindowInToOutput()
    {
        /*
        ----------------------------------------------------------------------------
        | STATUS_ENTRY_SIZE |    Data (datalen * sizeof(T))    | STATUS_ENTRY_SIZE |
        ----------------------------------------------------------------------------
        */
        if (blockIdx > 0) {
            return;
        }
        int32_t targetRankId = (1 - serverId) * localRankSize + localRank;  // 2个server的计算方式
        int64_t recvCount = this->len;
        uint64_t dataOffset = (targetRankId * queSize + STATUS_ENTRY_SIZE) / sizeof(T);
        CpGM2GMPingPong<T>(recvCount * sizeof(T), windowInTensor_[dataOffset],
                           recvDataOutputGt[targetRankId * this->len], COPYONLY);
        // AscendC::DumpTensor(windowInTensor_[dataOffset], 646, recvCount);
    }

    // 从recvData拆分数据并计算输出
    __aicore__ inline void SplitAndCalcData()
    {
        pipe.Reset();
        pipe.InitBuffer(tempBuf_, UB_ALIGN);                // 存放临时的立即数
        pipe.InitBuffer(tempBuf2_, 5000 * UB_ALIGN);        // MAX_BS <= 4096, 要能放下一个bs的数据
        pipe.InitBuffer(tempBuf3_, numExperts * UB_ALIGN);  // 要能放numExpert个数据

        if (blockIdx == 0) {
            // printflag("before BuildTokenUniquePerServerData\n");
            BuildTokenUniquePerServerData();
            // printflag("after BuildTokenUniquePerServerData\n");
        }
        if (blockIdx == 1) {
            // printflag("before BuildTokenSeverIdxData\n");
            BuildTokenSeverIdxData();
            // printflag("after BuildTokenSeverIdxData\n");
        }
        if (blockIdx == 2) {
            // printflag("before BuildCountOuterData\n");
            BuildCountOuterData();
            // printflag("after BuildCountOuterData\n");
        }
        if (blockIdx == 3) {
            // printflag("before BuildEpRankTokenCntAndSrcDstData\n");
            BuildEpRankTokenCntAndSrcDstData();
            // printflag("after BuildEpRankTokenCntAndSrcDstData\n");
        }
        if (blockIdx == 4) {
            // printflag("before BuildExpandIdxData\n");
            BuildExpandIdxData();
            // printflag("after BuildExpandIdxData\n");
        }
        if (blockIdx == 5) {
            // printflag("before BuildOffsetInnerData\n");
            BuildOffsetInnerData();
            // printflag("after BuildOffsetInnerData\n");
        }
    }

    __aicore__ inline void BuildTokenSeverIdxData()
    {
        // printflag("enter BuildTokenSeverIdxData\n");
        // 计算 tokenServerIdxOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen + tokenServerCntLen
        int32_t curRankDataOffset = rank * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS;
        // AscendC::DumpTensor(recvDataOutputGt[curRankDataOffset], 652, 16);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < serverNum; ++i) {
            int32_t recvOffset = curRankDataOffset + i * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数
            // PRINTF("[BuildTokenSeverIdxData] rank:%d, blockIdx:%d, recvOffset:%d\n", rank, blockIdx, recvOffset);

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);

            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            // AscendC::DumpTensor(tmpLt, 667, 16);

            int32_t tarOffset = i * MAX_BS;
            DataCopyPad(tokenServerIdxOutputGT_[tarOffset], tmpLt, copyParams);
            // AscendC::DumpTensor(tokenServerIdxOutputGT_[tarOffset], 671, 16);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        // AscendC::DumpTensor(tokenServerIdxOutputGT_, 677, 16);
    }

    __aicore__ inline void BuildExpandIdxData()
    {
        // printflag("enter BuildExpandIdxData\n");
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // 计算 expandIdxOutputGT_ , 对应于输入 tokenExpertIdx
        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen + tokenServerCntLen +
        // tokenServerIdxLen
        int32_t curRankDataOffset =
            rank * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum;
        // AscendC::DumpTensor(recvDataOutputGt[curRankDataOffset], 725, 16);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < numExperts; ++i) {
            int32_t recvOffset = curRankDataOffset + i * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数
            // PRINTF("[BuildExpandIdxData] rank:%d, blockIdx:%d, recvOffset:%d\n", rank, blockIdx, recvOffset);

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);

            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            // AscendC::DumpTensor(tmpLt, 740, 16);

            int32_t tarOffset = i * MAX_BS;
            DataCopyPad(expandIdxOutputGT_[tarOffset], tmpLt, copyParams);
            // AscendC::DumpTensor(expandIdxOutputGT_[tarOffset], 744, 16);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        // AscendC::DumpTensor(expandIdxOutputGT_, 750, 16);
    }

    __aicore__ inline void GetEpRankSumCnt(int32_t srcRank, LocalTensor<int32_t> &epTokenCntLt)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        SyncFunc<AscendC::HardEvent::S_MTE2>();

        int32_t epTokenCntOffset = srcRank * len;
        // AscendC::DumpTensor(recvDataOutputGt[epTokenCntOffset], 652, 16);
        DataCopyPad(epTokenCntLt, recvDataOutputGt[epTokenCntOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 假设epTokenCntGt为 [2,2,2,2] --> 起始前缀和 [0,2,4,6]
        int32_t preCnt = 0;
        int32_t curVal = 0;
        uint32_t localServerExpNum = numExperts / rankSize * localRankSize;
        for (int32_t i = 0; i < numExperts; ++i) {
            if (i % localServerExpNum == 0) {
                preCnt = 0;
            }
            curVal = epTokenCntLt(i);
            pipe_barrier(PIPE_ALL);
            epTokenCntLt(i) = preCnt;  // 设置为前一个元素的前缀和
            pipe_barrier(PIPE_ALL);
            preCnt += curVal;
        }
        // AscendC::DumpTensor(epTokenCntLt, 748, 16);
    }

    __aicore__ inline void BuildOffsetInnerData()
    {
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<int32_t> epTokenCntLt = tempBuf3_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // 计算 offsetInnerOutputGT_ , 为全局的 expandIdx
        // shape[num_rank * max_bs, expertNum]  value: inner_offset
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            // 1.获取本卡发给每个expert的token个数起始前缀和
            GetEpRankSumCnt(srcRank, epTokenCntLt);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            int32_t dataOffset =
                srcRank * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum;
            for (int tokId = 0; tokId < MAX_BS; ++tokId) {
                int32_t recvOffset = dataOffset + tokId * numExperts;  // 每次从recvdata中拷贝 numExperts 个数
                // PRINTF("[BuildOffsetInnerData] rank:%d, blockIdx:%d, recvOffset:%d\n", rank, blockIdx, recvOffset);

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                // 2.每个token发到expert的顺序, 当前token的expand_idx
                DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();

                // 3.求token在每个专家上的偏移
                // 遍历每个token的expand_idx, 如果不为-1,则加上 epTokenCntLt 中对应专家的值(对应列), 否则保持-1;
                for (int32_t expId = 0; expId < numExperts; ++expId) {
                    int val = tmpLt(expId);
                    if (val == -1) {
                        continue;
                    }
                    val += epTokenCntLt(expId);
                    pipe_barrier(PIPE_ALL);
                    tmpLt(expId) = val;
                    pipe_barrier(PIPE_ALL);
                }

                SyncFunc<AscendC::HardEvent::S_MTE3>();
                // AscendC::DumpTensor(tmpLt, 793, 16);

                int32_t tarOffset = (srcRank * MAX_BS * numExperts) + tokId * numExperts;
                DataCopyPad(offsetInnerOutputGT_[tarOffset], tmpLt, copyParams);
                // AscendC::DumpTensor(offsetInnerOutputGT_[tarOffset], 797, 16);

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    }

    __aicore__ inline void BuildCountOuterData()
    {
        // printflag("enter BuildCountOuterData\n");
        // 计算 countOuterOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen
        int32_t curRankDataOffset = rank * len + numExperts + serverNum + MAX_BS * serverNum;
        // PRINTF("[BuildCountOuterData] rank:%d, blockIdx:%d, curRankDataOffset:%d\n", rank, blockIdx,
        // curRankDataOffset);

        DataCopyPad(tmpLt, recvDataOutputGt[curRankDataOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        // AscendC::DumpTensor(tmpLt, 744, 16);

        DataCopyPad(countOuterOutputGT_, tmpLt, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        // AscendC::DumpTensor(countOuterOutputGT_, 749, 16);
    }

    __aicore__ inline void BuildTokenUniquePerServerData()
    {
        // printflag("enter BuildTokenUniquePerServerData\n");
        // 计算 tokensUniquePerServerOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        int32_t curRankDataOffset = rank * len + numExperts;  // offset + numTokensPerExpertLen
        // PRINTF("[BuildTokenUniquePerServerData] rank:%d, blockIdx:%d, curRankDataOffset:%d\n", rank, blockIdx,
        // curRankDataOffset);
        DataCopyPad(tmpLt, recvDataOutputGt[curRankDataOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        DataCopyPad(tokensUniquePerServerOutputGT_, tmpLt, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void BuildEpRankTokenCntAndSrcDstData()
    {
        // printflag("enter BuildEpRankTokenCntAndSrcDstData\n");
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();

        GlobalTensor<int32_t> gEpRankTokenCntGT_;
        gEpRankTokenCntGT_.SetGlobalBuffer(
            (__gm__ int32_t *)(tmpDataInput),
            gNumTokensPerExpertAlignLen / sizeof(int32_t));  // tmpDataInput地址用作临时存数
        // 计算 epRankTokenCntOutputGT_
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams1{false, 0, 0, 0};
        int32_t curRankDataOffset = rank * len;
        // AscendC::DumpTensor(recvDataOutputGt[curRankDataOffset], 652, 16);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < rankSize; ++i) {
            int32_t recvOffset = i * len;  // 每次从recvdata中拷贝 MAX_BS 个数
            // PRINTF("[BuildEpRankTokenCntAndSrcDstData1] rank:%d, blockIdx:%d, recvOffset:%d\n", rank, blockIdx,
            // recvOffset);

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams1, padParams1);

            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            // AscendC::DumpTensor(tmpLt, 667, 16);

            int32_t tarOffset = i * numExperts;
            DataCopyPad(gEpRankTokenCntGT_[tarOffset], tmpLt, copyParams1);
            // AscendC::DumpTensor(tokenServerIdxOutputGT_[tarOffset], 671, 16);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3

        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCopyExtParams copyParams2{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams2{false, 0, 0, 0};
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        // shape[rankSize, numExperts] --> shape[numExperts, rankSize]  value: cnt
        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            for (int curExp = 0; curExp < numExperts; ++curExp) {
                int32_t inOffset = srcRank * numExperts + curExp;  // 只拷贝一个值
                // PRINTF("[BuildEpRankTokenCntAndSrcDstData2] rank:%d, blockIdx:%d, srcRank:%d, curExp:%d, inOffset:%d
                // \n",
                //     rank, blockIdx, srcRank, curExp, inOffset);

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                DataCopyPad(tmpLt, gEpRankTokenCntGT_[inOffset], copyParams2, padParams2);

                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                // AscendC::DumpTensor(tmpLt, 667, 16);

                int32_t outOffset = curExp * rankSize + srcRank;
                DataCopyPad(epRankTokenCntOutputGT_[outOffset], tmpLt, copyParams2);
                // AscendC::DumpTensor(epRankTokenCntOutputGT_[tarOffset], 671, 16);

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();

        // 计算 localEpTokenCntOutputGT_ , shape[localExperts, rankSize]  value: sumCnt 前缀和
        int32_t localExpertNum = numExperts / rankSize;
        int32_t preCnt = 0;
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < localExpertNum; ++i) {
            for (int j = 0; j < rankSize; ++j) {
                int32_t inOffset = (rank * localExpertNum + i) * rankSize + j;  // 拷贝当前专家的1个值，对应不同rank来的

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                DataCopyPad(tmpLt, epRankTokenCntOutputGT_[inOffset], copyParams2, padParams2);

                SyncFunc<AscendC::HardEvent::MTE2_S>();
                int32_t cnt = tmpLt(0);
                pipe_barrier(PIPE_ALL);
                preCnt += cnt;
                tmpLt(0) = preCnt;
                pipe_barrier(PIPE_ALL);
                // PRINTF("[BuildEpRankTokenCntAndSrcDstData3] rank:%d, blockIdx:%d, i:%d, j:%d, inOffset:%d, cnt:%d,
                // preCnt:%d \n",
                //     rank, blockIdx, i, j, inOffset, cnt, preCnt);

                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                // AscendC::DumpTensor(tmpLt, 667, 16);

                int32_t outOffset = i * rankSize + j;
                DataCopyPad(localEpTokenCntOutputGT_[outOffset], tmpLt, copyParams2);
                // AscendC::DumpTensor(epRankTokenCntOutputGT_[tarOffset], 671, 16);

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();

        GlobalTensor<int32_t> gExpertMaxBsSrcGT_;
        gExpertMaxBsSrcGT_.SetGlobalBuffer(
            (__gm__ int32_t *)(tmpDataInput + (gNumTokensPerExpertAlignLen) / sizeof(int32_t)),
            gExpertMaxBsSrcOffsetAlignLen / sizeof(int32_t));  // sendDataInput地址用作临时存数
        // 计算 gExpertMaxBsSrcGT_
        DataCopyExtParams copyParams3{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams3{false, 0, 0, 0};

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < rankSize; ++i) {
            int32_t dataOffset = i * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum +
                                 MAX_BS * numExperts;
            for (int j = 0; j < numExperts; ++j) {
                int32_t recvOffset = dataOffset + j * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数
                // PRINTF("[BuildEpRankTokenCntAndSrcDstData1] rank:%d, blockIdx:%d, recvOffset:%d\n", rank, blockIdx,
                // recvOffset);

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams3, padParams3);

                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                // AscendC::DumpTensor(tmpLt, 667, 16);

                int32_t tarOffset = (i * numExperts * MAX_BS) + j * MAX_BS;
                DataCopyPad(gExpertMaxBsSrcGT_[tarOffset], tmpLt, copyParams3);
                // if (i == 0) {
                //     AscendC::DumpTensor(gExpertMaxBsSrcGT_[tarOffset], 1025, 16);
                // }

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();

        /** 计算 srcOffsetRankTokenIdxOutputGT_ / dstOffsetRankTokenIdxOutputGT_
         *   shape[num_expert, num_rank, max_bs]  value: src_offset/dst_offset <--- shape[num_rank, num_expert, max_bs]
         */
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        DataCopyExtParams copyParams4{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams4{false, 0, 0, 0};
        int32_t dstOffsetStart = 0;
        LocalTensor<int32_t> dstOffsetLt = tempBuf_.Get<int32_t>();  // 存立即数的buf

        for (int expId = 0; expId < numExperts; ++expId) {
            if (expId % localExpertNum == 0) {
                dstOffsetStart = 0;  // 每次所属rank递增后，计算desOffset的起始位置需要重置为0
            }
            for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
                DataCopyPad(tmpLt, epRankTokenCntOutputGT_[expId * rankSize + srcRank], copyParams4,
                            padParams4);  // 只拷贝一个数
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                int32_t validTokenCnt = tmpLt(0);
                pipe_barrier(PIPE_ALL);

                for (int tokId = 0; tokId < MAX_BS; ++tokId) {
                    event_t eventId = EVENT_ID0;
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                    SyncFunc<AscendC::HardEvent::S_MTE2>();  // 复用tmpLt，加一个同步
                    int32_t inIdx = srcRank * numExperts * MAX_BS + expId * MAX_BS + tokId;
                    DataCopyPad(tmpLt, gExpertMaxBsSrcGT_[inIdx], copyParams4, padParams4);  // 只拷贝一个数
                    SyncFunc<AscendC::HardEvent::MTE2_S>();
                    int32_t srcOffset = tmpLt(0);
                    pipe_barrier(PIPE_ALL);

                    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                    int32_t outIdx = expId * rankSize * MAX_BS + srcRank * MAX_BS + tokId;
                    DataCopyPad(srcOffsetRankTokenIdxOutputGT_[outIdx], tmpLt, copyParams4);
                    // AscendC::DumpTensor(tmpLt, 667, 16);

                    if (tokId < validTokenCnt) {
                        dstOffsetLt(0) = dstOffsetStart;
                        pipe_barrier(PIPE_ALL);
                        dstOffsetStart++;  // 有效token，写入当前rank的output目的偏移位置需要递增
                    } else {
                        dstOffsetLt(0) = -1;
                        pipe_barrier(PIPE_ALL);
                    }

                    // if (srcOffset == 4096) {
                    //     PRINTF("[BuildEpRankTokenCntAndSrcDstData4] rank:%d, blockIdx:%d, expId:%d, srcRank:%d,
                    //     tokId:%d, \
                    //         inIdx:%d, srcOffset:%d, validTokenCnt:%d, outIdx:%d, dstOffsetStart:%d\n",
                    //         rank, blockIdx, expId, srcRank, tokId, inIdx, srcOffset, validTokenCnt, outIdx,
                    //         dstOffsetStart);
                    // }

                    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

                    DataCopyPad(dstOffsetRankTokenIdxOutputGT_[outIdx], dstOffsetLt, copyParams4);
                    // AscendC::DumpTensor(dstOffsetRankTokenIdxOutputGT_[outIdx], 671, 16);

                    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
                }
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    GlobalTensor<T> sendDataInputGt;
    GlobalTensor<int> tokenPerExpertDataInputGt;
    GlobalTensor<int> tmpDataInputGt;
    GlobalTensor<T> sendDataOffsetOutputGt;
    GlobalTensor<T> recvDataOutputGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<T> remoteGt;

    __gm__ T *sendDataInput;
    __gm__ int *tokenPerExpertDataInput;
    __gm__ int *tmpDataInput;
    __gm__ T *sendDataOffsetOutput;
    __gm__ T *recvDataOutput;

    int64_t queLen;
    int64_t queSize;
    int64_t queElemLen;  // 共享内存队列里每个元素大小（以sizeof(T)计）

    int64_t coreNumBetween;  // 分层通信第一阶段，Server间通信使用的核数
    int64_t coreNumWithin;   // 分层通信第二阶段，Server内通信使用的核数
    int32_t rankNumPerCore;  // 每个核负责的rank数

    // RDMA相关变量
    int32_t serverNum;                    // Server数量
    int32_t serverId;                     // 本卡所属的server ID
    int32_t targetRank[MULTI_RANK_SIZE];  // 当前核心跨Server发送数据的目标rank Id，即数据最终的目标rank
    int32_t targetRankNum;  // 当前核心跨Server发送数据的目标rank Id的数量，小于等于MULTI_RANK_SIZE
    int64_t perRankDataNum;

    int rank;
    int rankSize;
    int localRank = 0;
    int localRankSize = 0;  // 在910A5中，表示一块板子上使用的卡数，在910B上表示单机内卡数。
    int xRankSize = 0;
    int yRankSize = 0;
    int xRankIdx = 0;
    int yRankIdx = 0;
    uint32_t extraFlag;
    int root;
    int sendPerGroup = 3;
    int topkNum;
    int64_t numExperts;
    int64_t numTokens;
    int64_t len;
    int64_t magic;
    int64_t blockIdx;  // 当前aicore序号
    int64_t blockNum;  // 当前rank的总aicore数
    int64_t timeout;
    GM_ADDR scale;
    GM_ADDR shareAddrs[CAM_MAX_RANK_SIZE];  // 共享内存地址列表
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};
    TPipe pipe;  // pipe工具类
    TBuf<QuePosition::VECCALC> tBuf;
    SyncCollectives sync;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GlobalTensor<int32_t> magicTensor_;  // 用于存放magic，位于windowInstatusTensor_之前
    GlobalTensor<uint32_t> batchWriteInfoTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;  // 用于rank间状态同步
    GlobalTensor<T> windowInTensor_;
    GlobalTensor<int32_t> windowOutstatusTensor_;  // 用于rank间状态同步
    GlobalTensor<T> windowOutTensor_;
    TBuf<> batchWriteInfoBuf_;  // 临时存放 batch write info
    TBuf<> tempBuf_;
    TBuf<> statusBuf_;
    LocalTensor<int32_t> statusTensor_;  // 临时存放statusFlag
    TBuf<> tokenPerExpertDataBuf;
    TBuf<> sendDataOffsetBuf;
    TBuf<> sendDataBuf;
    TBuf<> tempBuf2_;
    TBuf<> tempBuf3_;

    uint32_t sendDataAlignLen{0};
    uint32_t tokenPerExpertDataAlignLen{0};
    uint32_t sendDataOffsetAlignLen{0};

    uint32_t numTokensPerExpertAlignLen{0};   // 每个expert从本卡接收的token个数，对应一个rank的数据
    uint32_t gNumTokensPerExpertAlignLen{0};  // 全局，包含所有rank的
    uint32_t numTokensUniquePerServerAlignLen{0};  // 每个server从本卡接收的token个数(去重)，对应一个rank的
    uint32_t gNumTokensUniquePerServerAlignLen{0};  // 全局，包含所有rank的
    uint32_t numTokensPerServerAlignLen{0};  // 本卡每个token发到每个server的个数(不去重), 对应一个rank的
    uint32_t gNumTokensPerServerAlignLen{0};  // 全局，包含所有rank的
    uint32_t tokenServerCntAlignLen{0};       // 本卡每个token发给多少个server, 对应一个rank的
    uint32_t gTokenServerCntAlignLen{0};      // 全局，包含所有rank的
    uint32_t tokenServerIdxAlignLen{0};       // 本卡每个token发送给各个server的顺序, 对应一个rank的
    uint32_t gTokenServerIdxAlignLen{0};      // 全局，包含所有rank的
    uint32_t tokenExpertIdxAlignLen{0};       // 每个token发到expert的顺序, 对应一个rank的
    uint32_t gTokenExpertIdxAlignLen{0};      // 全局，包含所有rank的
    uint32_t expertMaxBsSrcOffsetAlignLen{0};  // 每个expert从本卡接收的token的server内offset, 对应一个rank的
    uint32_t gExpertMaxBsSrcOffsetAlignLen{0};  // 全局，包含所有rank的
    uint32_t expertMaxBsOriOffsetAlignLen{0};  // 每个expert从本卡接收的token在原卡上的origin_offset, 对应一个rank的
    uint32_t gExpertMaxBsOriOffsetAlignLen{0};  // 全局，包含所有rank的

    // GM_ADDR dataSpaceGT_;
    __gm__ int32_t *epRankTokenCntOutputGM_;
    GlobalTensor<int32_t>
        tokenServerIdxOutputGT_;  // token发送给对应server的token序号，-1表示没有，0-N表示序号 [bs, serverNum]
    GlobalTensor<int32_t>
        tokensUniquePerServerOutputGT_;  // 当前rank发送给对应server的token个数 [serverNum] -> value:count数量
    GlobalTensor<int32_t>
        epRankTokenCntOutputGT_;  // 每个专家、从rank接收的token数量 [expert_num, rank_num] -> value:token_cnt
    GlobalTensor<int32_t> localEpTokenCntOutputGT_;  // 本卡每个专家、从rank接收的token数量 [local_expert_num, rank_num]
    GlobalTensor<int32_t> srcOffsetRankTokenIdxOutputGT_;  // 每个专家、从rank接收的token源端偏移 [expert_num, rank_num,
                                                           // token_idx] -> value:src_offset
    GlobalTensor<int32_t> dstOffsetRankTokenIdxOutputGT_;  // 每个专家、从rank接收的token目的端偏移 [expert_num,
                                                           // rank_num, token_idx] -> value:dst_offset
    GlobalTensor<int32_t> countInnerOutputGT_;   // token给各个server发送个数    弃用
    GlobalTensor<int32_t> offsetInnerOutputGT_;  // 存放全局的expandIdx, [globalBs, expertNum]
    GlobalTensor<int32_t> countOuterOutputGT_;   // 每个token发送到的server数量 [bs] -> value:server数量
    GlobalTensor<int32_t> offsetOuterOutputGT_;  // 每个token在server上的位次    同tokenServerIdxOutputGT_
    GlobalTensor<int32_t>
        expandIdxOutputGT_;  // 给同一专家的token个数 [bs * numExperts], topk_idx的同专家前缀和扩维到所有专家维度
};

template <typename T>
template <typename F>
FORCE_INLINE_AICORE void NotifyDispatchA2<T>::SetAtomic(int op)
{
    PipeBarrier<PIPE_ALL>();
    if (op != -1) {
#ifdef __DAV_C220_VEC__
        SetAtomicOpType<F>(op);
#endif
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <HardEvent eventType>
FORCE_INLINE_AICORE void NotifyDispatchA2<T>::SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

template <typename T>
FORCE_INLINE_AICORE void NotifyDispatchA2<T>::UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <typename K, typename U>
FORCE_INLINE_AICORE void NotifyDispatchA2<T>::CpGM2GMPingPong(int64_t dataSizeRemain,
                                                              const GlobalTensor<U> &sendDataInputGt,
                                                              const GlobalTensor<K> &recvDataOutputGT, int op)
{
    // General case (U = K), input/output are the same, share one UB
    // Only when conversion is needed (U->K), UB will be divided into two parts according to the ratio of
    // sizeof(U):sizeof(K) and aligned to 32 bytes
    constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
    constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(K) + sizeof(U)) / UB_ALIGN_SIZE * UB_ALIGN_SIZE;
    constexpr int32_t inputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(U);
    constexpr int32_t outputUbBlockSize = std::is_same_v<K, U> ? ubBlockSize : ubAlignNum * sizeof(K);

    __gm__ U *input = const_cast<__gm__ U *>(sendDataInputGt.GetPhyAddr());
    __gm__ K *output = const_cast<__gm__ K *>(recvDataOutputGT.GetPhyAddr());
    __ubuf__ U *inputUB[2] = {(__ubuf__ U *)(UB_HEAD_OFFSET), (__ubuf__ U *)(UB_MID_OFFSET)};
    __ubuf__ K *outputUB[2] = {(__ubuf__ K *)inputUB[0], (__ubuf__ K *)inputUB[1]};
    if constexpr (!std::is_same_v<K, U>) {
        outputUB[0] = (__ubuf__ K *)(inputUB[0] + inputUbBlockSize / sizeof(U));
        outputUB[1] = (__ubuf__ K *)(inputUB[1] + inputUbBlockSize / sizeof(U));
    }
    int inputOffsetNum = 0;
    int outputOffsetNum = 0;
    if (dataSizeRemain <= 0) {
        return;
    }

    SetAtomic<K>(op);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2 waits for MTE3
    for (int64_t i = 0; dataSizeRemain > 0; i++) {
        // size and dataSizeRemain both refer to the output size
        uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
        event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(K) * sizeof(U));
        if constexpr (!std::is_same_v<K, U>) {
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl((i & 1) ? outputUB[0] : outputUB[1], (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE,
                     size / sizeof(K));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
        }
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUB2GM(output + outputOffsetNum, (i & 1) ? outputUB[0] : outputUB[1], size);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

        dataSizeRemain -= size;
        inputOffsetNum += (size / sizeof(K));
        outputOffsetNum += (size / sizeof(K));
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);  // MTE2 waits for MTE3

    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);  // Scalar waits for MTE3
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);

    UnsetAtomic(op);
    return;
}

#endif /* ALL2ALL_V_LAYERED_RDMA_H */
