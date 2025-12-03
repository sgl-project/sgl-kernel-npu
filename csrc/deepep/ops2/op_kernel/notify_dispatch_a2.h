#ifndef NOTIFY_DISPATCH_A2_H
#define NOTIFY_DISPATCH_A2_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "moe_distribute_base.h"
#include "notify_dispatch_tiling_a2.h"

using namespace AscendC;
using namespace Moe;

#define KERNELS_ARGS_FUN_A2_ALL2ALL()                                                                                  \
    GM_ADDR sendDataInput, GM_ADDR tokenPerExpertDataInput, GM_ADDR tmpDataInput, GM_ADDR sendDataOffsetOutput,        \
        GM_ADDR recvDataOutput, int64_t len, int64_t numTokens, int64_t topkNum, int64_t numExperts, int op, int root, \
        int cycleCount, GM_ADDR scale, int64_t scaleCount, GM_ADDR offset, int localRank, int localRankSize,           \
        GM_ADDR tokenServerIdxOutput, GM_ADDR tokensUniquePerServerOutput, GM_ADDR epRankTokenCntOutput,               \
        GM_ADDR localEpTokenCntOutput, GM_ADDR srcOffsetRankTokenIdxOutput, GM_ADDR dstOffsetRankTokenIdxOutput,       \
        GM_ADDR offsetInnerOutput, GM_ADDR countOuterOutput, GM_ADDR expandIdxOutput, GM_ADDR totalRecvTokensOutput,   \
        GM_ADDR workspace, GM_ADDR tiling

#define KERNELS_ARGS_CALL_A2_ALL2ALL()                                                                          \
    sendDataInput, tokenPerExpertDataInput, tmpDataInput, sendDataOffsetOutput, recvDataOutput, len, numTokens, \
        topkNum, numExperts, op, root, cycleCount, scale, scaleCount, offset, localRank, localRankSize,         \
        tokenServerIdxOutput, tokensUniquePerServerOutput, epRankTokenCntOutput, localEpTokenCntOutput,         \
        srcOffsetRankTokenIdxOutput, dstOffsetRankTokenIdxOutput, offsetInnerOutput, countOuterOutput,          \
        expandIdxOutput, totalRecvTokensOutput, workspace, tiling

// #define ENABLE_PRINT
#ifdef ENABLE_PRINT
#define CAM_PRINT(fmt, ...)                  \
    do {                                     \
        AscendC::printf(fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define CAM_PRINT(fmt, ...)
#endif

#define printflag(ss)                                                      \
    if (blockIdx < coreNumBetween) {                                       \
        CAM_PRINT("========rank:%d coreIdx:%d " #ss "\n", rank, blockIdx); \
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
    constexpr static uint32_t TEMP_BUF_LEN = 128 * 1024;  // tBuf注册长度为128K，剩余部分注册为其他buffer

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
    constexpr static uint64_t EXP_TOKEN_COUNT_FLAG_CNT = UB_ALIGN / sizeof(uint64_t);  // 4
    constexpr static uint32_t GM_ALIGN = 64;                                           // GM按64字节对齐

    constexpr static uint32_t MAX_BS = 4096;  // 每卡支持的最大bs
    // Synchronization flag occupies length
    constexpr static int64_t FLAG_UNIT_INT_NUM = 4;
    constexpr static int64_t MAGIC_MASK = ~((1LL << 32) - 1);

public:
    __aicore__ inline NotifyDispatchA2(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_A2_ALL2ALL())
    {
        InitAll2AllLayeredRdma(KERNELS_ARGS_CALL_A2_ALL2ALL());

        tokenPerExpertDataAlignLen = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataOffsetAlignLen = Ceil(numExperts * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataAlignLen = Ceil(len * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;  // 数据长度
        perRankDataNum = len;                                                     // 发送所有数据

        InitTensorLen();

        InitShare();
        // 初始化核分组, 需要外部调用保证所有的server的localRankSize均相同
        serverNum = CeilDiv(rankSize, localRankSize);
        serverId = rank / localRankSize;
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

        sendDataInputGt.SetGlobalBuffer((__gm__ T *)sendDataInput);
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        tmpDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tmpDataInput);
        sendDataOffsetOutputGt.SetGlobalBuffer((__gm__ T *)sendDataOffsetOutput);
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);

        gRankEpTokenCntGT_.SetGlobalBuffer(
            (__gm__ int32_t *)(tmpDataInput),
            gNumTokensPerExpertAlignLen / sizeof(int32_t));  // tmpDataInput地址用作临时存数
        gExpertMaxBsSrcGT_.SetGlobalBuffer(
            (__gm__ int32_t *)(tmpDataInput + gNumTokensPerExpertAlignLen),
            gExpertMaxBsSrcOffsetAlignLen / sizeof(int32_t));  // tmpDataInput地址用作临时存数

        tokenServerIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)tokenServerIdxOutput);
        tokensUniquePerServerOutputGT_.SetGlobalBuffer((__gm__ int32_t *)tokensUniquePerServerOutput);
        epRankTokenCntOutputGT_.SetGlobalBuffer((__gm__ int32_t *)epRankTokenCntOutput);
        localEpTokenCntOutputGT_.SetGlobalBuffer((__gm__ int64_t *)localEpTokenCntOutput);
        srcOffsetRankTokenIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)srcOffsetRankTokenIdxOutput);
        dstOffsetRankTokenIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)dstOffsetRankTokenIdxOutput);
        offsetInnerOutputGT_.SetGlobalBuffer((__gm__ int32_t *)offsetInnerOutput);
        countOuterOutputGT_.SetGlobalBuffer((__gm__ int32_t *)countOuterOutput);
        expandIdxOutputGT_.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutput);
        totalRecvTokensOutputGT_.SetGlobalBuffer((__gm__ int32_t *)totalRecvTokensOutput);

        // 初始化RDMA相关变量
        // dataSpaceGT_ = workspace; // 需要预留大一些空间供存放交换后拆分出来的数据
        windowInGM_ = this->shareAddrs[rank];
        windowOutGM_ =
            hccl_.GetWindowsOutAddr(rank) + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE + notifyMemoryOffset;
        batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t *)(workspace), rankSize * U32_PER_ITEM);
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

            // 第二阶段，处理server内通信
            ProcessWithinServer();
            SyncAll<true>();

            // 交换后的数据拆分和计算输出
            SplitAndCalcData();
            SyncAll<true>();

            hccl_.Finalize();
        }
    }

private:
    __aicore__ inline void InitAll2AllLayeredRdma(KERNELS_ARGS_FUN_A2_ALL2ALL())
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
        notifyMemoryOffset = winContext_[COMM_EP_IDX]->winSize - IPC_BUFF_MAX_SIZE * 2;
        // 设置并自增magic
        magicTensor_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsInAddr(rank) + IPC_DATA_OFFSET -
                                                         blockNum * sizeof(uint64_t) * EXP_TOKEN_COUNT_FLAG_CNT +
                                                         notifyMemoryOffset));

        pipe.InitBuffer(this->tBuf, TEMP_BUF_LEN);
        LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
        tempLocal(0) = 1;
        // 使用atomic方式实现+1
        AscendC::SetAtomicAdd<uint64_t>();
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
            this->shareAddrs[i] =
                hccl_.GetWindowsInAddr(i) + notifyMemoryOffset + (magic % PING_PONG_SIZE) * IPC_BUFF_MAX_SIZE;
        }
    }

    template <typename K, typename U = K>
    __aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U> &sendDataInputGt,
                                           const GlobalTensor<K> &recvDataOutputGT, int op);
    template <typename F>
    __aicore__ inline void SetAtomic(int op);
    __aicore__ inline void UnsetAtomic(int op);
    template <HardEvent eventType>
    __aicore__ inline void SetWaitEvent(event_t eventId);

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

    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (magic * 2ULL + value);
    }

    // Wait for a part of synchronization flags within a rank
    __aicore__ inline void WaitOneRankPartFlag(__gm__ uint64_t *waitAddr, int64_t flagNum, uint64_t checkValue)
    {
        GlobalTensor<uint64_t> globalWait;
        globalWait.SetGlobalBuffer(waitAddr, flagNum * FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localWait = tBuf.GetWithOffset<uint64_t>(flagNum * FLAG_UNIT_INT_NUM, 0);
        bool isSync = true;
        uint64_t checkedFlagNum = 0;
        do {
            // Copy global synchronization flags to local
            DataCopy(localWait, globalWait[checkedFlagNum * FLAG_UNIT_INT_NUM],
                     (flagNum - checkedFlagNum) * FLAG_UNIT_INT_NUM);
            SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

            // Check if the synchronization flags are equal to checkValue
            isSync = true;
            uint64_t remainToCheck = flagNum - checkedFlagNum;
            for (auto i = 0; i < remainToCheck; ++i) {
                // Continue waiting if any core has not reached the checkValue phase
                uint64_t v = localWait.GetValue(i * FLAG_UNIT_INT_NUM);
                if ((v & MAGIC_MASK) != (checkValue & MAGIC_MASK) || v < checkValue) {
                    isSync = false;
                    checkedFlagNum += i;
                    break;
                }
            }
        } while (!isSync);
    }

    __aicore__ inline void SetInnerFlag(uint64_t magic, uint64_t eventID, int64_t setRank, int64_t setBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        // SetFlag((__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM, value);
        __gm__ uint64_t *setAddr = (__gm__ uint64_t *)(shareAddrs[setRank]) + setBlock * FLAG_UNIT_INT_NUM;

        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
        SetWaitEvent<HardEvent::MTE2_S>(EVENT_ID0);
        GlobalTensor<uint64_t> globalSet;
        globalSet.SetGlobalBuffer(setAddr, FLAG_UNIT_INT_NUM);
        LocalTensor<uint64_t> localSet = tBuf.GetWithOffset<uint64_t>(1, 0);
        localSet.SetValue(0, value);

        // Copy global synchronization flag to local
        SetWaitEvent<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(globalSet, localSet, FLAG_UNIT_INT_NUM);
        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // Wait for a single inner-card synchronization flag
    __aicore__ inline void WaitInnerFlag(uint64_t magic, uint64_t eventID, int64_t waitRank, int64_t waitBlock)
    {
        uint64_t value = MergeMagicWithValue(magic, eventID);
        WaitOneRankPartFlag((__gm__ uint64_t *)(shareAddrs[waitRank]) + waitBlock * FLAG_UNIT_INT_NUM, 1, value);
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

        for (int i = 0; i < localRankSize; ++i) {
            int32_t curServerRankId = serverId * localRankSize + i;
            for (int j = 0; j < serverNum; ++j) {
                // 给当前server每卡写入serverNum个标记，位置为 rank + j * localRankSize
                int32_t offset = rank + j * rankSize;  // rank0: 0,16 / rank8: 8,24
                // rank0,server0: 0-7,16-23  rank8,server1: 8-15,24-31
                SetInnerFlag(magic, 1, curServerRankId, offset);
            }
        }
    }

    __aicore__ inline void ShareToShareSlice()
    {
        // 从Server内其他卡的共享内存(有serverNum块数据)对应位置拷贝数据到本卡的output
        uint32_t coreForDataBlock = (localRankSize * serverNum) / coreNumWithin;  // 8*2/16=1
        uint32_t remainDataBlock = (localRankSize * serverNum) % coreNumWithin;   // 8*2%16=0
        uint32_t startDataBlockId = coreForDataBlock * blockIdx;
        if (blockIdx < remainDataBlock) {
            startDataBlockId += blockIdx;
            coreForDataBlock += 1;
        } else {
            startDataBlockId += remainDataBlock;
        }
        uint32_t endDataBlockId = startDataBlockId + coreForDataBlock;
        if (coreForDataBlock == 0) {
            return;
        }

        // printflag("ShareToShareSlice\n");
        int64_t recvCount = this->len;
        for (int i = startDataBlockId; i < endDataBlockId; ++i) {
            int32_t targetRankId = serverId * localRankSize + (i / serverNum);  // 表示从当前server第rankId去读
            // 对应为targetRankId的其他server同号卡 id: 0-15
            int32_t serverTarRankId = (i % serverNum) * localRankSize + (i / serverNum);

            // server0: 0-7,16-23   server1: 8-15,24-31
            int32_t offset = (i / serverNum + serverId * localRankSize) + (i % serverNum) * rankSize;
            WaitInnerFlag(magic, 1, rank, offset);

            remoteGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[targetRankId] + IPC_DATA_OFFSET +
                                                  serverTarRankId * queSize +
                                                  STATUS_ENTRY_SIZE));  // 该rank上的第server块
            CpGM2GMPingPong<T>(recvCount * sizeof(T), remoteGt, recvDataOutputGt[serverTarRankId * this->len],
                               COPYONLY);
        }
    }

    __aicore__ inline void ProcessBetweenServer()
    {
        InputToWindowOut();
        ConstructBatchWriteInfo();
        SyncAll<true>();
        SendRdma();
        WaitRdma();
        SyncAll<true>();
        WindowInToOutput();
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

        windowOutstatusTensor_(statusFlagOffset / sizeof(int32_t)) = FLAG_VALUE;
        windowOutstatusTensor_(sourceRankIdOffset / sizeof(int32_t)) = rank;
        windowOutstatusTensor_(destRankIdOffset / sizeof(int32_t)) = targetRankId;
        windowOutstatusTensor_(dataLenOffset / sizeof(int32_t)) = (int32_t)datalen;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowOutstatusTensor_[(statusEntryOffset / sizeof(int32_t))]);
        windowOutstatusTensor_(dataFlagOffset / sizeof(int32_t)) = FLAG_VALUE;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowOutstatusTensor_[(dataFlagOffset / sizeof(int32_t))]);
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

        int64_t datalen = 0;
        int32_t statusFlag = 0;
        int32_t dataFlag = 0;
        while (statusFlag != FLAG_VALUE) {
            DataCopy(statusTensor_[0], windowInstatusTensor_[targetRankId * queSize / sizeof(int32_t)],
                     U32_STATUS_ENTRY);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            statusFlag = statusTensor_(FLAG_OFFSET / sizeof(int32_t));
            datalen = statusTensor_(DATALEN_OFFSET / sizeof(int32_t));
            PipeBarrier<PIPE_MTE2>();
        }

        uint64_t dataFlagOffset = (targetRankId * queSize + datalen * sizeof(T) + STATUS_ENTRY_SIZE) / sizeof(int32_t);
        while (dataFlag != FLAG_VALUE) {
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            dataFlag = dataFlagLocal(0);
            PipeBarrier<PIPE_MTE2>();
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
    }

    // 从recvData拆分数据并计算输出
    __aicore__ inline void SplitAndCalcData()
    {
        pipe.Reset();
        pipe.InitBuffer(tempBuf_, UB_ALIGN);  // 存放临时的立即数
        pipe.InitBuffer(tempBuf2_,
                        Ceil(MAX_BS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // MAX_BS <= 4096, 要能放下一个bs的数据
        pipe.InitBuffer(tempBuf3_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // 要能放numExpert个数据
        pipe.InitBuffer(tempBuf7_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // 要能放numExpert个数据
        pipe.InitBuffer(tempBuf8_,
                        Ceil(MAX_BS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // MAX_BS <= 4096, 要能放下一个bs的数据
        pipe.InitBuffer(tempBuf9_,
                        Ceil(MAX_BS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // MAX_BS <= 4096, 要能放下一个bs的数据
        pipe.InitBuffer(tempBuf10_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);  // 要能放numExpert个数据

        pipe.InitBuffer(tempBuf4_, 1000 * sizeof(float));  // 要能放localExp从所有rank接收token的数据
        pipe.InitBuffer(tempBuf5_, 1000 * sizeof(float));  // 存放中间临时数据
        pipe.InitBuffer(tempBuf6_, 1000 * sizeof(float));  // 存放中间临时数据

        pipe.InitBuffer(tempBuf11_, Ceil(1 * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);  // 存放中间临时数据

        GetRankEpTokenCntData(0, blockNum);
        GetExpertMaxBsSrcData(0, blockNum);
        SyncAll<true>();
        BuildEpRankTokenCntData(0, blockNum);
        SyncAll<true>();
        BuildLocalEpRankTokenCntData(0, blockNum);

        int32_t coreNumPerFunc = CeilDiv(static_cast<int32_t>(blockNum), 2);
        if (blockIdx < coreNumPerFunc) {
            if (blockIdx == 0) {
                BuildTokenUniquePerServerData();
                BuildTotalRecvTokensData();
            }
            if (blockIdx == 1) {
                BuildTokenSeverIdxData();
            }
            if (blockIdx == 2) {
                BuildCountOuterData();
            }
            if (blockIdx == 3) {
                BuildExpandIdxData();
            }
            if (blockIdx > 3) {
                int32_t beginCoreId = 4;
                int32_t remainCoreNum = coreNumPerFunc - 4;
                BuildOffsetInnerData(beginCoreId, remainCoreNum);
            }
        } else {
            int32_t beginCoreId = coreNumPerFunc;
            int32_t remainCoreNum = blockNum - coreNumPerFunc;
            BuildSrcDstOffsetData(beginCoreId, remainCoreNum);
        }
    }

    __aicore__ inline void BuildTokenSeverIdxData()
    {
        // 计算 tokenServerIdxOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<int32_t> dstLt = tempBuf9_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        LocalTensor<int32_t> fullOneLt = tempBuf8_.Get<int32_t>();
        Duplicate<int32_t>(fullOneLt, 1, MAX_BS);
        PipeBarrier<PIPE_V>();

        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen + tokenServerCntLen
        int32_t curRankDataOffset = rank * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS;

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < serverNum; ++i) {
            int32_t recvOffset = curRankDataOffset + i * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            Sub(dstLt, tmpLt, fullOneLt, MAX_BS);  // 所有偏移值-1，为-1的表示不发给该server
            PipeBarrier<PIPE_V>();

            SyncFunc<AscendC::HardEvent::V_MTE3>();

            int32_t tarOffset = i * MAX_BS;
            DataCopyPad(tokenServerIdxOutputGT_[tarOffset], dstLt, copyParams);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    }

    __aicore__ inline void BuildExpandIdxData()
    {
        // printflag("enter BuildExpandIdxData\n");
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<int32_t> dstLt = tempBuf9_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        LocalTensor<int32_t> fullOneLt = tempBuf8_.Get<int32_t>();
        Duplicate<int32_t>(fullOneLt, 1, MAX_BS);
        PipeBarrier<PIPE_V>();

        // 计算 expandIdxOutputGT_ , 对应于输入 tokenExpertIdx
        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen + tokenServerCntLen +
        // tokenServerIdxLen
        int32_t curRankDataOffset =
            rank * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum;
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = 0; i < numExperts; ++i) {
            int32_t recvOffset = curRankDataOffset + i * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);

            SyncFunc<AscendC::HardEvent::MTE2_V>();

            Sub(dstLt, tmpLt, fullOneLt, MAX_BS);  // 所有偏移值-1，为-1的表示不发给该server
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();

            int32_t tarOffset = i * MAX_BS;
            DataCopyPad(expandIdxOutputGT_[tarOffset], dstLt, copyParams);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    }

    __aicore__ inline void GetEpRankSumCnt(int32_t srcRank, LocalTensor<int32_t> &epTokenCntLt)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        SyncFunc<AscendC::HardEvent::S_MTE2>();

        int32_t epTokenCntOffset = srcRank * len;
        DataCopyPad(epTokenCntLt, recvDataOutputGt[epTokenCntOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // 假设epTokenCntGt为 [2,2,2,2] --> 起始前缀和, 跨server的专家要重新从0计数[0,2,0,2]
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
    }

    __aicore__ inline void BuildOffsetInnerForRank(int32_t targetRankId, int32_t index, uint32_t startTokenId,
                                                   uint32_t endTokenId)
    {
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<int32_t> tmpSumLt = tempBuf7_.Get<int32_t>();
        LocalTensor<int32_t> tmp2Lt = tempBuf10_.Get<int32_t>();
        LocalTensor<int32_t> maskLt = tempBuf3_.Get<int32_t>();

        LocalTensor<int32_t> fullOneLt = tempBuf9_.Get<int32_t>();
        Duplicate<int32_t>(fullOneLt, 1, numExperts);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> epTokenCntLt = tempBuf8_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // 1.获取本卡发给每个expert的token个数起始前缀和
        GetEpRankSumCnt(targetRankId, epTokenCntLt);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        int32_t dataOffset =
            targetRankId * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum;
        for (int tokId = startTokenId; tokId < endTokenId; ++tokId) {
            int32_t recvOffset = dataOffset + tokId * numExperts;  // 每次从recvdata中拷贝 numExperts 个数

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            // 2.每个token发到expert的顺序, 当前token的expand_idx, 假设 [2,2,0,0]
            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);

            // 3.求token在每个专家上的偏移
            // 当前server上专家token的前缀和与每个token的expand_idx进行相加，为负数的表示没有发给该专家
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            // 通过与1比较, 获取0/1的tensor, 用于作为被乘数 [1,1,0,0]
            Mins(maskLt, tmpLt, 1, numExperts);
            // 所有偏移值-1，为-1的表示不发给该专家
            Sub(tmp2Lt, tmpLt, fullOneLt, numExperts);
            PipeBarrier<PIPE_V>();

            Mul(tmpLt, epTokenCntLt, maskLt, numExperts);  // 相乘后掩盖掉不发送的专家
            PipeBarrier<PIPE_V>();

            Add(tmpSumLt, tmp2Lt, tmpLt, numExperts);
            PipeBarrier<PIPE_V>();

            SyncFunc<AscendC::HardEvent::V_MTE3>();

            int32_t tarOffset = index * MAX_BS * numExperts + tokId * numExperts;
            DataCopyPad(offsetInnerOutputGT_[tarOffset], tmpSumLt, copyParams);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    }

    __aicore__ inline void BuildOffsetInnerData(int32_t beginCoreId, int32_t validCoreNum)
    {
        // 分核处理token，2 server
        int32_t vBlockIdx = blockIdx - beginCoreId;     // 相对当前函数处理的 blockIdx
        uint32_t coreForToken = MAX_BS / validCoreNum;  // 4096 / 20 = 204
        uint32_t remainToken = MAX_BS % validCoreNum;   // 4096 % 20 = 16
        uint32_t startTokenId = coreForToken * vBlockIdx;
        if (vBlockIdx < remainToken) {
            startTokenId += vBlockIdx;
            coreForToken += 1;
        } else {
            startTokenId += remainToken;
        }
        uint32_t endTokenId = startTokenId + coreForToken;
        if (coreForToken == 0) {
            return;
        }

        // 计算对端rank，需构造对端rank的offsetInner数据（2个server的计算方式）
        int32_t curRankId = rank;
        int32_t peerRankId = (1 - serverId) * localRankSize + localRank;         // 2个server的计算方式
        int32_t firstRankId = curRankId < peerRankId ? curRankId : peerRankId;   // 取小的rank
        int32_t secondRankId = curRankId < peerRankId ? peerRankId : curRankId;  // 取大的rank

        // 计算 offsetInnerOutputGT_ (包含本端rank和对端rank的offsetInner信息)
        // shape[max_bs, expertNum]  value: inner_offset
        BuildOffsetInnerForRank(firstRankId, 0, startTokenId, endTokenId);   // 先处理rankId小的
        BuildOffsetInnerForRank(secondRankId, 1, startTokenId, endTokenId);  // 再处理rankId大的
    }

    __aicore__ inline void BuildCountOuterData()
    {
        // 计算 countOuterOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        // offset + numTokensPerExpertLen + numTokensUniquePerServerLen + numTokensPerServerLen
        int32_t curRankDataOffset = rank * len + numExperts + serverNum + MAX_BS * serverNum;

        DataCopyPad(tmpLt, recvDataOutputGt[curRankDataOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        DataCopyPad(countOuterOutputGT_, tmpLt, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void BuildTokenUniquePerServerData()
    {
        // 计算 tokensUniquePerServerOutputGT_
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(serverNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        int32_t curRankDataOffset = rank * len + numExperts;  // offset + numTokensPerExpertLen
        DataCopyPad(tmpLt, recvDataOutputGt[curRankDataOffset], copyParams, padParams);

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

        DataCopyPad(tokensUniquePerServerOutputGT_, tmpLt, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void GetRankEpTokenCntData(int32_t beginCoreId, int32_t validCoreNum)
    {
        // 分核处理，2 server，每个核处理一个rank
        int32_t vBlockIdx = blockIdx - beginCoreId;  // 相对当前函数处理的 blockIdx
        uint32_t coreForRank = rankSize / validCoreNum;
        uint32_t remainRank = rankSize % validCoreNum;
        uint32_t startRankId = coreForRank * vBlockIdx;
        if (vBlockIdx < remainRank) {
            startRankId += vBlockIdx;
            coreForRank += 1;
        } else {
            startRankId += remainRank;
        }
        uint32_t endRankId = startRankId + coreForRank;
        if (coreForRank == 0) {
            return;
        }

        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        // 获取 gRankEpTokenCntGT_
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams1{false, 0, 0, 0};
        int32_t curRankDataOffset = rank * len;

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = startRankId; i < endRankId; ++i) {
            int32_t recvOffset = i * len;  // 每次从recvdata中拷贝 numExperts 个数

            event_t eventId = EVENT_ID0;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

            DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams1, padParams1);

            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);

            int32_t tarOffset = i * numExperts;
            DataCopyPad(gRankEpTokenCntGT_[tarOffset], tmpLt, copyParams1);

            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void GetExpertMaxBsSrcData(int32_t beginCoreId, int32_t validCoreNum)
    {
        // 分核处理，2 server，每个核处理一个rank
        int32_t vBlockIdx = blockIdx - beginCoreId;  // 相对当前函数处理的 blockIdx
        uint32_t coreForRank = rankSize / validCoreNum;
        uint32_t remainRank = rankSize % validCoreNum;
        uint32_t startRankId = coreForRank * vBlockIdx;
        if (vBlockIdx < remainRank) {
            startRankId += vBlockIdx;
            coreForRank += 1;
        } else {
            startRankId += remainRank;
        }
        uint32_t endRankId = startRankId + coreForRank;
        if (coreForRank == 0) {
            return;
        }

        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(MAX_BS * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = startRankId; i < endRankId; ++i) {
            int32_t dataOffset = i * len + numExperts + serverNum + MAX_BS * serverNum + MAX_BS + MAX_BS * serverNum +
                                 MAX_BS * numExperts;
            for (int j = 0; j < numExperts; ++j) {
                int32_t recvOffset = dataOffset + j * MAX_BS;  // 每次从recvdata中拷贝 MAX_BS 个数

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                DataCopyPad(tmpLt, recvDataOutputGt[recvOffset], copyParams, padParams);

                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);

                int32_t tarOffset = (i * numExperts * MAX_BS) + j * MAX_BS;
                DataCopyPad(gExpertMaxBsSrcGT_[tarOffset], tmpLt, copyParams);

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void BuildEpRankTokenCntData(int32_t beginCoreId, int32_t validCoreNum)
    {
        // 分核处理，2 server，每个核处理一个rank
        int32_t vBlockIdx = blockIdx - beginCoreId;  // 相对当前函数处理的 blockIdx
        uint32_t coreForRank = rankSize / validCoreNum;
        uint32_t remainRank = rankSize % validCoreNum;
        uint32_t startRankId = coreForRank * vBlockIdx;
        if (vBlockIdx < remainRank) {
            startRankId += vBlockIdx;
            coreForRank += 1;
        } else {
            startRankId += remainRank;
        }
        uint32_t endRankId = startRankId + coreForRank;
        if (coreForRank == 0) {
            return;
        }

        SyncFunc<AscendC::HardEvent::MTE3_S>();
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        // shape[rankSize, numExperts] --> shape[numExperts, rankSize]  value: cnt
        for (int srcRank = startRankId; srcRank < endRankId; ++srcRank) {
            for (int curExp = 0; curExp < numExperts; ++curExp) {
                int32_t inOffset = srcRank * numExperts + curExp;  // 只拷贝一个值

                event_t eventId = EVENT_ID0;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                DataCopyPad(tmpLt, gRankEpTokenCntGT_[inOffset], copyParams, padParams);

                AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);

                int32_t outOffset = curExp * rankSize + srcRank;
                DataCopyPad(epRankTokenCntOutputGT_[outOffset], tmpLt, copyParams);

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void BuildTotalRecvTokensData()
    {
        // 单核计算
        LocalTensor<int32_t> totalCnt = tempBuf_.Get<int32_t>();
        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<float> floatTmpLt = tempBuf4_.Get<float>();
        LocalTensor<float> floatTmpSumLt = tempBuf5_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tempBuf6_.Get<float>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rankSize * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        int32_t localExpertNum = numExperts / rankSize;
        int32_t sumVal = 0;
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int index = 0; index < localExpertNum; ++index) {
            int expId = rank * localExpertNum + index;
            DataCopyPad(tmpLt, epRankTokenCntOutputGT_[expId * rankSize], copyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(floatTmpLt, tmpLt, RoundMode::CAST_NONE, rankSize);
            PipeBarrier<PIPE_V>();
            ReduceSum(floatTmpSumLt, floatTmpLt, sharedTmpBuffer, rankSize);
            SyncFunc<AscendC::HardEvent::V_S>();
            // 加上该专家接收的token数
            sumVal += static_cast<int32_t>(floatTmpSumLt.GetValue(0));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3

        totalCnt(0) = sumVal;
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(totalRecvTokensOutputGT_, totalCnt, copyParams1);
    }

    __aicore__ inline void BuildLocalEpRankTokenCntData(int32_t beginCoreId, int32_t validCoreNum)
    {
        // 计算 localEpTokenCntOutputGT_ , shape[localExperts]  value: tokenCnt 非前缀和
        int32_t localExpertNum = numExperts / rankSize;
        int32_t vBlockIdx = blockIdx - beginCoreId;  // 相对当前函数处理的 blockIdx
        uint32_t coreForExp = localExpertNum / validCoreNum;
        uint32_t remainExp = localExpertNum % validCoreNum;
        uint32_t startExpId = coreForExp * vBlockIdx;
        if (vBlockIdx < remainExp) {
            startExpId += vBlockIdx;
            coreForExp += 1;
        } else {
            startExpId += remainExp;
        }
        uint32_t endExpId = startExpId + coreForExp;
        if (coreForExp == 0) {
            return;
        }

        LocalTensor<int64_t> tmpEpRecvLt = tempBuf11_.Get<int64_t>();
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(1 * sizeof(int64_t)), 0, 0, 0};

        LocalTensor<int32_t> tmpLt = tempBuf2_.Get<int32_t>();
        LocalTensor<float> floatTmpLt = tempBuf4_.Get<float>();
        LocalTensor<float> floatTmpSumLt = tempBuf5_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tempBuf6_.Get<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(rankSize * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
        for (int i = startExpId; i < endExpId; ++i) {
            int expId = rank * localExpertNum + i;
            DataCopyPad(tmpLt, epRankTokenCntOutputGT_[expId * rankSize], copyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(floatTmpLt, tmpLt, RoundMode::CAST_NONE, rankSize);
            PipeBarrier<PIPE_V>();
            ReduceSum(floatTmpSumLt, floatTmpLt, sharedTmpBuffer, rankSize);
            SyncFunc<AscendC::HardEvent::V_S>();
            // 该专家接收的token数
            int64_t recvCnt = static_cast<int64_t>(floatTmpSumLt.GetValue(0));

            tmpEpRecvLt(0) = recvCnt;
            pipe_barrier(PIPE_ALL);
            DataCopyPad(localEpTokenCntOutputGT_[i], tmpEpRecvLt, copyParams1);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);  // MTE2 waits for MTE3
    }

    __aicore__ inline void BuildSrcDstOffsetData(int32_t beginCoreId, int32_t validCoreNum)
    {
        int32_t localExpertNum = numExperts / rankSize;
        uint32_t curRankExpertStart = rank * localExpertNum;
        uint32_t curRankExpertEnd = curRankExpertStart + localExpertNum;

        // 对当前卡上的专家进行分核处理
        int32_t vBlockIdx = blockIdx - beginCoreId;  // 相对当前函数处理的 blockIdx
        uint32_t coreForExp = localExpertNum / validCoreNum;
        uint32_t remainExp = localExpertNum % validCoreNum;
        uint32_t startExpId = coreForExp * vBlockIdx + curRankExpertStart;
        if (vBlockIdx < remainExp) {
            startExpId += vBlockIdx;
            coreForExp += 1;
        } else {
            startExpId += remainExp;
        }
        uint32_t endExpId = startExpId + coreForExp;
        if (coreForExp == 0) {
            return;
        }

        /** 计算 srcOffsetRankTokenIdxOutputGT_ / dstOffsetRankTokenIdxOutputGT_
         *   shape[local_exp_num, num_rank, max_bs]  value: src_offset/dst_offset <--- shape[num_rank, num_expert,
         * max_bs]
         */
        LocalTensor<int32_t> expTokenCntLt = tempBuf2_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tempBuf4_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = tempBuf5_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tempBuf6_.Get<float>();

        LocalTensor<int32_t> tmpLt = tempBuf3_.Get<int32_t>();
        LocalTensor<int32_t> dstOffsetLt = tempBuf_.Get<int32_t>();  // 存立即数的buf
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);           // MTE2 waits for MTE3
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        for (int expId = startExpId; expId < endExpId; ++expId) {  // 全局专家id
            int32_t localExpId = expId - curRankExpertStart;       // 本地专家id

            int32_t dstOffsetStart = 0;  // 因为只处理当前rank的本地专家，dstOffset递增
            if (localExpId != 0) {
                // 从epRankTokenCntOutputGT_拷贝本卡当前专家前的所有专家接收token数
                int32_t copyCnt = localExpId * rankSize;
                DataCopyExtParams copyParams1{1, static_cast<uint32_t>(localExpId * rankSize * sizeof(int32_t)), 0, 0,
                                              0};
                DataCopyPadExtParams<int32_t> padParams1{false, 0, 0, 0};

                DataCopyPad(expTokenCntLt, epRankTokenCntOutputGT_[curRankExpertStart * rankSize], copyParams1,
                            padParams1);
                SyncFunc<AscendC::HardEvent::MTE2_V>();
                Cast(floatExpTokenCntLt, expTokenCntLt, RoundMode::CAST_NONE, copyCnt);
                PipeBarrier<PIPE_V>();
                ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, copyCnt);
                SyncFunc<AscendC::HardEvent::V_S>();
                // 当前专家的起始偏移，为上一个本地专家收的token总数
                dstOffsetStart = static_cast<int32_t>(floatExpTokenSumCntLt.GetValue(0));
            }

            for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
                DataCopyPad(tmpLt, epRankTokenCntOutputGT_[expId * rankSize + srcRank], copyParams,
                            padParams);  // 只拷贝一个数
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                int32_t validTokenCnt = tmpLt(0);
                pipe_barrier(PIPE_ALL);

                for (int tokId = 0; tokId < validTokenCnt; ++tokId) {
                    event_t eventId = EVENT_ID0;
                    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);

                    SyncFunc<AscendC::HardEvent::S_MTE2>();  // 复用tmpLt，加一个同步
                    int32_t inIdx = srcRank * numExperts * MAX_BS + expId * MAX_BS + tokId;
                    DataCopyPad(tmpLt, gExpertMaxBsSrcGT_[inIdx], copyParams, padParams);  // 只拷贝一个数
                    SyncFunc<AscendC::HardEvent::MTE2_S>();
                    int32_t srcOffsetVal = tmpLt(0) - 1;  // 给srcOffset-1，将偏移值从0开始
                    tmpLt(0) = srcOffsetVal;
                    pipe_barrier(PIPE_ALL);

                    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                    int32_t outIdx = expId * rankSize * MAX_BS + srcRank * MAX_BS + tokId;
                    DataCopyPad(srcOffsetRankTokenIdxOutputGT_[outIdx], tmpLt, copyParams);

                    if (tokId < validTokenCnt) {
                        dstOffsetLt(0) = dstOffsetStart;
                        pipe_barrier(PIPE_ALL);
                        dstOffsetStart++;  // 有效token，写入当前rank的output目的偏移位置需要递增
                    } else {
                        dstOffsetLt(0) = -1;
                        pipe_barrier(PIPE_ALL);
                    }

                    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();

                    DataCopyPad(dstOffsetRankTokenIdxOutputGT_[outIdx], dstOffsetLt, copyParams);

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
    uint64_t magic;
    int64_t blockIdx;  // 当前aicore序号
    int64_t blockNum;  // 当前rank的总aicore数
    int64_t timeout;
    GM_ADDR scale;
    GM_ADDR shareAddrs[CAM_MAX_RANK_SIZE];  // 共享内存地址列表
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};
    TPipe pipe;  // pipe工具类
    TBuf<QuePosition::VECCALC> tBuf;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GlobalTensor<uint64_t> magicTensor_;  // 用于存放magic，位于windowInstatusTensor_之前
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
    TBuf<> tempBuf4_;
    TBuf<> tempBuf5_;
    TBuf<> tempBuf6_;
    TBuf<> tempBuf7_;
    TBuf<> tempBuf8_;
    TBuf<> tempBuf9_;
    TBuf<> tempBuf10_;
    TBuf<> tempBuf11_;

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
    uint32_t notifyMemoryOffset{0};

    GlobalTensor<int32_t> gRankEpTokenCntGT_;  // 临时数据
    GlobalTensor<int32_t> gExpertMaxBsSrcGT_;  // 临时数据

    GlobalTensor<int32_t>
        tokenServerIdxOutputGT_;  // token发送给对应server的token序号，-1表示没有，0-N表示序号 [bs, serverNum]
    GlobalTensor<int32_t>
        tokensUniquePerServerOutputGT_;  // 当前rank发送给对应server的token个数 [serverNum] -> value:count数量
    GlobalTensor<int32_t>
        epRankTokenCntOutputGT_;  // 每个专家、从rank接收的token数量 [expert_num, rank_num] -> value:token_cnt
    GlobalTensor<int64_t> localEpTokenCntOutputGT_;  // 本卡每个专家接收的token数量 [local_expert_num]
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
    GlobalTensor<int32_t> totalRecvTokensOutputGT_;  // 本卡收到的token总数, [1] -> value:count
};

template <typename T>
template <typename F>
__aicore__ inline void NotifyDispatchA2<T>::SetAtomic(int op)
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
__aicore__ inline void NotifyDispatchA2<T>::SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

template <typename T>
__aicore__ inline void NotifyDispatchA2<T>::UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <typename K, typename U>
__aicore__ inline void NotifyDispatchA2<T>::CpGM2GMPingPong(int64_t dataSizeRemain,
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

#endif /* NOTIFY_DISPATCH_A2_H */
