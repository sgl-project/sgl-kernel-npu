#ifndef NOTIFY_DISPATCH_H
#define NOTIFY_DISPATCH_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "moe_distribute_base.h"

using namespace AscendC;
using namespace Moe;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define KERNELS_ARGS_FUN_ALL2ALL()                                                                                  \
    GM_ADDR sendDataInput, GM_ADDR tokenPerExpertDataInput, GM_ADDR sendDataOffsetOutput, GM_ADDR recvDataOutput,   \
        GM_ADDR recvCount, GM_ADDR recvOffset, GM_ADDR expertGlobalOffset, GM_ADDR srcrankInExpertOffset,           \
        GM_ADDR rInSrcrankOffset, GM_ADDR totalRecvTokens, GM_ADDR maxBs, GM_ADDR recvTokensPerExpert, int64_t len, \
        int32_t round, int32_t perRoundTokens, int32_t numTokens, int op, int root, int cycleCount, GM_ADDR scale,  \
        int32_t scaleCount, GM_ADDR offset, int localRank, int localRankSize, uint64_t totalWinSize

#define KERNELS_ARGS_CALL_ALL2ALL()                                                                                    \
    sendDataInput, tokenPerExpertDataInput, sendDataOffsetOutput, recvDataOutput, recvCount, recvOffset,               \
        expertGlobalOffset, srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert, len, \
        round, perRoundTokens, numTokens, op, root, cycleCount, scale, scaleCount, offset, localRank, localRankSize,   \
        totalWinSize

template <typename T>
class NotifyDispatch
{
    constexpr static int32_t MAX_RANK_PER_CORE = 8;
    constexpr static int32_t MULTI_RANK_SIZE = 48;
    constexpr static int32_t MAX_BUFFER_NUMBER = 10;
    constexpr static uint32_t UB_FLAG_SIZE = 8U * 1024U;

    constexpr static int32_t TOTAL_CNT_CORE = 0;
    constexpr static int32_t RECV_COUNT_CORE = 1;
    constexpr static int32_t RECV_OFFSET_CORE = 2;
    constexpr static int32_t MAX_BS_CORE = 3;
    constexpr static int32_t RECV_TOKEN_PER_EXP_CORE = 4;
    constexpr static int32_t EXP_GLOBAL_OFFSET_CORE = 5;
    constexpr static int32_t SRC_RANK_EXP_OFFSET_CORE = 6;
    constexpr static int32_t R_IN_SRCRANK_OFFSET_CORE = 7;
    // Synchronization flag occupies length
    constexpr static int64_t FLAG_UNIT_INT_NUM = 4;
    constexpr static int64_t MAGIC_MASK = ~((1LL << 32) - 1);

public:
    __aicore__ inline NotifyDispatch(int rank, int rankSize, uint32_t extraFlag)
        : rank(rank), rankSize(rankSize), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_ALL2ALL())
    {
        InitSmallFullMesh(KERNELS_ARGS_CALL_ALL2ALL());
        nodeNum = rankSize / localRankSize;
        localRankId = rank % localRankSize;
        localNodeId = rank / localRankSize;
        perNodeDataNum = GetDataCount(len, nodeNum);   // 128K/4 = 32K
        perRankDataNum = GetDataCount(len, rankSize);  // 128K/64 = 2K
        totalRecvTokens_ = totalRecvTokens;
        recvCount_ = recvCount;
        recvOffset_ = recvOffset;
        maxBs_ = maxBs;
        recvTokensPerExpert_ = recvTokensPerExpert;
        tokenPerExpertDataAlignLen = Ceil(round * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataOffsetAlignLen = Ceil(round * numExperts * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataAlignLen = Ceil(round * numExperts * sendPerGroup * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        recvDataAlignLen = Ceil(round * numExperts * sendPerGroup * sizeof(int32_t), UB_ALIGN_SIZE) *
                           UB_ALIGN_SIZE;  // 32 * 256 * 3 * 4 = 96KB
        sendTokensPerRankAlignLen = Ceil(numRanks * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendCountAlignLen =
            Ceil(round * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;  // 32 * 256 * 4 = 32KB

        // Initialize core grouping
        InitCoreGroup();
        // Initialize data slicing
        InitDataSlice();

        totalRecvTokens_ = totalRecvTokens;
        recvCount_ = recvCount;
        recvOffset_ = recvOffset;
        maxBs_ = maxBs;
        recvTokensPerExpert_ = recvTokensPerExpert;
        expertGlobalOffset_ = expertGlobalOffset;
        srcrankInExpertOffset_ = srcrankInExpertOffset;
        rInSrcrankOffset_ = rInSrcrankOffset;
        this->sendDataInput = (__gm__ T *)sendDataInput;
        this->tokenPerExpertDataInput = (__gm__ int32_t *)tokenPerExpertDataInput;
        this->sendDataOffsetOutput = (__gm__ T *)sendDataOffsetOutput;
        this->recvDataOutput = (__gm__ T *)recvDataOutput;
        sendDataInputGt.SetGlobalBuffer((__gm__ T *)sendDataInput);
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        sendDataOffsetOutputGt.SetGlobalBuffer((__gm__ T *)sendDataOffsetOutput);
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);
        recvDataOutGt.SetGlobalBuffer((__gm__ int32_t *)recvDataOutput);
        pipe.InitBuffer(sendCountBuf, tokenPerExpertDataAlignLen);
        pipe.InitBuffer(sendOffsetBuf, tokenPerExpertDataAlignLen);
        pipe.InitBuffer(recvDataBuf, recvDataAlignLen);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx < 1) {
            AssembleSendData();
        }
        SyncAll<true>();
        if (blockIdx < coreNumPerStageX) {
            InputToShareSlice();
        }
        if (blockIdx < coreNumPerStageY) {
            ShareToShareSlice();
        }
        SyncAll<true>();
        ReorderOutput();
        BuildTotalRecvTokens();  // 出错点
        BuildRecvCount();
        BuildRecvOffset();
        BuildMaxBs();
        BuildRecvTokenPerExp();
        BuildExpGlobalOffset();
        BuildsrcRankInExpOffset();
        BuildRInSrcrankOffset();
    }

private:
    __aicore__ inline void InitCoreGroup()
    {
        coreNumPerStageY = blockNum;
        coreNumPerStageX = blockNum;
        rankNumPerCore = (rankSize + blockNum - 1) / blockNum;
    }

    __aicore__ inline void InitDataSlice()
    {
        // The producer is responsible for moving the input data of this rank to shared memory, input-->share
        if (blockIdx < coreNumPerStageX) {
            // The ipcQue responsible for the current core
            writeGt.SetGlobalBuffer((__gm__ T *)(shareAddrs[rank] + IPC_DATA_OFFSET));
        }
    }

    __aicore__ inline void AssembleSendData()
    {
        pipe.Reset();
        pipe.InitBuffer(tokenPerExpertDataBuf, tokenPerExpertDataAlignLen);
        pipe.InitBuffer(sendDataBuf, sendDataAlignLen);
        pipe.InitBuffer(sendDataOffsetBuf, sendDataOffsetAlignLen);
        int batchRounds = 32;
        int localExpertsNum = numExperts / rankSize;
        int newSendDataAlignLen =
            Ceil(batchRounds * localExpertsNum * sendPerGroup * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        pipe.InitBuffer(newSendDataBuf, newSendDataAlignLen);

        tokenPerExpertTensor = tokenPerExpertDataBuf.Get<int32_t>();
        sendDataTensor = sendDataBuf.Get<T>();
        sendDataOffsetTensor = sendDataOffsetBuf.Get<T>();
        newSendDataTensor = newSendDataBuf.Get<T>();
        DataCopyExtParams tokenPerExpertParams = {1U, tokenPerExpertDataAlignLen, 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(tokenPerExpertTensor, tokenPerExpertDataInputGt, tokenPerExpertParams, copyPadExtParams);

        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        int realRound = (numTokens + perRoundTokens - 1) / perRoundTokens;
        int lastRoundNumTokens = numTokens % perRoundTokens;
        if (lastRoundNumTokens == 0 && numTokens > 0) {
            lastRoundNumTokens = perRoundTokens;
        }

        int prefixSum = 0;

        for (int r = 0; r < realRound; ++r) {
            prefixSum = 0;
            for (int i = 0; i < numExperts; ++i) {
                int numTokensExpert = tokenPerExpertTensor(r * numExperts + i);
                int baseUB = r * numExperts * sendPerGroup + i * sendPerGroup;
                sendDataTensor(baseUB) = numTokensExpert;
                sendDataTensor(baseUB + 1) = prefixSum;
                int roundNumTokens = (r == realRound - 1 ? lastRoundNumTokens : perRoundTokens);
                sendDataTensor(baseUB + 2) = roundNumTokens;
                sendDataOffsetTensor(r * numExperts + i) = prefixSum;
                prefixSum += numTokensExpert;
            }
        }

        for (int r = realRound; r < round; ++r) {
            for (int i = 0; i < numExperts; ++i) {
                int baseUB = r * numExperts * sendPerGroup + i * sendPerGroup;
                sendDataTensor(baseUB) = 0;
                sendDataTensor(baseUB + 1) = 0;
                sendDataTensor(baseUB + 2) = 0;
                sendDataOffsetTensor(r * numExperts + i) = 0;
            }
        }

        int totalRounds = round;
        if (round > 1) {
            for (int tr = 0; tr < rankSize; ++tr) {
                for (int rBase = 0; rBase < totalRounds; rBase += batchRounds) {
                    int currentBatch = (rBase + batchRounds > totalRounds) ? (totalRounds - rBase) : batchRounds;
                    for (int r = 0; r < currentBatch; ++r) {
                        int absRound = rBase + r;
                        for (int le = 0; le < localExpertsNum; ++le) {
                            int globalExpertIdx = tr * localExpertsNum + le;
                            int srcIdx = (absRound * numExperts + globalExpertIdx) * sendPerGroup;
                            int dstIdx = (r * localExpertsNum + le) * sendPerGroup;
                            newSendDataTensor(dstIdx) = sendDataTensor(srcIdx);
                            newSendDataTensor(dstIdx + 1) = sendDataTensor(srcIdx + 1);
                            newSendDataTensor(dstIdx + 2) = sendDataTensor(srcIdx + 2);
                        }
                    }
                    AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                    uint32_t copyLen = currentBatch * localExpertsNum * sendPerGroup * sizeof(int32_t);
                    DataCopyExtParams copyParams = {1U, copyLen, 0U, 0U, 0U};
                    uint64_t gmOffset = (uint64_t)tr * totalRounds * localExpertsNum * sendPerGroup +
                                        (uint64_t)rBase * localExpertsNum * sendPerGroup;
                    DataCopyPad(sendDataInputGt[gmOffset], newSendDataTensor[0], copyParams);
                    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                }
            }
        } else {
            DataCopyPad(sendDataInputGt, sendDataTensor, {1U, sendDataAlignLen, 0U, 0U, 0U});
        }
        DataCopyExtParams sendDataOffsetParams = {1U, sendDataOffsetAlignLen, 0U, 0U, 0U};
        DataCopyPad(sendDataOffsetOutputGt, sendDataOffsetTensor, sendDataOffsetParams);

        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // copy input to other rank share
    __aicore__ inline void InputToShareSlice()
    {
        __ubuf__ uint64_t *inputUB = (__ubuf__ uint64_t *)get_imm(0);
        int32_t copyOffset = blockIdx * rankNumPerCore;
        copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            readGt = sendDataInputGt[copyOffset * perRankDataNum];
            CpGM2GMPingPong<T>(copyLen * perRankDataNum * sizeof(T), readGt, writeGt[copyOffset * perRankDataNum],
                               COPYONLY);
            uint64_t v = MergeMagicWithValue(magic, 1);
            *inputUB = v;
            AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                CpUB2GM((__gm__ uint64_t *)(shareAddrs[i]) + rank * FLAG_UNIT_INT_NUM, inputUB, sizeof(uint64_t));
            }
            pipe_barrier(PIPE_ALL);
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
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);  // Wait for GM->UB

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

    /**
     * @brief Wait for the flags starting from the specified eventID on the specified card to become
     *        a value composed of the combination of magic and value.<br>
     *        Note: [eventID, eventID + flagNum)
     */
    __aicore__ inline void WaitSyncFlag(uint64_t magic, uint64_t value, uint64_t eventID, int32_t rank, int64_t flagNum)
    {
        uint64_t v = MergeMagicWithValue(magic, value);
        WaitOneRankPartFlag((__gm__ uint64_t *)(shareAddrs[rank]) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }

    __aicore__ inline void ShareToShareSlice()
    {
        __ubuf__ T *inputUB = (__ubuf__ T *)get_imm(96);
        int32_t copyOffset = blockIdx * rankNumPerCore;
        copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            int checkRank[MAX_RANK_PER_CORE];
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                checkRank[i - copyOffset] = i + rank % copyLen;
                if (checkRank[i - copyOffset] >= copyOffset + copyLen) {
                    checkRank[i - copyOffset] -= copyLen;
                }
            }
            for (int i = 0; i < copyLen; i++) {
                readGt1[i].SetGlobalBuffer((__gm__ T *)(shareAddrs[checkRank[i]] + IPC_DATA_OFFSET));
            }

            WaitSyncFlag(magic, 1, copyOffset, rank, copyLen);

            for (int i = 0; i < copyLen; i++) {
                CpGM2GMPingPong<T>(perRankDataNum * sizeof(T), readGt1[i][rank * perRankDataNum],
                                   recvDataOutputGt[checkRank[i] * perRankDataNum], COPYONLY);
            }
        }
    }

    __aicore__ inline void ReorderOutput()
    {
        pipe.Reset();
        pipe.InitBuffer(recvDataBuf, recvDataAlignLen);
        recvDataTensor = recvDataBuf.Get<T>();
        DataCopyExtParams recvDataParams = {1U, static_cast<uint32_t>(recvDataAlignLen), 0, 0, 0};
        DataCopyPadExtParams<T> DataCopyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(recvDataTensor, recvDataOutputGt, recvDataParams, DataCopyPadExtParams);
    }

    __aicore__ inline void ReorderSendCountOutput()
    {
        pipe.InitBuffer(sendCountBuf, sendCountAlignLen);
        sendCountTensor = sendCountBuf.Get<T>();
        Duplicate<T>(sendCountTensor, 0, sendCountAlignLen / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t r = 0; r < round; ++r) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = expId * rankSize + srcRank;
                    uint32_t pair_idx =
                        sendPerGroup * (srcRank * numLocalExperts * round + r * numLocalExperts + expId);
                    sendCountTensor(r * numExperts + index) = recvDataTensor(pair_idx);
                }
            }
        }
    }

    __aicore__ inline void ReorderSendOffsetOutput()
    {
        pipe.InitBuffer(sendOffsetBuf, sendCountAlignLen);
        sendOffsetTensor = sendOffsetBuf.Get<T>();
        Duplicate<T>(sendOffsetTensor, 0, sendCountAlignLen / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t r = 0; r < round; ++r) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = expId * rankSize + srcRank;
                    uint32_t pair_idx =
                        sendPerGroup * (srcRank * numLocalExperts * round + r * numLocalExperts + expId);
                    sendOffsetTensor(r * numExperts + index) = recvDataTensor(pair_idx + 1);
                }
            }
        }
    }

    __aicore__ inline void ReorderSendTokensPerRankOutput()
    {
        pipe.InitBuffer(sendTokensPerRankBuf, sendTokensPerRankAlignLen);
        pipe.InitBuffer(seenRoundBuf, sendTokensPerRankAlignLen);
        sendTokensPerRankTensor = sendTokensPerRankBuf.Get<int32_t>();
        seenRoundTensor = seenRoundBuf.Get<int32_t>();
        Duplicate<int32_t>(sendTokensPerRankTensor, 0, sendTokensPerRankAlignLen / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t r = 0; r < round; ++r) {
            Duplicate<int32_t>(seenRoundTensor, 0, sendTokensPerRankAlignLen / sizeof(int32_t));
            SyncFunc<AscendC::HardEvent::V_S>();
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = expId * rankSize + srcRank;
                    uint32_t pair_idx =
                        sendPerGroup * (srcRank * numLocalExperts * round + r * numLocalExperts + expId);
                    if (!seenRoundTensor(srcRank)) {
                        sendTokensPerRankTensor(srcRank) += recvDataTensor(pair_idx + 2);
                        seenRoundTensor(srcRank) = 1;
                    }
                }
            }
            SyncFunc<AscendC::HardEvent::S_V>();
        }
    }

    __aicore__ inline void BuildTotalRecvTokens()
    {
        if (blockIdx != TOTAL_CNT_CORE) {
            return;
        }

        ReorderSendCountOutput();
        pipe.InitBuffer(tmpBuf_, UB_ALIGN_SIZE);
        pipe.InitBuffer(tmpBuf2_, Ceil(round * numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        LocalTensor<int32_t> totalCntLt = tmpBuf_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tmpBuf2_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = sendCountBuf.Get<float>();
        LocalTensor<float> sharedTmpBuffer = recvDataBuf.Get<float>();

        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(floatExpTokenCntLt, sendCountTensor, RoundMode::CAST_NONE, round * numExperts);
        PipeBarrier<PIPE_V>();
        ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, round * numExperts);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t sumVal = static_cast<int32_t>(floatExpTokenSumCntLt.GetValue(0));
        PipeBarrier<PIPE_V>();
        totalCntLt(0) = sumVal;
        PipeBarrier<PIPE_V>();
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        // 拷贝到outputGT
        GlobalTensor<int32_t> totalCntGt;
        totalCntGt.SetGlobalBuffer((__gm__ int32_t *)totalRecvTokens_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(totalCntGt, totalCntLt, copyParams);
    }

    __aicore__ inline void BuildRecvCount()
    {
        // 只需要sendCountTensor
        if (blockIdx != RECV_COUNT_CORE) {
            return;
        }
        ReorderSendCountOutput();
        for (uint32_t r = 0; r < round; ++r) {
            int32_t recvCountNum = 0;
            for (uint32_t expId = 0; expId < numExperts / rankSize; ++expId) {
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = r * numExperts + expId * rankSize + srcRank;
                    recvCountNum += sendCountTensor(index);
                    sendCountTensor(index) = recvCountNum;
                }
            }
        }
        GlobalTensor<int32_t> recvCntGt;
        recvCntGt.SetGlobalBuffer((__gm__ int32_t *)recvCount_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(round * numExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(recvCntGt, sendCountTensor, copyParams);
    }

    __aicore__ inline void BuildRecvOffset()
    {
        // 只需要sendOffsetTensor
        if (blockIdx != RECV_OFFSET_CORE) {
            return;
        }
        ReorderSendOffsetOutput();
        GlobalTensor<T> recvOffsetGt;
        recvOffsetGt.SetGlobalBuffer((__gm__ int32_t *)recvOffset_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(round * numExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(recvOffsetGt, sendOffsetTensor, copyParams);
    }

    __aicore__ inline void BuildMaxBs()
    {
        // 只需要maxBsNum
        if (blockIdx != MAX_BS_CORE) {
            return;
        }
        ReorderSendTokensPerRankOutput();
        for (uint32_t srcRank = 0; srcRank < numRanks; ++srcRank) {
            uint32_t tempBs = sendTokensPerRankTensor(srcRank);
            maxBsNum = maxBsNum >= tempBs ? maxBsNum : tempBs;
        }
        GlobalTensor<int32_t> maxBsGt;
        maxBsGt.SetGlobalBuffer((__gm__ int32_t *)maxBs_);
        maxBsGt.SetValue(0, maxBsNum);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(maxBsGt);
    }

    __aicore__ inline void BuildRecvTokenPerExp()
    {
        // 只需要sendCountTensor
        if (blockIdx != RECV_TOKEN_PER_EXP_CORE) {
            return;
        }
        ReorderSendCountOutput();
        pipe.InitBuffer(tmpBuf_, Ceil(round * numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        LocalTensor<int32_t> tmpTensor = tmpBuf_.Get<int32_t>();
        for (uint32_t r = 0; r < round; r++) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                int32_t localRecvCount = 0;
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = r * numExperts + expId * rankSize + srcRank;
                    localRecvCount += sendCountTensor(index);
                }
                tmpTensor(r * numLocalExperts + expId) = localRecvCount;
            }
        }
        GlobalTensor<int32_t> recvTokenPerExpGt;
        recvTokenPerExpGt.SetGlobalBuffer((__gm__ int32_t *)recvTokensPerExpert_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(round * numLocalExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(recvTokenPerExpGt, tmpTensor, copyParams);
    }

    __aicore__ inline void BuildExpGlobalOffset()
    {
        // 只需要sendCountTensor
        if (blockIdx != EXP_GLOBAL_OFFSET_CORE) {
            return;
        }
        ReorderSendCountOutput();
        pipe.InitBuffer(tmpBuf_, Ceil(numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe.InitBuffer(tmpBuf2_, Ceil(numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        LocalTensor<int32_t> tmpTensor = tmpBuf_.Get<int32_t>();
        LocalTensor<int32_t> expTensor = tmpBuf2_.Get<int32_t>();
        Duplicate<int32_t>(tmpTensor, 0, numLocalExperts);
        expTensor(0) = 0;
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t localExpTotal = 0;
        for (uint32_t r = 0; r < round; r++) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                int32_t localRecvCount = 0;
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = r * numExperts + expId * rankSize + srcRank;
                    localRecvCount += sendCountTensor(index);
                }
                tmpTensor(expId) += localRecvCount;
            }
        }
        for (uint32_t expId = 1; expId < numLocalExperts; ++expId) {
            expTensor(expId) = expTensor(expId - 1) + tmpTensor(expId - 1);
        }
        GlobalTensor<int32_t> expGlobalOffsetGt;
        expGlobalOffsetGt.SetGlobalBuffer((__gm__ int32_t *)expertGlobalOffset_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numLocalExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(expGlobalOffsetGt, expTensor, copyParams);
    }

    __aicore__ inline void BuildsrcRankInExpOffset()
    {
        if (blockIdx != SRC_RANK_EXP_OFFSET_CORE) {
            return;
        }
        ReorderSendCountOutput();
        pipe.InitBuffer(tmpBuf_, Ceil(numRanks * numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe.InitBuffer(tmpBuf2_, Ceil(numRanks * numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        LocalTensor<int32_t> expSrcTotalTensor = tmpBuf_.Get<int32_t>();
        LocalTensor<int32_t> srcRankInExpOffsetTensor = tmpBuf2_.Get<int32_t>();
        Duplicate<int32_t>(expSrcTotalTensor, 0, numExperts);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t localExpTotal = 0;
        for (uint32_t r = 0; r < round; r++) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                int32_t localRecvCount = 0;
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t index = r * numExperts + expId * rankSize + srcRank;
                    localRecvCount = sendCountTensor(index);
                    expSrcTotalTensor(expId * numRanks + srcRank) += localRecvCount;
                }
            }
        }
        for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
            int32_t cumOffset = 0;
            for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                srcRankInExpOffsetTensor(expId * numRanks + srcRank) = cumOffset;
                cumOffset += expSrcTotalTensor(expId * numRanks + srcRank);
            }
        }
        GlobalTensor<int32_t> srcRankInExpOffsetGt;
        srcRankInExpOffsetGt.SetGlobalBuffer((__gm__ int32_t *)srcrankInExpertOffset_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(srcRankInExpOffsetGt, srcRankInExpOffsetTensor, copyParams);
    }

    __aicore__ inline void BuildRInSrcrankOffset()
    {
        if (blockIdx != R_IN_SRCRANK_OFFSET_CORE) {
            return;
        }
        ReorderSendCountOutput();
        pipe.InitBuffer(tmpBuf_, Ceil(round * numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe.InitBuffer(tmpBuf2_, Ceil(numRanks * numLocalExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        LocalTensor<int32_t> rInSrcrankOffsetTensor = tmpBuf_.Get<int32_t>();
        LocalTensor<int32_t> expSrcCumPrevTensor = tmpBuf2_.Get<int32_t>();
        Duplicate<int32_t>(expSrcCumPrevTensor, 0, numExperts);
        SyncFunc<AscendC::HardEvent::V_S>();
        for (uint32_t r = 0; r < round; r++) {
            for (uint32_t expId = 0; expId < numLocalExperts; ++expId) {
                int32_t localRecvCount = 0;
                for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
                    uint32_t pairIdx = r * numExperts + expId * rankSize + srcRank;
                    uint32_t index = expId * rankSize + srcRank;
                    uint32_t cIdx = expId * numRanks * round + srcRank * round + r;
                    int32_t recvCnt = sendCountTensor(pairIdx);
                    int32_t offset = expSrcCumPrevTensor(index);
                    rInSrcrankOffsetTensor(cIdx) = offset;
                    expSrcCumPrevTensor(index) = offset + recvCnt;
                }
            }
        }
        GlobalTensor<int32_t> rInSrcrankOffsetGt;
        rInSrcrankOffsetGt.SetGlobalBuffer((__gm__ int32_t *)rInSrcrankOffset_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(round * numExperts * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(rInSrcrankOffsetGt, rInSrcrankOffsetTensor, copyParams);
    }

    __aicore__ inline int64_t GetDataCount(const int64_t dataLen, const int64_t useBlockNum);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId, uint8_t ctxIdx);
    __aicore__ inline uint64_t GetMagicValue(void);
    __aicore__ inline void InitSmallFullMesh(KERNELS_ARGS_FUN_ALL2ALL());
    template <typename F>
    __aicore__ inline void SetAtomic(int op);
    __aicore__ inline void UnsetAtomic(int op);
    template <HardEvent eventType>
    __aicore__ inline void SetWaitEvent(event_t eventId);
    template <typename K, typename U = K>
    __aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U> &sendDataInputGt,
                                           const GlobalTensor<K> &recvDataOutputGT, int op);

    GlobalTensor<T> sendDataInputGt;
    GlobalTensor<int> tokenPerExpertDataInputGt;
    GlobalTensor<T> sendDataOffsetOutputGt;
    GlobalTensor<T> recvDataOutputGt;
    GlobalTensor<int32_t> recvDataOutGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<T> readGt1[MAX_BUFFER_NUMBER];
    GlobalTensor<T> ipcGT;
    GlobalTensor<int64_t> sendCountMatrixGm;
    __gm__ T *sendDataInput;
    __gm__ int *tokenPerExpertDataInput;
    __gm__ T *sendDataOffsetOutput;
    __gm__ T *recvDataOutput;
    int64_t perNodeDataNum;
    int64_t perRankDataNum;
    int64_t curRankDataNum;

    int32_t nodeNum;
    int32_t localRankId;
    int32_t localNodeId;
    int32_t coreNumPerStageX;  // Number of cores used per stage
    int32_t coreNumPerStageY;  // Number of cores used per stage
    int32_t coreNumPerStageZ;  // Number of cores used per stage
    int32_t coreNumPerRank;    // Number of cores allocated per rank
    int32_t rankNumPerCore;    // Number of ranks responsible per core
    int32_t copyLen;           // Length of the current data slice being copied (in terms of T)

    // for coll
    int rank;
    int rankSize;
    int localRank = 0;
    int localRankSize = 0;
    int xRankSize = 0;
    int yRankSize = 0;
    int xRankIdx = 0;
    int yRankIdx = 0;
    uint32_t extraFlag;
    int round;
    int32_t perRoundTokens;
    int numTokens;
    int numRanks;
    int sendPerGroup = 3;
    int root;
    int64_t len;
    int32_t numExperts;
    int32_t numLocalExperts;
    uint64_t magic{0};
    int32_t blockIdx;  // Index of the current aicore
    int32_t blockNum;  // Total number of aicores for the current rank
    uint32_t maxBsNum{0};
    GM_ADDR scale;
    GM_ADDR shareAddrs[CAM_MAX_RANK_SIZE];  // List of shared memory addresses
    GM_ADDR totalRecvTokens_;
    GM_ADDR recvCount_;
    GM_ADDR recvOffset_;
    GM_ADDR expertGlobalOffset_;
    GM_ADDR srcrankInExpertOffset_;
    GM_ADDR rInSrcrankOffset_;
    GM_ADDR maxBs_;
    GM_ADDR recvTokensPerExpert_;
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<> tokenPerExpertDataBuf;
    TBuf<> sendDataOffsetBuf;
    TBuf<> sendCountBuf;
    TBuf<> sendOffsetBuf;
    TBuf<> sendDataBuf;
    TBuf<> newSendDataBuf;
    TBuf<> recvDataBuf;
    TBuf<> sendTokensPerRankBuf;
    TBuf<> seenRoundBuf;

    LocalTensor<int32_t> tokenPerExpertTensor;
    LocalTensor<T> sendDataTensor;
    LocalTensor<T> sendDataOffsetTensor;
    LocalTensor<T> newSendDataTensor;
    LocalTensor<int32_t> sendCountTensor;
    LocalTensor<int32_t> sendOffsetTensor;
    LocalTensor<int32_t> sendTokensPerRankTensor;
    LocalTensor<int32_t> recvDataTensor;
    LocalTensor<int32_t> seenRoundTensor;

    uint32_t sendDataAlignLen{0};
    uint32_t tokenPerExpertDataAlignLen{0};
    uint32_t recvDataAlignLen{0};
    uint32_t sendDataOffsetAlignLen{0};
    uint32_t sendCountAlignLen{0};
    uint32_t sendTokensPerRankAlignLen{0};

    TBuf<> tmpBuf_;
    TBuf<> tmpBuf2_;
    TBuf<> tmpBuf3_;
    TBuf<> tmpBuf4_;
};

template <typename T>
__aicore__ inline int64_t NotifyDispatch<T>::GetDataCount(const int64_t dataLen, const int64_t useBlockNum)
{
    return dataLen / useBlockNum;
}

template <typename T>
__aicore__ inline GM_ADDR NotifyDispatch<T>::GetWindAddrByRankId(const int32_t rankId, uint8_t ctxIdx)
{
    uint32_t curRankId = rank;
#ifdef OPT_RANK_OFFSET
#pragma message("use rank offset")
    if (curRankId == rankId) {
        return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + rankId * OPT_RANK_OFFSET;
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn) +
           rankId * OPT_RANK_OFFSET;
#else
    if (curRankId == rankId) {
        return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn);
    }
    return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn);
#endif
}

// Assign values to winContext_[COMM_EP_IDX] and blockIdx before calling
template <typename T>
__aicore__ inline uint64_t NotifyDispatch<T>::GetMagicValue(void)
{
    uint64_t magic = 0;
    GlobalTensor<uint64_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ uint64_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[blockIdx * UB_ALIGN_SIZE]);
    magic = selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE);
    if (magic <= 0) {
        magic = 1;
    }
    selfDataStatusTensor(blockIdx * UB_ALIGN_SIZE) = magic + 1;
    return magic;
}

template <typename T>
__aicore__ inline void NotifyDispatch<T>::InitSmallFullMesh(KERNELS_ARGS_FUN_ALL2ALL())
{
    this->root = root;
    this->len = len;
    this->round = round;
    this->perRoundTokens = perRoundTokens;
    this->numRanks = rankSize;
    this->numExperts = len / sendPerGroup / round;
    this->numLocalExperts = numExperts / rankSize;
    this->numTokens = numTokens;
    this->scale = scale;
    this->localRank = localRank;
    this->localRankSize = localRankSize;
    this->xRankSize = localRankSize;
    this->yRankSize = rankSize / localRankSize;
    this->xRankIdx = rank % localRankSize;
    this->yRankIdx = rank / localRankSize;
    blockIdx = GetBlockIdx();
    blockNum = GetBlockNum();
    uint8_t ctxIdx;

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    this->magic = GetMagicValue();
    ctxIdx = COMM_EP_IDX;
    uint64_t winDataOffset = (this->magic % PING_PONG_SIZE) * (totalWinSize / 2);

    shareAddrs[rank] = GetWindAddrByRankId(rank, ctxIdx) + winDataOffset;

    int32_t rankNumPerCore = (rankSize + blockNum - 1) / blockNum;
    int32_t copyOffset = blockIdx * rankNumPerCore;
    int32_t copyLen = rankSize - copyOffset < rankNumPerCore ? rankSize - copyOffset : rankNumPerCore;
    if (copyLen > 0) {
        for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
            shareAddrs[i] = GetWindAddrByRankId(i, ctxIdx) + winDataOffset;
        }
    }

    // When the number of cores is more than the number of ranks, each core is responsible for fetching data from a
    // specified rank
    int coreNumPerRank = blockNum / rankSize;  // Calculate the number of cores assigned to read for each rank, e.g., 48
                                               // cores 4 ranks, each rank is assigned 12 cores
    int maxCore = coreNumPerRank * rankSize;   // Calculate the maximum number of cores that can be used for reading,
                                               // cores exceeding this number will not take action
    if (blockIdx < maxCore) {
        // Calculate the rank to be read based on the block, 48 cores divided into 4 groups
        int readRank = blockIdx / coreNumPerRank;
        shareAddrs[readRank] = GetWindAddrByRankId(readRank, ctxIdx) + winDataOffset;
    }

    pipe.InitBuffer(tBuf, UB_FLAG_SIZE);
}

/**
 * @brief Copy data from GM to GM with ping-pong method.
 * @tparam dataSizeRemain The remaining size of data to be copied.
 * @tparam K The type of output data.
 * @tparam U The type of input data.
 * @param sendDataInputGt The global tensor of send data.
 * @param recvDataOutputGT The global tensor of recv data.
 * @param op The operation to be performed during the copy.
 * @details This function copies data from global memory to global memory using a ping-pong method.
 * It first checks if the input and output types are the same. If they are, it uses a single buffer.
 * If they are not, it divides the buffer according to the size ratio of the types and aligns it to 32 bytes.
 * Then, it sets the atomic operation, waits for the flags, and performs the copy operation.
 */
template <typename T>
template <typename K, typename U>
__aicore__ inline void NotifyDispatch<T>::CpGM2GMPingPong(int64_t dataSizeRemain,
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

template <typename T>
template <typename F>
__aicore__ inline void NotifyDispatch<T>::SetAtomic(int op)
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
__aicore__ inline void NotifyDispatch<T>::UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <HardEvent eventType>
__aicore__ inline void NotifyDispatch<T>::SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

#endif  // NOTIFY_DISPATCH_H
