#ifndef NOTIFY_DISPATCH_H
#define NOTIFY_DISPATCH_H

#include <climits>
#include "kernel_operator.h"

#include "shmem_api.h"
#include "comm_args.h"
#include "data_copy.h"
// #include "moe_distribute_base.h"

using namespace AscendC;
using namespace Moe;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define KERNELS_ARGS_FUN_ALL2ALL()                                                                                    \
    GM_ADDR tokenPerExpertDataInput, GM_ADDR sendDataOffsetOutput, GM_ADDR totalRecvTokens, GM_ADDR recvCount,        \
        GM_ADDR maxBs, GM_ADDR recvTokensPerExpert, int64_t len, int64_t numTokens, int op, int root, int cycleCount, \
        GM_ADDR scale, int64_t scaleCount, GM_ADDR offset, int localRank, int localRankSize, uint64_t shmemPtr

#define KERNELS_ARGS_CALL_ALL2ALL()                                                                             \
    tokenPerExpertDataInput, sendDataOffsetOutput, totalRecvTokens, recvCount, maxBs, recvTokensPerExpert, len, \
        numTokens, op, root, cycleCount, scale, scaleCount, offset, localRank, localRankSize, shmemPtr

template <typename T>
class NotifyDispatch
{
    constexpr static uint64_t WIN_MAGIC_OFFSET = 100UL * 1024UL;  // notify(50kb) + dispatch&combine(50kb)
    constexpr static uint64_t HALF_WIN_STATE_OFFSET =
        8 * 1024UL * 1024UL;  // notify(2MB) + dispatch(3MB) + combine(3MB)

    constexpr static int UB_ALIGN_SIZE = 32;
    constexpr static int64_t MAX_RANK_PER_CORE = 8;
    constexpr static int64_t MULTI_RANK_SIZE = 48;
    constexpr static int64_t MAX_BUFFER_NUMBER = 10;
    constexpr static uint32_t UB_FLAG_SIZE = 8U * 1024U;
    // Synchronization flag occupies length
    constexpr static int64_t FLAG_UNIT_INT_NUM = 4;
    constexpr static int64_t MAGIC_MASK = ~((1LL << 32) - 1);

public:
    __aicore__ inline NotifyDispatch(int epRankId_, int epWorldSize_, uint32_t extraFlag)
        : epRankId_(epRankId_), epWorldSize_(epWorldSize_), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_ALL2ALL())
    {
        InitSmallFullMesh(KERNELS_ARGS_CALL_ALL2ALL());
        nodeNum = epWorldSize_ / localRankSize;
        localRankId = epRankId_ % localRankSize;
        localNodeId = epRankId_ / localRankSize;
        rankNumPerCore = (epWorldSize_ + blockNum_ - 1) / blockNum_;
        perNodeDataNum = len / nodeNum;       // 128K/4 = 32K
        perRankDataNum = len / epWorldSize_;  // len = sendCount 组装的三元组信息长度 len / 16
        totalRecvTokens_ = totalRecvTokens;
        recvCount_ = recvCount;
        recvOffset_ = recvOffset;
        maxBs_ = maxBs;
        recvTokensPerExpert_ = recvTokensPerExpert;
        recvDataAlignLen_ = Ceil(numExperts * epWorldSize_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        sendDataOffsetAlignLen = Ceil(numExperts * sizeof(T), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        tokenPerExpertDataAlignLen_ = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        allRecvCountDataAlignLen_ = Ceil(numExperts * epWorldSize_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        this->tokenPerExpertDataInput = (__gm__ int32_t *)tokenPerExpertDataInput;
        this->sendDataOffsetOutput = (__gm__ T *)sendDataOffsetOutput;
        this->recvDataOutput = (__gm__ T *)recvDataOutput;
        allRecvCountOutput_ = (__gm__ int32_t *)recvCount_;
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        sendDataOffsetOutputGt.SetGlobalBuffer((__gm__ T *)sendDataOffsetOutput);
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);

        recvDataGt_.SetGlobalBuffer((__gm__ int32_t *)allRecvCountOutput_);
        recvCountOutGT_.SetGlobalBuffer((__gm__ int32_t *)allRecvCountOutput_);
        pipe_.InitBuffer(sendCountBuf_, recvDataAlignLen_);
        // pipe_.InitBuffer(sendOffsetBuf, tokenPerExpertDataAlignLen_);
        pipe_.InitBuffer(recvDataBuf_, recvDataAlignLen_);
    }

    __aicore__ inline void Process()
    {
        AllGatherSendCount();  // allgather 每个rank的sendCount
        SyncAll<true>();
        ReorderOutput();
        if (blockIdx_ == 0) {
            BuildMaxBs();
        } else if (blockIdx_ == 1) {
            BuildRecvCount();
        } else if (blockIdx_ == 2) {
            BuildTotalRecvTokens();
        }
        SyncAll<true>();
    }

private:
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
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
    int64_t perNodeDataNum;
    int64_t perRankDataNum;
    int64_t curRankDataNum;

    int64_t nodeNum;
    int64_t localRankId;
    int64_t localNodeId;
    int64_t coreNumPerRank;  // Number of cores allocated per epRankId_
    int64_t rankNumPerCore;  // Number of ranks responsible per core
    int64_t copyLen;         // Length of the current data slice being copied (in terms of T)

    // for coll
    int epRankId_;
    int epWorldSize_;
    int64_t blockIdx_;  // Index of the current aicore
    int64_t blockNum_;  // Total number of aicores for the current epRankId_
    int localRank = 0;
    int localRankSize = 0;
    uint32_t extraFlag;
    int32_t numTokens_;
    int sendPerGroup = 3;
    int64_t len;
    int64_t numExperts;
    uint64_t magic{0};
    uint32_t bufferId_{0};

    GlobalTensor<int> tokenPerExpertDataInputGt;
    GlobalTensor<T> sendDataOffsetOutputGt;
    GlobalTensor<T> recvDataOutputGt;
    GlobalTensor<int32_t> recvDataGt_;
    GlobalTensor<T> readGt1[MAX_BUFFER_NUMBER];
    GlobalTensor<int32_t> recvCountOutGT_;

    LocalTensor<int32_t> sendCountTensor_;
    LocalTensor<int32_t> sendOffsetTensor;
    LocalTensor<int32_t> recvDataTensor_;
    uint32_t sendDataAlignLen_{0};
    uint32_t tokenPerExpertDataAlignLen_{0};
    uint32_t allRecvCountDataAlignLen_{0};
    uint32_t recvDataAlignLen_{0};
    uint32_t sendDataOffsetAlignLen{0};

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<> tokenPerExpertDataBuf_;
    TBuf<> sendDataOffsetBuf;
    TBuf<> sendCountBuf_;
    TBuf<> sendOffsetBuf;
    TBuf<> sendDataBuf_;
    TBuf<> recvDataBuf_;
    TBuf<> tmpBuf_;
    TBuf<> tmpBuf2_;
    TBuf<> tmpBuf3_;
    TBuf<> tmpBuf4_;

    __gm__ int *tokenPerExpertDataInput;
    __gm__ T *sendDataOffsetOutput;
    __gm__ T *recvDataOutput;
    __gm__ int32_t *allRecvCountOutput_;
    GM_ADDR scale;
    GM_ADDR totalRecvTokens_;
    GM_ADDR recvCount_;
    GM_ADDR recvOffset_;
    GM_ADDR maxBs_;
    GM_ADDR recvTokensPerExpert_;

    GM_ADDR gva_gm;
    GM_ADDR shareRecvDataAddrs[CAM_MAX_RANK_SIZE];  // List of shmem asymmetric output addresses (send_data)

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx)
    {
        perCoreNum = totalNum / blockNum_;
        uint32_t remainderRankNum = totalNum % blockNum_;

        startIdx = perCoreNum * blockIdx_;
        if (blockIdx_ < remainderRankNum) {
            perCoreNum++;
            startIdx += blockIdx_;
        } else {
            startIdx += remainderRankNum;
        }
        endIdx = startIdx + perCoreNum;
    }

    // allgather每个rank的recvCount，采用分核策略，
    __aicore__ inline void AllGatherSendCount()
    {
        uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
        SplitCoreCal(epWorldSize_, rankNumPerBlock, startRankId, endRankId);
        if (rankNumPerBlock == 0U) {
            return;
        }
        for (uint32_t targetRankId = startRankId; targetRankId < endRankId; targetRankId++) {
            shmem_get_int32_mem_nbi(recvDataGt_[targetRankId], tokenPerExpertDataInputGt, numExperts, targetRankId);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ReorderOutput()
    {
        recvDataTensor_ = recvDataBuf_.Get<int32_t>();
        DataCopyExtParams recvDataParams = {1U, static_cast<uint32_t>(recvDataAlignLen_), 0, 0, 0};
        DataCopyPadExtParams<int32_t> DataCopyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(recvDataTensor_, recvDataGt_, recvDataParams, DataCopyPadExtParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReorderSendCountOutput()
    {
        sendCountTensor_ = sendCountBuf_.Get<int32_t>();
        Duplicate<int32_t>(sendCountTensor_, 0, recvDataAlignLen_ / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t expId = 0; expId < numExperts; ++expId) {
            for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
                uint32_t index = expId * epWorldSize_ + srcRank;
                uint32_t pairIdx = srcRank * numExperts / epWorldSize_ + expId;
                sendCountTensor_(index) = recvDataTensor_(pairIdx);
            }
        }
    }

    __aicore__ inline void BuildMaxBs()
    {
        // 只需要sendCountTensor
        int32_t maxBsNum = 0;
        pipe_.InitBuffer(tmpBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf2_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf3_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf4_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        LocalTensor<int32_t> maxBsLt = tmpBuf_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tmpBuf2_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = tmpBuf3_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tmpBuf4_.Get<float>();
        SyncFunc<AscendC::HardEvent::S_V>();
        for (uint32_t srcRankId = 0; srcRankId < epWorldSize_; srcRankId++) {
            Cast(floatExpTokenCntLt, sendCountTensor_[numExperts * srcRankId], RoundMode::CAST_NONE, numExperts);
            PipeBarrier<PIPE_V>();
            ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, numExperts);
            SyncFunc<AscendC::HardEvent::V_S>();
            int32_t curRankBsNum = static_cast<int32_t>(floatExpTokenSumCntLt.GetValue(0));
            maxBsNum = curRankBsNum > maxBsNum ? curRankBsNum : maxBsNum;
            PipeBarrier<PIPE_V>();
        }

        SyncFunc<AscendC::HardEvent::V_S>();
        // 拷贝到outputGT
        GlobalTensor<int32_t> maxBsGt;
        maxBsGt.SetGlobalBuffer((__gm__ int32_t *)maxBs_);
        maxBsGt.SetValue(0, maxBsNum);
    }

    __aicore__ inline void BuildRecvCount()
    {
        // 只需要sendCountTensor
        ReorderSendCountOutput();
        int32_t recvCountNum = 0;
        for (uint32_t expId = 0; expId < numExperts; ++expId) {
            for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
                uint32_t index = expId * epWorldSize_ + srcRank;
                recvCountNum += sendCountTensor_(expId * epWorldSize_ + srcRank);
                sendCountTensor_(index) = recvCountNum;
            }
        }

        GlobalTensor<int32_t> recvCntGt;
        recvCntGt.SetGlobalBuffer((__gm__ int32_t *)recvCount_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * epWorldSize_ * sizeof(int32_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(recvCntGt, sendCountTensor_, copyParams);
    }

    __aicore__ inline void BuildTotalRecvTokens()
    {
        // 只需要sendCountTensor
        // 只计算当前rank需要接收的那部分
        ReorderSendCountOutput();
        pipe_.InitBuffer(tmpBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf2_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf3_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf4_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        LocalTensor<int32_t> totalCntLt = tmpBuf_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tmpBuf2_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = tmpBuf3_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tmpBuf4_.Get<float>();
        SyncFunc<AscendC::HardEvent::S_V>();
        Cast(floatExpTokenCntLt, sendCountTensor_[epRankId_ * numExperts], RoundMode::CAST_NONE, numExperts);
        PipeBarrier<PIPE_V>();
        ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, numExperts);
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

    // 分核向对应rank发送flag
    __aicore__ inline void SetShmemFlag()
    {
        __ubuf__ uint64_t *inputUB = (__ubuf__ uint64_t *)get_imm(0);
        int64_t copyOffset = blockIdx_ * rankNumPerCore;  // 16个rank 每个核负责一个rank
        copyLen = epWorldSize_ - copyOffset < rankNumPerCore ? epWorldSize_ - copyOffset : rankNumPerCore;
        if (copyLen > 0) {
            uint64_t v = MergeMagicWithValue(magic, 1);
            *inputUB = v;
            AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            for (int i = copyOffset; i < copyOffset + copyLen; ++i) {
                GM_ADDR remote_state = GetWindStateAddrByRankId(i);

                CpUB2GM((__gm__ uint64_t *)(remote_state) + epRankId_ * FLAG_UNIT_INT_NUM, inputUB, sizeof(uint64_t));
            }
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline uint64_t MergeMagicWithValue(uint64_t magic, uint64_t value)
    {
        // magic as the high part, eventID as the low part, combined into a value for comparison
        return (magic * 2ULL + value);
    }

    // Wait for a part of synchronization flags within a epRankId_
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
    __aicore__ inline void WaitShmemFlag(uint64_t magic, uint64_t value, uint64_t eventID, int32_t epRankId_,
                                         int64_t flagNum)
    {
        uint64_t v = MergeMagicWithValue(magic, value);
        GM_ADDR remote_state = GetWindStateAddrByRankId(epRankId_);
        WaitOneRankPartFlag((__gm__ uint64_t *)(remote_state) + eventID * FLAG_UNIT_INT_NUM, flagNum, v);
    }
};
template <typename T>
__aicore__ inline GM_ADDR NotifyDispatch<T>::GetWindStateAddrByRankId(const int32_t rankId)
{
    auto ptr = shmem_ptr((__gm__ T *)gva_gm, rankId);

    return (GM_ADDR)(ptr) + WIN_MAGIC_OFFSET + bufferId_ * HALF_WIN_STATE_OFFSET;
}

// Assign values to gva_gm and blockIdx_ before calling, magic buffer 24kb
template <typename T>
__aicore__ inline uint64_t NotifyDispatch<T>::GetMagicValue(void)
{
    uint64_t magic = 0;
    GlobalTensor<uint64_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(gva_gm);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ uint64_t *)(statusDataSpaceGm));
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[blockIdx_ * UB_ALIGN_SIZE]);
    magic = selfDataStatusTensor(blockIdx_ * UB_ALIGN_SIZE);
    if (magic <= 0) {
        magic = 1;
    }
    selfDataStatusTensor(blockIdx_ * UB_ALIGN_SIZE) = magic + 1;
    return magic;
}

template <typename T>
__aicore__ inline void NotifyDispatch<T>::InitSmallFullMesh(KERNELS_ARGS_FUN_ALL2ALL())
{
    this->len = len;
    this->numExperts = len / sendPerGroup;
    numTokens_ = numTokens_;
    this->scale = scale;
    this->localRank = localRank;
    this->localRankSize = localRankSize;
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    gva_gm = (GM_ADDR)shmemPtr;  // 作为原来的 winContext_

    this->magic = GetMagicValue();
    bufferId_ = this->magic % PING_PONG_SIZE;

    // When the number of cores is more than the number of ranks, each core is responsible for fetching data from a
    // specified epRankId_
    int coreNumPerRank =
        blockNum_ / epWorldSize_;  // Calculate the number of cores assigned to read for each epRankId_, e.g., 48
                                   // cores 4 ranks, each epRankId_ is assigned 12 cores
    int maxCore = coreNumPerRank * epWorldSize_;  // Calculate the maximum number of cores that can be used for reading,
                                                  // cores exceeding this number will not take action

    pipe_.InitBuffer(tBuf, UB_FLAG_SIZE);
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
