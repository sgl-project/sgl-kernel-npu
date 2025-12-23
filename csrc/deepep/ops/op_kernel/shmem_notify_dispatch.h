#ifndef SHMEM_NOTIFY_DISPATCH_H
#define SHMEM_NOTIFY_DISPATCH_H

#include <climits>
#include "kernel_operator.h"

#include "shmem_api.h"
#include "comm_args.h"
#include "data_copy.h"

using namespace AscendC;
using namespace Moe;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define KERNELS_ARGS_FUN_ALLGATHER()                                                                                    \
    GM_ADDR tokenPerExpertData, GM_ADDR recvDataOutput, GM_ADDR totalRecvTokens,         \
        GM_ADDR maxBs, GM_ADDR recvTokensPerExpert, GM_ADDR putOffset, int64_t len, uint32_t topkNum, int root,  \
        int localRank, int localRankSize, uint64_t shmemPtr

#define KERNELS_ARGS_CALL_ALLGATHER()                                                                             \
    tokenPerExpertData, recvDataOutput, totalRecvTokens, maxBs, recvTokensPerExpert, putOffset, len, \
        topkNum, root, localRank, localRankSize, shmemPtr

template <typename T>
class ShmemNotifyDispatch
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
    __aicore__ inline ShmemNotifyDispatch(int epRankId_, int epWorldSize_, uint32_t extraFlag)
        : epRankId_(epRankId_), epWorldSize_(epWorldSize_), extraFlag(extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN_ALLGATHER())
    {
        this->len = len;
        this->numExperts = len / sendPerGroup; // len为 num_tokens_per_expert长度，即专家数
        this->localRank = localRank;
        this->localRankSize = localRankSize;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();

        gva_gm = (GM_ADDR)shmemPtr;  // 作为原来的 winContext_
        this->magic = 0; // 使用shmem_barrier可以考虑不需要 magic
        bufferId_ = this->magic % PING_PONG_SIZE;

        nodeNum = epWorldSize_ / localRankSize;
        localRankId = epRankId_ % localRankSize;
        localNodeId = epRankId_ / localRankSize;
        rankNumPerCore = (epWorldSize_ + blockNum_ - 1) / blockNum_;
        topkNum_ = topkNum;
        perRankDataNum = len;  // allgather, 发送所有数据
        tokenPerExpertData_ = tokenPerExpertData;
        totalRecvTokens_ = totalRecvTokens;
        allRecvCount_ = putOffset;
        maxBs_ = maxBs;
        recvTokensPerExpert_ = recvTokensPerExpert;
        recvData_ = recvDataOutput;

        recvDataAlignLen_ = Ceil(numExperts * epWorldSize_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        tokenPerExpertDataAlignLen_ = Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
        allRecvCountDataAlignLen_ = Ceil(numExperts * epWorldSize_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;

        this->tokenPerExpertDataInput = (__gm__ int32_t *)tokenPerExpertData;
        tokenPerExpertDataInputGt.SetGlobalBuffer((__gm__ int32_t *)tokenPerExpertDataInput);
        this->recvDataOutput = (__gm__ T *)recvDataOutput;
        recvDataOutputGt.SetGlobalBuffer((__gm__ T *)recvDataOutput);
        recvDataGt_.SetGlobalBuffer((__gm__ int32_t *)recvDataOutput);
        recvCntGt.SetGlobalBuffer((__gm__ int32_t *)allRecvCount_);

        pipe_.InitBuffer(tBuf, UB_FLAG_SIZE);
    }

    __aicore__ inline void Process()
    {
        AllGatherSendData();  // allgather 每个rank的sendCount
        SyncAll<true>();
        shmem_barrier_all();  // 全卡同步，确保数据已经获取完

        ReloadRecvData();
        int32_t remainBlockIdx = (blockNum_ / 2);
        BuildTotalRecvCount();
        if (blockIdx_ == remainBlockIdx) {
            BuildMaxBs();
        } else if (blockIdx_ == remainBlockIdx + 1) {
            BuildTotalRecvTokens();
        } else if (blockIdx_ == remainBlockIdx + 2) {
            BuildRecvTokenPerExp();
        }
        SyncAll<true>();
    }

private:
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline uint64_t GetMagicValue(void);
    template <typename F>
    __aicore__ inline void SetAtomic(int op);
    __aicore__ inline void UnsetAtomic(int op);
    template <HardEvent eventType>
    __aicore__ inline void SetWaitEvent(event_t eventId);
    template <typename K, typename U = K>
    __aicore__ inline void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U> &sendDataInputGt,
                                           const GlobalTensor<K> &recvDataOutputGT, int op);
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
    uint32_t topkNum_;
    int sendPerGroup = 1;
    int64_t len;
    int64_t numExperts;
    uint64_t magic{0};
    uint32_t bufferId_{0};

    GlobalTensor<int> tokenPerExpertDataInputGt;
    GlobalTensor<T> recvDataOutputGt;
    GlobalTensor<int32_t> recvDataGt_;
    GlobalTensor<T> readGt1[MAX_BUFFER_NUMBER];
    // GlobalTensor<int32_t> recvCountOutGT_;
    GlobalTensor<int32_t> recvCntGt;

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
    TBuf<> sendCountBuf_;
    TBuf<> recvDataBuf_;
    TBuf<> localRecvDataBuf_;
    TBuf<> tmpBuf_;
    TBuf<> tmpBuf2_;
    TBuf<> tmpBuf3_;
    TBuf<> tmpBuf4_;

    __gm__ int *tokenPerExpertDataInput;
    __gm__ T *recvDataOutput;
    __gm__ int32_t *allRecvCountOutput_;
    GM_ADDR tokenPerExpertData_;
    GM_ADDR totalRecvTokens_;
    GM_ADDR allRecvCount_;
    GM_ADDR maxBs_;
    GM_ADDR recvTokensPerExpert_;
    GM_ADDR recvData_;

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

    // allgather每个rank的num_tokens_per_expert，采用分核策略
    __aicore__ inline void AllGatherSendData()
    {
        uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
        SplitCoreCal(epWorldSize_, rankNumPerBlock, startRankId, endRankId);
        if (rankNumPerBlock == 0U) {
            return;
        }

        AscendC::GlobalTensor<int32_t> gmRemoteDataGt;
        for (uint32_t targetRankId = startRankId; targetRankId < endRankId; targetRankId++) {
            auto ptr = shmem_ptr(tokenPerExpertData_, targetRankId);
            gmRemoteDataGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ptr));

            CpGM2GMPingPong<int32_t>(numExperts * sizeof(int32_t), gmRemoteDataGt, 
                                recvDataGt_[targetRankId * numExperts], COPYONLY);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ReloadRecvData()
    {
        pipe_.Reset();
        pipe_.InitBuffer(recvDataBuf_, recvDataAlignLen_);

        recvDataTensor_ = recvDataBuf_.Get<int32_t>();
        DataCopyExtParams recvDataParams = {1U, static_cast<uint32_t>(recvDataAlignLen_), 0, 0, 0};
        DataCopyPadExtParams<int32_t> DataCopyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(recvDataTensor_, recvDataGt_, recvDataParams, DataCopyPadExtParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReorderRecvDataOutput(int32_t rankId, LocalTensor<int32_t> &transLt, bool isCumSum = false)
    {
        // SyncFunc<AscendC::HardEvent::MTE3_S>();
        uint32_t moeExpertPerRankNum = numExperts / epWorldSize_;
        uint32_t startExpId = rankId * moeExpertPerRankNum;
        uint32_t endExpId = rankId * moeExpertPerRankNum + moeExpertPerRankNum;

        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        // 对recv_data进行转置
        int32_t prefixSum = 0; // 每卡求前缀和，调整为偏移，起始偏移从0开始
        for (uint32_t expId = startExpId; expId < endExpId; ++expId) {
            for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
                uint32_t index = (expId - startExpId) * epWorldSize_ + srcRank;
                uint32_t pairIdx = srcRank * numExperts + expId;

                int32_t curRecvCount = recvDataTensor_(pairIdx);
                transLt(index) = isCumSum ? prefixSum : curRecvCount; // 根据是否需要前缀和进行填充
                prefixSum += curRecvCount;
            }
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }

    __aicore__ inline void BuildMaxBs()
    {
        // 需要recvData
        pipe_.InitBuffer(localRecvDataBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf2_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf3_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf4_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};

        LocalTensor<int32_t> tokenPerExpertDataLt = localRecvDataBuf_.Get<int32_t>();
        LocalTensor<int32_t> maxBsLt = tmpBuf_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tmpBuf2_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = tmpBuf3_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tmpBuf4_.Get<float>();

        int32_t maxBsNum = 0;
        for (uint32_t srcRankId = 0; srcRankId < epWorldSize_; srcRankId++) {
            DataCopy(tokenPerExpertDataLt, recvDataTensor_[numExperts * srcRankId], numExperts);
            PipeBarrier<PIPE_ALL>();
            SyncFunc<AscendC::HardEvent::MTE2_V>();

            Cast(floatExpTokenCntLt, tokenPerExpertDataLt, RoundMode::CAST_NONE, numExperts);
            PipeBarrier<PIPE_V>();
            ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, numExperts);
            SyncFunc<AscendC::HardEvent::V_S>();

            int32_t curRankBsNum = static_cast<int32_t>(floatExpTokenSumCntLt(0));
            maxBsNum = curRankBsNum > maxBsNum ? curRankBsNum : maxBsNum;
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();

        // 拷贝到outputGT
        GlobalTensor<int32_t> maxBsGt;
        maxBsGt.SetGlobalBuffer((__gm__ int32_t *)maxBs_);

        maxBsGt.SetValue(0, maxBsNum / topkNum_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(maxBsGt);
    }

    __aicore__ inline void BuildTotalRecvCount()
    {
        uint32_t maxUseCoreNum = epWorldSize_ > (blockNum_ / 2) ? (blockNum_ / 2) : epWorldSize_;
        uint32_t perCoreNum = epWorldSize_ / maxUseCoreNum;
        uint32_t remainderRankNum = epWorldSize_ % maxUseCoreNum;

        uint32_t startRankId = perCoreNum * blockIdx_;
        if (blockIdx_ < remainderRankNum) {
            perCoreNum += 1;
            startRankId += blockIdx_;
        } else {
            startRankId += remainderRankNum;
        }
        uint32_t endRankId = startRankId + perCoreNum;
        if (perCoreNum == 0U || blockIdx_ >= maxUseCoreNum) {
            return;
        }

        pipe_.InitBuffer(sendCountBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        LocalTensor<int32_t> recvTokenLt = sendCountBuf_.Get<int32_t>();

        for (uint32_t rank = startRankId; rank < endRankId; ++rank) {
            // 每卡求前缀和
            ReorderRecvDataOutput(rank, recvTokenLt, true); // localExpNum * ranks

            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(numExperts * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(recvCntGt[rank * numExperts], recvTokenLt, copyParams);
        }
    }

    __aicore__ inline void BuildTotalRecvTokens()
    {
        // 需要recvData, 转置后取当前rank的部分
        pipe_.InitBuffer(localRecvDataBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf_, Ceil(1 * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf2_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf3_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf4_, Ceil(numExperts * sizeof(float), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        LocalTensor<int32_t> recvTokenLt = localRecvDataBuf_.Get<int32_t>();
        LocalTensor<int32_t> totalCntLt = tmpBuf_.Get<int32_t>();
        LocalTensor<float> floatExpTokenCntLt = tmpBuf2_.Get<float>();
        LocalTensor<float> floatExpTokenSumCntLt = tmpBuf3_.Get<float>();
        LocalTensor<float> sharedTmpBuffer = tmpBuf4_.Get<float>();

        int32_t sumVal = 0; // 所有rank中接收token最大的
        for (uint32_t srcRankId = 0; srcRankId < epWorldSize_; srcRankId++) {
            ReorderRecvDataOutput(srcRankId, recvTokenLt, false); // localExpNum * ranks

            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(floatExpTokenCntLt, recvTokenLt, RoundMode::CAST_NONE, numExperts);
            PipeBarrier<PIPE_V>();
            ReduceSum(floatExpTokenSumCntLt, floatExpTokenCntLt, sharedTmpBuffer, numExperts);
            SyncFunc<AscendC::HardEvent::V_S>();
            int32_t recvCnt = static_cast<int32_t>(floatExpTokenSumCntLt.GetValue(0));
            PipeBarrier<PIPE_ALL>();
            sumVal = sumVal > recvCnt ? sumVal : recvCnt;
        }

        // 拷贝到outputGT
        GlobalTensor<int32_t> totalCntGt;
        totalCntGt.SetGlobalBuffer((__gm__ int32_t *)totalRecvTokens_);

        totalCntGt.SetValue(0, sumVal);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(totalCntGt);
    }

    __aicore__ inline void BuildRecvTokenPerExp()
    {
        // 需要recvData, 转置后取当前rank的部分
        uint32_t moeExpertPerRankNum = numExperts / epWorldSize_;
        pipe_.InitBuffer(localRecvDataBuf_, Ceil(numExperts * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
        pipe_.InitBuffer(tmpBuf_, Ceil(moeExpertPerRankNum * sizeof(int64_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);

        LocalTensor<int32_t> recvTokenLt = localRecvDataBuf_.Get<int32_t>();
        ReorderRecvDataOutput(epRankId_, recvTokenLt, false); // localExpNum * ranks
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        LocalTensor<int64_t> tmpTensor = tmpBuf_.Get<int64_t>();
        for (uint32_t expId = 0; expId < moeExpertPerRankNum; ++expId) {
            int64_t localRecvCount = 0;
            for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
                uint32_t index = expId * epWorldSize_ + srcRank;
                localRecvCount += recvTokenLt(index);
            }
            tmpTensor(expId) = localRecvCount;
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        GlobalTensor<int64_t> recvTokenPerExpGt;
        recvTokenPerExpGt.SetGlobalBuffer((__gm__ int64_t *)recvTokensPerExpert_);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(moeExpertPerRankNum * sizeof(int64_t)), 0, 0, 0};
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(recvTokenPerExpGt, tmpTensor, copyParams);
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
__aicore__ inline GM_ADDR ShmemNotifyDispatch<T>::GetWindStateAddrByRankId(const int32_t rankId)
{
    auto ptr = shmem_ptr((__gm__ T *)gva_gm, rankId);

    return (GM_ADDR)(ptr) + WIN_MAGIC_OFFSET + bufferId_ * HALF_WIN_STATE_OFFSET;
}

// Assign values to gva_gm and blockIdx_ before calling, magic buffer 24kb
template <typename T>
__aicore__ inline uint64_t ShmemNotifyDispatch<T>::GetMagicValue(void)
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
__aicore__ inline void ShmemNotifyDispatch<T>::CpGM2GMPingPong(int64_t dataSizeRemain,
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
__aicore__ inline void ShmemNotifyDispatch<T>::SetAtomic(int op)
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
__aicore__ inline void ShmemNotifyDispatch<T>::UnsetAtomic(int op)
{
    if (op != -1) {
        AscendC::SetAtomicNone();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
template <HardEvent eventType>
__aicore__ inline void ShmemNotifyDispatch<T>::SetWaitEvent(event_t eventId)
{
    AscendC::SetFlag<eventType>(eventId);
    AscendC::WaitFlag<eventType>(eventId);
}

#endif  // SHMEM_NOTIFY_DISPATCH_H
