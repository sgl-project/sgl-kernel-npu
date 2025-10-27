#ifndef DISPATCH_LAYOUT_A2_H
#define DISPATCH_LAYOUT_A2_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "sync_collectives.h"
#include "moe_distribute_base.h"
#include "dispatch_layout_tiling.h"

namespace MoeDispatchLayoutA2 {

constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t MAX_BATCH_SIZE = 4096U;
constexpr uint32_t TEMP_BATCH_SIZE = 64U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

using namespace AscendC;
using namespace Moe;
template <typename T>
class DispatchLayoutA2
{
public:
    __aicore__ inline DispatchLayoutA2(){};

    __aicore__ inline void Init(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                GM_ADDR isTokenInRank, GM_ADDR notifySendData, GM_ADDR workspace, TPipe *pipe,
                                const DispatchLayoutTilingData *tilingData)
    {
        numTokens_ = tilingData->dispatchLayoutInfo.numTokens;
        numRanks_ = tilingData->dispatchLayoutInfo.numRanks;
        numExperts_ = tilingData->dispatchLayoutInfo.numExperts;
        numTopk_ = tilingData->dispatchLayoutInfo.numTopk;
        localRankSize_ = tilingData->dispatchLayoutInfo.localRankSize;
        serverNum_ = (numRanks_ + localRankSize_ - 1) / localRankSize_;
        tpipe_ = pipe;

        coreIdx_ = GetBlockIdx();
        uint32_t maxAivNum = GetBlockNum() - 1;
        aivNum_ = numTokens_ <= maxAivNum ? numTokens_ : maxAivNum;
        uint32_t temp = numTokens_ / aivNum_;
        uint32_t restNum = numTokens_ % aivNum_;
        int64_t topkIdxOffset;
        int64_t isTokenOffset;
        int64_t serverOffsetOffset;
        int64_t serverNumOffset;
        tempTokens_ = temp;

        if (coreIdx_ < aivNum_) {
            if (coreIdx_ < restNum) {
                tempTokens_++;
            }
            topkIdx32AlignIntLen_ = Ceil(tempTokens_ * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
            numTokensPerRank32AlignIntLen_ = Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            numTokensPerExpert32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            isTokenInRank32AlignIntLen_ = Ceil(tempTokens_ * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            localTokenServerUniqCount32AlignIntLen_ = Ceil(serverNum_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            localTokenServerTotalCount32AlignIntLen_ =
                Ceil(tempTokens_ * serverNum_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            localTokenServerNum32AlignIntLen_ = Ceil(tempTokens_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

            if (coreIdx_ < restNum) {
                topkIdxOffset = coreIdx_ * tempTokens_ * numTopk_ * sizeof(int64_t);
                isTokenOffset = coreIdx_ * tempTokens_ * numRanks_ * sizeof(T);
                serverOffsetOffset = coreIdx_ * tempTokens_ * serverNum_ * sizeof(T);
                serverNumOffset = coreIdx_ * tempTokens_ * sizeof(T);
            } else {
                topkIdxOffset = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(int64_t);
                isTokenOffset = (restNum + coreIdx_ * tempTokens_) * numRanks_ * sizeof(T);
                serverOffsetOffset = (restNum + coreIdx_ * tempTokens_) * serverNum_ * sizeof(T);
                serverNumOffset = (restNum + coreIdx_ * tempTokens_) * sizeof(T);
            }

            topkIdxGM_.SetGlobalBuffer((__gm__ int64_t *)(topkIdx + topkIdxOffset));
            numTokensPerRankGM_.SetGlobalBuffer((__gm__ T *)numTokensPerRank);
            numTokensPerExpertSrcGM_.SetGlobalBuffer((__gm__ T *)numTokensPerExpert);
            numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T *)notifySendData);
            isTokenInRankGM_.SetGlobalBuffer((__gm__ T *)(isTokenInRank + isTokenOffset));
            localTokenServerUniqCountGM_.SetGlobalBuffer((__gm__ T *)(notifySendData) + numExperts_);
            localTokenServerTotalCountGM_.SetGlobalBuffer((__gm__ T *)(notifySendData + serverOffsetOffset) +
                                                          numExperts_ + serverNum_);
            localTokenServerNumGM_.SetGlobalBuffer((__gm__ T *)(notifySendData + serverNumOffset) + numExperts_ +
                                                   serverNum_ * (MAX_BATCH_SIZE + 1));
        }
        if (coreIdx_ == aivNum_) {
            expertRankTokenIdx32AlignIntLen_ =
                Ceil(numExperts_ * TEMP_BATCH_SIZE * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            sendTokenIdx32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            localTokenServerOffset32AlignIntLen_ = Ceil(numTokens_ * serverNum_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
            topkIdxGM_.SetGlobalBuffer((__gm__ int64_t *)topkIdx);
            localTokenServerOffsetGM_.SetGlobalBuffer((__gm__ T *)notifySendData + numExperts_ + serverNum_ +
                                                      MAX_BATCH_SIZE * (serverNum_ + 1));
            sendTokenIdxGM_.SetGlobalBuffer((__gm__ T *)notifySendData + numExperts_ + serverNum_ +
                                            MAX_BATCH_SIZE * (1 + 2 * serverNum_));
            expertRankTokenIdxGM_.SetGlobalBuffer((__gm__ T *)notifySendData + numExperts_ + serverNum_ +
                                                  MAX_BATCH_SIZE * (1 + 2 * serverNum_ + numExperts_));
        }
    }

    __aicore__ inline void Process()
    {
        if (coreIdx_ < aivNum_) {
            MultiCoreCompute();
        }
        if (coreIdx_ == aivNum_) {
            ComputeServerOffset();
        }
    }

private:
    __aicore__ inline void MultiCoreCompute()
    {
        tpipe_->Reset();
        tpipe_->InitBuffer(topkIdxBuf_, topkIdx32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
        tpipe_->InitBuffer(isTokenInRankBuf_, isTokenInRank32AlignIntLen_);
        tpipe_->InitBuffer(localTokenServerUniqCountBuf_, localTokenServerUniqCount32AlignIntLen_);
        tpipe_->InitBuffer(localTokenServerTotalCountBuf_, localTokenServerTotalCount32AlignIntLen_);
        tpipe_->InitBuffer(localTokenServerNumBuf_, localTokenServerNum32AlignIntLen_);
        tpipe_->InitBuffer(seenRankBuf_, numRanks_ * sizeof(T));
        tpipe_->InitBuffer(seenServerBuf_, serverNum_ * sizeof(T));
        LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
        const DataCopyExtParams dataCopyParams{1U, topkIdx32AlignIntLen_, 0U, 0U, 0U};
        const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
        DataCopyPad(topkIdxTensor, topkIdxGM_, dataCopyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        LocalTensor<T> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<T>();
        LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
        LocalTensor<T> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<T>();
        LocalTensor<T> localTokenServerUniqCountTensor = localTokenServerUniqCountBuf_.AllocTensor<T>();
        LocalTensor<T> localTokenServerTotalCountTensor = localTokenServerTotalCountBuf_.AllocTensor<T>();
        LocalTensor<T> localTokenServerNumTensor = localTokenServerNumBuf_.AllocTensor<T>();
        LocalTensor<T> seenRankTensor = seenRankBuf_.AllocTensor<T>();
        LocalTensor<T> seenServerTensor = seenServerBuf_.AllocTensor<T>();
        Duplicate<T>(numTokensPerRankTensor, 0, numRanks_);
        Duplicate<T>(numTokensPerExpertTensor, 0, numExperts_);
        Duplicate<T>(isTokenInRankTensor, 0, tempTokens_ * numRanks_);
        Duplicate<T>(localTokenServerUniqCountTensor, 0, serverNum_);
        Duplicate<T>(localTokenServerTotalCountTensor, 0, tempTokens_ * serverNum_);
        Duplicate<T>(localTokenServerNumTensor, 0, tempTokens_);
        SyncFunc<AscendC::HardEvent::V_S>();
        int experts_per_rank = numExperts_ / numRanks_;
        for (int i = 0; i < tempTokens_; ++i) {
            SyncFunc<AscendC::HardEvent::S_V>();
            Duplicate<T>(seenRankTensor, 0, numRanks_);
            Duplicate<T>(seenServerTensor, 0, serverNum_);
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int j = 0; j < numTopk_; ++j) {
                int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                uint32_t per_expert_num = numTokensPerExpertTensor.GetValue(expert_idx) + 1;
                numTokensPerExpertTensor.SetValue(expert_idx, per_expert_num);
                int rank_id = expert_idx / experts_per_rank;
                int server_id = rank_id / localRankSize_;
                if (!seenServerTensor.GetValue(server_id)) {
                    uint32_t uniqCount = localTokenServerUniqCountTensor.GetValue(server_id);
                    localTokenServerUniqCountTensor.SetValue(server_id, uniqCount + 1);
                    seenServerTensor.SetValue(server_id, 1);
                    uint32_t sendServerNum = localTokenServerNumTensor.GetValue(i);
                    localTokenServerNumTensor.SetValue(i, sendServerNum + 1);
                }
                uint32_t totalCount = localTokenServerTotalCountTensor.GetValue(i * serverNum_ + server_id) + 1;
                localTokenServerTotalCountTensor.SetValue(i * serverNum_ + server_id, totalCount);
                if (!seenRankTensor.GetValue(rank_id)) {
                    uint32_t per_rank_num = numTokensPerRankTensor.GetValue(rank_id) + 1;
                    isTokenInRankTensor.SetValue(i * numRanks_ + rank_id, 1);
                    seenRankTensor.SetValue(rank_id, 1);
                    numTokensPerRankTensor.SetValue(rank_id, per_rank_num);
                }
            }
        }
        uint32_t sendSize = tempTokens_ * numRanks_ * sizeof(T);
        const DataCopyExtParams isTokenInRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
        sendSize = tempTokens_ * sizeof(T);
        DataCopyPad(isTokenInRankGM_, isTokenInRankTensor, isTokenInRankDataCopyParams);
        const DataCopyExtParams localTokenServerNumParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(localTokenServerNumGM_, localTokenServerNumTensor, localTokenServerNumParams);
        sendSize = tempTokens_ * serverNum_ * sizeof(T);
        const DataCopyExtParams localTokenServerTotalCountParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(localTokenServerTotalCountGM_, localTokenServerTotalCountTensor, localTokenServerTotalCountParams);
        sendSize = serverNum_ * sizeof(T);
        AscendC::SetAtomicAdd<T>();
        const DataCopyExtParams localTokenServerUniqCountParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(localTokenServerUniqCountGM_, localTokenServerUniqCountTensor, localTokenServerUniqCountParams);
        const DataCopyExtParams numTokensPerRankDataCopyParams{1U, numTokensPerRank32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
        const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
        DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
        DataCopyPad(numTokensPerExpertSrcGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void ComputeServerOffset()
    {
        tpipe_->Reset();
        tpipe_->InitBuffer(localTokenServerOffsetBuf_, localTokenServerOffset32AlignIntLen_);
        tpipe_->InitBuffer(seenServerBuf_, serverNum_ * sizeof(T));
        tpipe_->InitBuffer(sendTokenIdxBuf_, sendTokenIdx32AlignIntLen_);
        tpipe_->InitBuffer(expertRankTokenIdxBuf_, expertRankTokenIdx32AlignIntLen_);
        tpipe_->InitBuffer(countExpertBuf_, numExperts_ * sizeof(T));
        tpipe_->InitBuffer(countServerBuf_, serverNum_ * sizeof(T));
        LocalTensor<T> localTokenServerOffsetTensor = localTokenServerOffsetBuf_.AllocTensor<T>();
        LocalTensor<T> seenServerTensor = seenServerBuf_.AllocTensor<T>();
        LocalTensor<T> countExpertTensor = countExpertBuf_.AllocTensor<T>();
        LocalTensor<T> countServerTensor = countServerBuf_.AllocTensor<T>();
        LocalTensor<T> expertRankTokenIdxTensor = expertRankTokenIdxBuf_.AllocTensor<T>();
        LocalTensor<T> sendTokenIdxTensor = sendTokenIdxBuf_.AllocTensor<T>();
        SyncFunc<AscendC::HardEvent::S_V>();
        Duplicate<T>(countExpertTensor, 0, numExperts_);
        Duplicate<T>(countServerTensor, 0, serverNum_);
        Duplicate<T>(localTokenServerOffsetTensor, -1, localTokenServerOffset32AlignIntLen_ / sizeof(T));
        SyncFunc<AscendC::HardEvent::V_S>();
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t experts_per_rank = numExperts_ / numRanks_;
        int32_t offset;
        for (int i = 0; i < numTokens_; i++) {
            Duplicate<T>(sendTokenIdxTensor, -1, sendTokenIdx32AlignIntLen_ / sizeof(T));
            Duplicate<T>(seenServerTensor, 0, serverNum_);
            for (int j = 0; j < numTopk_; j++) {
                SyncFunc<AscendC::HardEvent::V_S>();
                int32_t expert_id = topkIdxGM_.GetValue(i * numTopk_ + j);
                int32_t server_id = (expert_id / experts_per_rank) / localRankSize_;
                if (!seenServerTensor.GetValue(server_id)) {
                    offset = countServerTensor.GetValue(server_id);
                    localTokenServerOffsetTensor.SetValue(i * serverNum_ + server_id, offset);
                    countServerTensor.SetValue(server_id, offset + 1);
                    seenServerTensor.SetValue(server_id, 1);
                }
                offset = localTokenServerOffsetTensor.GetValue(i * serverNum_ + server_id);
                int32_t count = countExpertTensor.GetValue(expert_id);
                expertRankTokenIdxTensor.SetValue(expert_id * TEMP_BATCH_SIZE + count % TEMP_BATCH_SIZE, offset);
                sendTokenIdxTensor.SetValue(expert_id, count);
                count++;
                countExpertTensor.SetValue(expert_id, count);
                if (count % TEMP_BATCH_SIZE == 0) {
                    SyncFunc<AscendC::HardEvent::S_MTE3>();
                    const DataCopyExtParams expertRankTokendataCopyParams{1U, TEMP_BATCH_SIZE * sizeof(T), 0U, 0U, 0U};
                    DataCopyPad(expertRankTokenIdxGM_[expert_id * MAX_BATCH_SIZE + count - TEMP_BATCH_SIZE],
                                expertRankTokenIdxTensor[expert_id * TEMP_BATCH_SIZE], expertRankTokendataCopyParams);
                    SyncFunc<AscendC::HardEvent::MTE3_V>();
                    Duplicate(expertRankTokenIdxTensor[expert_id * TEMP_BATCH_SIZE], 0, TEMP_BATCH_SIZE);
                    SyncFunc<AscendC::HardEvent::V_S>();
                }
            }
            const DataCopyExtParams sendTokenIdxdataCopyParams{1U, sendTokenIdx32AlignIntLen_, 0U, 0U, 0U};
            DataCopyPad(sendTokenIdxGM_[i * numExperts_], sendTokenIdxTensor, sendTokenIdxdataCopyParams);
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        uint32_t sendSize = numTokens_ * serverNum_ * sizeof(T);
        const DataCopyExtParams localTokenServerOffsetParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(localTokenServerOffsetGM_, localTokenServerOffsetTensor, localTokenServerOffsetParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        for (int i = 0; i < numExperts_; i++) {
            int32_t count = countExpertTensor.GetValue(i);
            uint32_t rest = count % TEMP_BATCH_SIZE;
            if (rest) {
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                const DataCopyExtParams expertRankTokendataCopyParams{1U, uint32_t(rest * sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(expertRankTokenIdxGM_[i * MAX_BATCH_SIZE + count - rest],
                            expertRankTokenIdxTensor[i * TEMP_BATCH_SIZE], expertRankTokendataCopyParams);
                SyncFunc<AscendC::HardEvent::MTE3_V>();
            }
        }
    }

    GlobalTensor<int64_t> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> numTokensPerExpertSrcGM_;
    GlobalTensor<T> isTokenInRankGM_;
    GlobalTensor<T> localTokenServerOffsetGM_;
    GlobalTensor<T> localTokenServerUniqCountGM_;
    GlobalTensor<T> localTokenServerTotalCountGM_;
    GlobalTensor<T> localTokenServerNumGM_;
    GlobalTensor<T> expertRankTokenIdxGM_;
    GlobalTensor<T> sendTokenIdxGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> localTokenServerOffsetBuf_;
    TBuf<> localTokenServerUniqCountBuf_;
    TBuf<> localTokenServerTotalCountBuf_;
    TBuf<> localTokenServerNumBuf_;
    TBuf<> seenRankBuf_;
    TBuf<> seenServerBuf_;
    TBuf<> countExpertBuf_;
    TBuf<> countServerBuf_;
    TBuf<> sendTokenIdxBuf_;
    TBuf<> expertRankTokenIdxBuf_;

    TPipe *tpipe_{nullptr};
    uint32_t numTokens_{0};
    uint32_t numRanks_{0};
    uint32_t numExperts_{0};
    uint32_t numTopk_{0};
    uint32_t localRankSize_{0};
    uint32_t serverNum_{0};
    uint32_t coreIdx_{0};
    uint32_t aivNum_{0};
    uint32_t tempTokens_{0};

    uint32_t topkIdx32AlignIntLen_{0};
    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
    uint32_t isTokenInRank32AlignIntLen_{0};
    uint32_t localTokenServerOffset32AlignIntLen_{0};
    uint32_t localTokenServerUniqCount32AlignIntLen_{0};
    uint32_t localTokenServerTotalCount32AlignIntLen_{0};
    uint32_t localTokenServerNum32AlignIntLen_{0};
    uint32_t sendTokenIdx32AlignIntLen_{0};
    uint32_t expertRankTokenIdx32AlignIntLen_{0};
};
}  // namespace MoeDispatchLayoutA2

#endif  // DISPATCH_LAYOUT_A2_H
