#ifndef DISPATCH_LAYOUT_H
#define DISPATCH_LAYOUT_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "sync_collectives.h"
#include "moe_distribute_base.h"
#include "dispatch_layout_tiling.h"
namespace MoeDispatchLayout {

constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t ONE_PIECE = 8U;

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
class DispatchLayout
{
public:
    __aicore__ inline DispatchLayout(){};

    __aicore__ inline void Init(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                GM_ADDR isTokenInRank, GM_ADDR notifySendData, GM_ADDR sendTokenIdxSmall,
                                GM_ADDR tokenIdxMap, GM_ADDR validBs, GM_ADDR workspace, TPipe *pipe,
                                const DispatchLayoutTilingData *tilingData)
    {
        numTokens_ = tilingData->dispatchLayoutInfo.numTokens;
        numRanks_ = tilingData->dispatchLayoutInfo.numRanks;
        numExperts_ = tilingData->dispatchLayoutInfo.numExperts;
        numTopk_ = tilingData->dispatchLayoutInfo.numTopk;
        perRoundTokens_ = tilingData->dispatchLayoutInfo.perRoundTokens;
        rankId_ = tilingData->dispatchLayoutInfo.rankId;
        round_ = (numTokens_ + perRoundTokens_ - 1) / perRoundTokens_;
        tpipe_ = pipe;

        topkIdx_ = topkIdx;
        numTokensPerExpert_ = numTokensPerExpert;
        isTokenInRank_ = isTokenInRank;
        sendTokenIdxSmall_ = sendTokenIdxSmall;
        tokenIdxMap_ = tokenIdxMap;
        validBs_ = validBs;

        uint32_t maxAivNum = GetBlockNum();
        coreIdx_ = GetBlockIdx();
        int32_t firstRoundTokens = (round_ == 1) ? numTokens_ : perRoundTokens_;
        aivNum_ = firstRoundTokens <= maxAivNum ? firstRoundTokens : maxAivNum;
        tempTokens_ = firstRoundTokens / aivNum_;
        int32_t restNum = firstRoundTokens % aivNum_;
        if (coreIdx_ < restNum) {
            ++tempTokens_;
        }
        topkIdx32AlignIntLen_ = Ceil(tempTokens_ * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerRank32AlignIntLen_ = Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerExpert32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        onePiece32AlignIntLen_ = Ceil(ONE_PIECE * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        isTokenInRank32AlignIntLen_ = Ceil(tempTokens_ * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        tokenIdx32AlignIntLen_ = Ceil(tempTokens_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

        if (coreIdx_ < restNum) {
            topkIdxOffset_ = coreIdx_ * tempTokens_ * numTopk_ * sizeof(int64_t);
            sendIdxOffset_ = coreIdx_ * tempTokens_ * numTopk_ * sizeof(T);
            isTokenOffset_ = coreIdx_ * tempTokens_ * numRanks_ * sizeof(T);
            tokenIdxOffset_ = coreIdx_ * tempTokens_ * sizeof(T);
        } else {
            topkIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(int64_t);
            sendIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(T);
            isTokenOffset_ = (restNum + coreIdx_ * tempTokens_) * numRanks_ * sizeof(T);
            tokenIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * sizeof(T);
        }

        tempExpertGM_.SetGlobalBuffer((__gm__ T *)notifySendData);
        onePieceGM_.SetGlobalBuffer((__gm__ T *)(notifySendData + numTokensPerExpert32AlignIntLen_ * aivNum_));
        numTokensPerRankGM_.SetGlobalBuffer((__gm__ T *)numTokensPerRank);
        validBsGM_.SetGlobalBuffer((__gm__ T *)validBs);
    }

    __aicore__ inline void Process()
    {
        tpipe_->Reset();
        tpipe_->InitBuffer(topkIdxBuf_, topkIdx32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
        tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
        tpipe_->InitBuffer(isTokenInRankBuf_, isTokenInRank32AlignIntLen_);
        tpipe_->InitBuffer(seenRankBuf_, numRanks_ * sizeof(T));
        tpipe_->InitBuffer(sendTokenIdxSmallBuf_, topkIdx32AlignIntLen_);
        tpipe_->InitBuffer(tokenIdxMapBuf_, tokenIdx32AlignIntLen_);
        tpipe_->InitBuffer(validBsBuf_, 1 * sizeof(T));
        tpipe_->InitBuffer(onePieceBuf_, ONE_PIECE * sizeof(T));
        LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
        LocalTensor<T> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<T>();
        LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
        LocalTensor<T> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<T>();
        LocalTensor<T> seenRankTensor = seenRankBuf_.AllocTensor<T>();
        LocalTensor<T> sendTokenIdxSmallTensor = sendTokenIdxSmallBuf_.AllocTensor<T>();
        LocalTensor<T> tokenIdxMapTensor = tokenIdxMapBuf_.AllocTensor<T>();
        LocalTensor<T> validBsTensor = validBsBuf_.AllocTensor<T>();
        LocalTensor<T> onePieceTensor = onePieceBuf_.AllocTensor<T>();

        int32_t validCount = 0;
        int32_t preRoundCount = 0;
        for (int r = 0; r < round_; r++) {
            uint32_t roundTokens = perRoundTokens_;
            if (r == round_ - 1 && (numTokens_ % perRoundTokens_ != 0)) {
                roundTokens = numTokens_ % perRoundTokens_;
                uint32_t temp = roundTokens / aivNum_;
                uint32_t restNum = roundTokens % aivNum_;
                tempTokens_ = temp;
                if (coreIdx_ < restNum) {
                    tempTokens_++;
                }
                topkIdx32AlignIntLen_ = Ceil(tempTokens_ * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
                isTokenInRank32AlignIntLen_ = Ceil(tempTokens_ * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
                tokenIdx32AlignIntLen_ = Ceil(tempTokens_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

                if (coreIdx_ < restNum) {
                    topkIdxOffset_ = coreIdx_ * tempTokens_ * numTopk_ * sizeof(int64_t);
                    sendIdxOffset_ = coreIdx_ * tempTokens_ * numTopk_ * sizeof(T);
                    isTokenOffset_ = coreIdx_ * tempTokens_ * numRanks_ * sizeof(T);
                    tokenIdxOffset_ = coreIdx_ * tempTokens_ * sizeof(T);
                } else {
                    topkIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(int64_t);
                    sendIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(T);
                    isTokenOffset_ = (restNum + coreIdx_ * tempTokens_) * numRanks_ * sizeof(T);
                    tokenIdxOffset_ = (restNum + coreIdx_ * tempTokens_) * sizeof(T);
                }
            }

            uint32_t maxAivNum = GetBlockNum();
            aivNum_ = roundTokens <= maxAivNum ? roundTokens : maxAivNum;
            if (coreIdx_ >= aivNum_) {
                SyncAll<true>();
                SyncAll<true>();
                continue;
            }

            int64_t round_topkIdx_offset = r * perRoundTokens_ * numTopk_ * sizeof(int64_t);
            int64_t round_sendIdx_offset = r * perRoundTokens_ * numTopk_ * sizeof(T);

            sendTokenIdxSmallGM_.SetGlobalBuffer(
                (__gm__ T *)(sendTokenIdxSmall_ + round_sendIdx_offset + sendIdxOffset_));
            topkIdxGM_.SetGlobalBuffer((__gm__ int64_t *)(topkIdx_ + round_topkIdx_offset + topkIdxOffset_));
            numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T *)(numTokensPerExpert_ + numExperts_ * r * sizeof(T)));
            // tokens * rank;
            isTokenInRankGM_.SetGlobalBuffer(
                (__gm__ T *)(isTokenInRank_ + r * perRoundTokens_ * numRanks_ * sizeof(T) + isTokenOffset_));
            tokenIdxMapGM_.SetGlobalBuffer(
                (__gm__ T *)(tokenIdxMap_ + r * perRoundTokens_ * sizeof(T) + tokenIdxOffset_));

            const DataCopyExtParams dataCopyParams{1U, topkIdx32AlignIntLen_, 0U, 0U, 0U};
            const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopyPad(topkIdxTensor, topkIdxGM_, dataCopyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            SyncFunc<AscendC::HardEvent::MTE3_V>();
            Duplicate<T>(numTokensPerRankTensor, 0, numTokensPerRank32AlignIntLen_ / sizeof(T));
            Duplicate<T>(isTokenInRankTensor, 0, isTokenInRank32AlignIntLen_ / sizeof(T));
            Duplicate<T>(numTokensPerExpertTensor, 0, numTokensPerExpert32AlignIntLen_ / sizeof(T));
            Duplicate<T>(tokenIdxMapTensor, -1, tokenIdx32AlignIntLen_ / sizeof(T));
            Duplicate<T>(onePieceTensor, 0, onePiece32AlignIntLen_ / sizeof(T));
            if (r > 0) {
                SyncFunc<AscendC::HardEvent::V_S>();
                preRoundsCount_ += preRoundCount;
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                const DataCopyExtParams clearGmParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
                DataCopyPad(tempExpertGM_[coreIdx_ * numExperts_], numTokensPerExpertTensor, clearGmParams);
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                const DataCopyExtParams clearOnePieceParams{1U, onePiece32AlignIntLen_, 0U, 0U, 0U};
                DataCopyPad(onePieceGM_[(coreIdx_ + 1) * ONE_PIECE], onePieceTensor, clearOnePieceParams);
                PipeBarrier<PIPE_MTE3>();
            }
            SyncAll<true>();

            int experts_per_rank = numExperts_ / numRanks_;
            int32_t tempACount = 0;
            for (int i = 0; i < tempTokens_; ++i) {
                int32_t validFlag = 0;
                SyncFunc<AscendC::HardEvent::S_V>();
                Duplicate<T>(seenRankTensor, 0, numRanks_);
                SyncFunc<AscendC::HardEvent::V_S>();
                for (int j = 0; j < numTopk_; ++j) {
                    int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                    if (expert_idx < 0 || expert_idx >= numExperts_) {
                        continue;
                    }
                    validFlag = 1;
                    uint32_t per_expert_num = numTokensPerExpertTensor.GetValue(expert_idx) + 1;
                    numTokensPerExpertTensor.SetValue(expert_idx, per_expert_num);
                    int rank_id = expert_idx / experts_per_rank;
                    if (!seenRankTensor.GetValue(rank_id)) {
                        uint32_t per_rank_num = numTokensPerRankTensor.GetValue(rank_id) + 1;
                        isTokenInRankTensor.SetValue(i * numRanks_ + rank_id, 1);
                        seenRankTensor.SetValue(rank_id, 1);
                        numTokensPerRankTensor.SetValue(rank_id, per_rank_num);
                    }
                }
                tempACount = validFlag ? tempACount + 1 : tempACount;
            }
            onePieceTensor.SetValue(0, tempACount);
            uint32_t sendSize = tempTokens_ * numRanks_ * sizeof(T);
            const DataCopyExtParams isTokenInRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(isTokenInRankGM_, isTokenInRankTensor, isTokenInRankDataCopyParams);
            AscendC::SetAtomicAdd<T>();
            const DataCopyExtParams tempExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
            const DataCopyExtParams onePieceDataCopyParams{1U, onePiece32AlignIntLen_, 0U, 0U, 0U};
            for (int i = coreIdx_ + 1; i < aivNum_ + 1; ++i) {
                if (i < aivNum_) {
                    DataCopyPad(tempExpertGM_[i * numExperts_], numTokensPerExpertTensor, tempExpertDataCopyParams);
                }
                DataCopyPad(onePieceGM_[i * ONE_PIECE], onePieceTensor, onePieceDataCopyParams);
            }
            sendSize = numRanks_ * sizeof(T);
            const DataCopyExtParams numTokensPerRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
            sendSize = numExperts_ * sizeof(T);
            const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
            AscendC::SetAtomicNone();
            PipeBarrier<PIPE_MTE3>();
            SyncAll<true>();
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            const DataCopyPadExtParams<T> tempPadParams{false, 0U, 0U, 0U};
            DataCopyPad(numTokensPerExpertTensor, tempExpertGM_[coreIdx_ * numExperts_], tempExpertDataCopyParams,
                        tempPadParams);
            DataCopyPad(onePieceTensor, onePieceGM_[coreIdx_ * ONE_PIECE], onePieceDataCopyParams, tempPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            int32_t preCount = onePieceTensor.GetValue(0);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            DataCopyPad(onePieceTensor, onePieceGM_[aivNum_ * ONE_PIECE], onePieceDataCopyParams, tempPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            preRoundCount = onePieceTensor.GetValue(0);
            int32_t tempBCount = 0;
            for (int i = 0; i < tempTokens_; ++i) {
                int32_t validFlag = 0;
                for (int j = 0; j < numTopk_; ++j) {
                    int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                    if (expert_idx < 0 || expert_idx > numExperts_) {
                        continue;
                    }
                    validFlag = 1;
                    T valT = numTokensPerExpertTensor(expert_idx);
                    sendTokenIdxSmallTensor(i * numTopk_ + j) = valT;
                    numTokensPerExpertTensor(expert_idx) = valT + 1;
                }
                if (validFlag) {
                    int32_t validOffset = preRoundsCount_ + preCount + tempBCount;
                    tokenIdxMapTensor.SetValue(i, validOffset);
                    ++tempBCount;
                } else {
                    tokenIdxMapTensor.SetValue(i, -1);
                }
            }
            validCount += tempBCount;
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            const DataCopyExtParams sendTokenIdxSmallDataCopyParams{
                1U, static_cast<uint32_t>(tempTokens_ * numTopk_ * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(sendTokenIdxSmallGM_, sendTokenIdxSmallTensor, sendTokenIdxSmallDataCopyParams);
            sendSize = tempTokens_ * sizeof(T);
            const DataCopyExtParams tokenIdxMapDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(tokenIdxMapGM_, tokenIdxMapTensor, tokenIdxMapDataCopyParams);
        }
        validBsTensor.SetValue(0, validCount);
        AscendC::SetAtomicAdd<T>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t sendSize = 1 * sizeof(T);
        const DataCopyExtParams validBsDataCopyParams{1U, sendSize, 0U, 0U, 0U};
        DataCopyPad(validBsGM_, validBsTensor, validBsDataCopyParams);
        AscendC::SetAtomicNone();
    }

private:
    GlobalTensor<int64_t> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> isTokenInRankGM_;
    GlobalTensor<T> tokenIdxMapGM_;
    GlobalTensor<T> validBsGM_;
    GlobalTensor<T> tempExpertGM_;
    GlobalTensor<T> onePieceGM_;
    GlobalTensor<T> sendTokenIdxSmallGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> seenRankBuf_;
    TBuf<> sendTokenIdxSmallBuf_;
    TBuf<> tokenIdxMapBuf_;
    TBuf<> validBsBuf_;
    TBuf<> onePieceBuf_;

    TPipe *tpipe_{nullptr};
    uint32_t numTokens_{0};
    uint32_t numRanks_{0};
    uint32_t numExperts_{0};
    uint32_t numTopk_{0};
    uint32_t coreIdx_{0};
    uint32_t aivNum_{0};
    uint32_t tempTokens_{0};
    uint32_t round_{0};
    uint32_t rankId_{0};
    uint32_t perRoundTokens_{0};
    uint32_t preRoundsCount_{0};
    int64_t topkIdxOffset_{0};
    int64_t sendIdxOffset_{0};
    int64_t isTokenOffset_{0};
    int64_t tokenIdxOffset_{0};

    uint32_t topkIdx32AlignIntLen_{0};
    uint32_t tokenIdx32AlignIntLen_{0};
    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
    uint32_t onePiece32AlignIntLen_{0};
    uint32_t isTokenInRank32AlignIntLen_{0};

    GM_ADDR topkIdx_;
    GM_ADDR numTokensPerExpert_;
    GM_ADDR isTokenInRank_;
    GM_ADDR sendTokenIdxSmall_;
    GM_ADDR tokenIdxMap_;
    GM_ADDR validBs_;
};
}  // namespace MoeDispatchLayout

#endif  // DISPATCH_LAYOUT_H
