/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#ifndef FIRST_FILL_SCATTER_COPY_KERNEL_H
#define FIRST_FILL_SCATTER_COPY_KERNEL_H

#include "kernel_operator.h"

namespace FirstFillScatterCopyNs {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BLOCK_SHIFT = 7;
constexpr int64_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr int64_t K_ROPE_DIM = 64;
constexpr int64_t KV_CACHE_DIM = 512;
constexpr int64_t COMBINED_TOKEN_STRIDE = KV_CACHE_DIM + K_ROPE_DIM;
constexpr int64_t K_ROPE_UB_BYTES = K_ROPE_DIM * sizeof(uint16_t);
constexpr int64_t KV_CACHE_UB_BYTES = KV_CACHE_DIM * sizeof(uint16_t);

template <typename T, bool COMBINED_LAYOUT = false>
class FirstFillScatterCopyKernel
{
public:
    __aicore__ inline FirstFillScatterCopyKernel(TPipe *pipe,
                                                 const __gm__ FusedScatterCopySparseFlashAttentionTilingData *tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR hbmKRoPE, GM_ADDR hbmKvCache, GM_ADDR dramKRoPE, GM_ADDR dramKvCache,
                                GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR srcTokenIds, GM_ADDR dstSlots,
                                GM_ADDR copyCounts, GM_ADDR qEnds, bool refreshOnly, GM_ADDR current, GM_ADDR positions,
                                GM_ADDR currentValid, bool residentEnabled)
    {
        blockIdx_ = GetBlockIdx();
        refreshOnly_ = refreshOnly;
        residentEnabled_ = residentEnabled;
        splitHost_ = COMBINED_LAYOUT && tiling_->splitHost != 0;
        qEndsGm_.SetGlobalBuffer((__gm__ int32_t *)qEnds);
        if (residentEnabled_) {
            currentGm_.SetGlobalBuffer((__gm__ T *)current);
            positionsGm_.SetGlobalBuffer((__gm__ int64_t *)positions);
            currentValidGm_.SetGlobalBuffer((__gm__ uint8_t *)currentValid);
        }
        batchSize_ = tiling_->baseParams.batchSize;
        copyCap_ = tiling_->missCap;
        totalPairSlots_ = static_cast<int64_t>(batchSize_) * copyCap_;
        usedCoreNum_ = tiling_->singleCoreParams.usedCoreNum * 2U;
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        kRopeUbOffset_ = KV_CACHE_UB_BYTES / sizeof(T);
        // Keep one payload in VECOUT while MTE2 prefetches the next random
        // DRAM row into VECIN.  SCATTER has no vector compute with which to
        // hide either transfer, so a two-slot bound queue is the whole
        // software pipeline.
        pipe_->InitBuffer(copyQueue_, 2, KV_CACHE_UB_BYTES + K_ROPE_UB_BYTES);

        hbmKRoPEGm_.SetGlobalBuffer((__gm__ T *)hbmKRoPE);
        hbmKvCacheGm_.SetGlobalBuffer((__gm__ T *)hbmKvCache);
        dramKRoPEGm_.SetGlobalBuffer((__gm__ T *)dramKRoPE);
        dramKvCacheGm_.SetGlobalBuffer((__gm__ T *)dramKvCache);
        hbmBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)hbmBlockTable);
        dramBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)dramBlockTable);
        srcTokenIdsGm_.SetGlobalBuffer((__gm__ int32_t *)srcTokenIds);
        dstSlotsGm_.SetGlobalBuffer((__gm__ int32_t *)dstSlots);
        copyCountsGm_.SetGlobalBuffer((__gm__ int32_t *)copyCounts);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        cachedBatchIdx_ = -1;
        cachedCopyCount_ = 0;

        int64_t currentFlatPair = FindNextValidPair(blockIdx_);
        CopyAddress currentAddress;
        while (currentFlatPair < totalPairSlots_ && !ResolveAddress(currentFlatPair, currentAddress)) {
            currentFlatPair = FindNextValidPair(currentFlatPair + usedCoreNum_);
        }
        if (currentFlatPair >= totalPairSlots_) {
            return;
        }

        CopyIn(currentAddress);
        while (true) {
            int64_t nextFlatPair = FindNextValidPair(currentFlatPair + usedCoreNum_);
            CopyAddress nextAddress;
            while (nextFlatPair < totalPairSlots_ && !ResolveAddress(nextFlatPair, nextAddress)) {
                nextFlatPair = FindNextValidPair(nextFlatPair + usedCoreNum_);
            }

            const bool hasNext = nextFlatPair < totalPairSlots_;
            if (hasNext) {
                // Issue the next DRAM->UB transfer before draining the current
                // UB->HBM payload.  The two queue slots keep both operations
                // in flight without changing per-core destination ownership.
                CopyIn(nextAddress);
            }
            CopyOut(currentAddress);
            if (!hasNext) {
                break;
            }
            currentFlatPair = nextFlatPair;
            currentAddress = nextAddress;
        }
    }

private:
    struct CopyAddress {
        int64_t srcKv = 0;
        int64_t dstKv = 0;
        int64_t srcRope = 0;
        int64_t dstRope = 0;
        bool fromCurrent = false;
    };

    __aicore__ inline int64_t FirstFlatPairAtOrAfter(int64_t start)
    {
        if (start <= blockIdx_) {
            return blockIdx_;
        }
        int64_t steps = CeilDiv(start - blockIdx_, static_cast<int64_t>(usedCoreNum_));
        return blockIdx_ + steps * usedCoreNum_;
    }

    __aicore__ inline int64_t FindNextValidPair(int64_t flatPairIdx)
    {
        while (flatPairIdx < totalPairSlots_) {
            int64_t batchIdx = flatPairIdx / copyCap_;
            int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * copyCap_);
            if (batchIdx != cachedBatchIdx_) {
                cachedCopyCount_ = copyCountsGm_.GetValue(batchIdx);
                cachedCopyStart_ = 0;
                if (refreshOnly_) {
                    // Native LIM appends at most Q rewritten-resident pairs.
                    // Refreshing the last Q pairs covers that suffix; any extra
                    // ordinary misses are harmlessly copied again by attention.
                    const int32_t qEnd = qEndsGm_.GetValue(batchIdx);
                    const int32_t qStart = batchIdx == 0 ? 0 : qEndsGm_.GetValue(batchIdx - 1);
                    const int32_t qCount = qEnd - qStart;
                    ASSERT_MSG(qCount >= 0 && qCount <= 16, "invalid query count for resident refresh");
                    cachedCopyStart_ = cachedCopyCount_ > qCount ? cachedCopyCount_ - qCount : 0;
                }
                ASSERT_MSG(cachedCopyCount_ >= 0 && cachedCopyCount_ <= copyCap_,
                           "copy_count exceeds the SCATTER input capacity.");
                cachedBatchIdx_ = batchIdx;
            }
            if (copyIdx < cachedCopyStart_) {
                flatPairIdx = FirstFlatPairAtOrAfter(batchIdx * copyCap_ + cachedCopyStart_);
                continue;
            }
            if (copyIdx < cachedCopyCount_) {
                return flatPairIdx;
            }
            flatPairIdx = FirstFlatPairAtOrAfter((batchIdx + 1) * copyCap_);
        }
        return totalPairSlots_;
    }

    __aicore__ inline bool ResolveAddress(int64_t flatPairIdx, CopyAddress &address)
    {
        int64_t batchIdx = flatPairIdx / copyCap_;
        int32_t copyIdx = static_cast<int32_t>(flatPairIdx - batchIdx * copyCap_);
        int64_t pairOffset = batchIdx * copyCap_ + copyIdx;
        int32_t srcTokenId = srcTokenIdsGm_.GetValue(pairOffset);
        int32_t dstSlot = dstSlotsGm_.GetValue(pairOffset);
        ASSERT_MSG(srcTokenId >= 0 && dstSlot >= 0, "active src_token_ids and dst_slots must be non-negative.");
        if (srcTokenId < 0 || dstSlot < 0) {
            return false;
        }

        int64_t srcBlockCol = static_cast<int64_t>(srcTokenId) >> BLOCK_SHIFT;
        int64_t srcBlockOffset = static_cast<int64_t>(srcTokenId) & BLOCK_MASK;
        int64_t dstBlockCol = static_cast<int64_t>(dstSlot) >> BLOCK_SHIFT;
        int64_t dstBlockOffset = static_cast<int64_t>(dstSlot) & BLOCK_MASK;
        ASSERT_MSG(dstBlockCol < tiling_->hbmMaxBlockNum, "destination slot exceeds its block table.");
        if (dstBlockCol >= tiling_->hbmMaxBlockNum) {
            return false;
        }
        int32_t dstPhysicalBlock = hbmBlockTableGm_.GetValue(batchIdx * tiling_->hbmMaxBlockNum + dstBlockCol);
        ASSERT_MSG(dstPhysicalBlock >= 0 &&
                       (!residentEnabled_ || static_cast<uint32_t>(dstPhysicalBlock) < tiling_->residentHbmPages),
                   "destination block table entry is invalid.");
        if (dstPhysicalBlock < 0 ||
            (residentEnabled_ && static_cast<uint32_t>(dstPhysicalBlock) >= tiling_->residentHbmPages)) {
            return false;
        }
        const int64_t dstRow = static_cast<int64_t>(dstPhysicalBlock) * BLOCK_SIZE + dstBlockOffset;
        if (residentEnabled_) {
            const int32_t qStart = batchIdx == 0 ? 0 : qEndsGm_.GetValue(batchIdx - 1);
            const int32_t qEnd = qEndsGm_.GetValue(batchIdx);
            if (qStart >= 0 && qEnd > qStart && qEnd <= tiling_->residentCurrentRows) {
                const int64_t first = positionsGm_.GetValue(qStart);
                const int64_t q = qStart + static_cast<int64_t>(srcTokenId) - first;
                if (q >= qStart && q < qEnd) {
                    ASSERT_MSG(currentValidGm_.GetValue(q) != 0 && positionsGm_.GetValue(q) == srcTokenId,
                               "active current source must be valid and consecutive");
                    address.fromCurrent = true;
                    address.srcKv = q * COMBINED_TOKEN_STRIDE;
                    address.srcRope = address.srcKv + KV_CACHE_DIM;
                }
            }
        }
        int64_t srcRow = 0;
        if (!address.fromCurrent) {
            ASSERT_MSG(srcBlockCol < tiling_->dramMaxBlockNum, "source token exceeds the DRAM block table.");
            if (srcBlockCol >= tiling_->dramMaxBlockNum) return false;
            const int32_t srcPhysicalBlock =
                dramBlockTableGm_.GetValue(batchIdx * tiling_->dramMaxBlockNum + srcBlockCol);
            ASSERT_MSG(srcPhysicalBlock >= 0, "source block table entry must be non-negative.");
            if (srcPhysicalBlock < 0) return false;
            srcRow = static_cast<int64_t>(srcPhysicalBlock) * BLOCK_SIZE + srcBlockOffset;
            ASSERT_MSG(!residentEnabled_ || srcRow < tiling_->residentHostRows,
                       "source physical token exceeds host capacity.");
            if (residentEnabled_ && srcRow >= tiling_->residentHostRows) return false;
        }
        if constexpr (COMBINED_LAYOUT) {
            if (!address.fromCurrent) {
                address.srcKv = srcRow * (splitHost_ ? KV_CACHE_DIM : COMBINED_TOKEN_STRIDE);
                address.srcRope = srcRow * (splitHost_ ? K_ROPE_DIM : COMBINED_TOKEN_STRIDE);
            }
            address.dstKv = dstRow * COMBINED_TOKEN_STRIDE;
            address.dstRope = dstRow * COMBINED_TOKEN_STRIDE;
        } else {
            address.srcKv = srcRow * KV_CACHE_DIM;
            address.dstKv = dstRow * KV_CACHE_DIM;
            address.srcRope = srcRow * K_ROPE_DIM;
            address.dstRope = dstRow * K_ROPE_DIM;
        }
        return true;
    }

    __aicore__ inline void CopyIn(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};

        if (address.fromCurrent) {
            DataCopyPad(local, currentGm_[address.srcKv], kvParams, padParams);
            DataCopyPad(local[kRopeUbOffset_], currentGm_[address.srcRope], ropeParams, padParams);
        } else {
            DataCopyPad(local, dramKvCacheGm_[address.srcKv], kvParams, padParams);
            DataCopyPad(local[kRopeUbOffset_], dramKRoPEGm_[address.srcRope], ropeParams, padParams);
        }
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(const CopyAddress &address)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(KV_CACHE_UB_BYTES), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(K_ROPE_UB_BYTES), 0, 0, 0};

        DataCopyPad(hbmKvCacheGm_[address.dstKv], local, kvParams);
        DataCopyPad(hbmKRoPEGm_[address.dstRope], local[kRopeUbOffset_], ropeParams);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

private:
    TPipe *pipe_;
    const __gm__ FusedScatterCopySparseFlashAttentionTilingData *tiling_;
    int32_t blockIdx_ = -1;
    uint32_t usedCoreNum_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t copyCap_ = 0;
    int64_t totalPairSlots_ = 0;
    int32_t kRopeUbOffset_ = 0;
    int64_t cachedBatchIdx_ = -1;
    int32_t cachedCopyCount_ = 0;
    int32_t cachedCopyStart_ = 0;
    bool refreshOnly_ = false;
    bool residentEnabled_ = false;
    bool splitHost_ = false;
    GlobalTensor<int32_t> qEndsGm_;

    GlobalTensor<T> hbmKRoPEGm_;
    GlobalTensor<T> hbmKvCacheGm_;
    GlobalTensor<T> dramKRoPEGm_;
    GlobalTensor<T> dramKvCacheGm_;
    GlobalTensor<T> currentGm_;
    GlobalTensor<int64_t> positionsGm_;
    GlobalTensor<uint8_t> currentValidGm_;
    GlobalTensor<int32_t> hbmBlockTableGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> srcTokenIdsGm_;
    GlobalTensor<int32_t> dstSlotsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
};

}  // namespace FirstFillScatterCopyNs
#endif  // FIRST_FILL_SCATTER_COPY_KERNEL_H
