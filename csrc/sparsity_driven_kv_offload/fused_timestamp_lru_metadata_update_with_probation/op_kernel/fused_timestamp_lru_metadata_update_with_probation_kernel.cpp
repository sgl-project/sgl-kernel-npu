// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// A2/A3 AIV implementation of timestamp LRU victim selection and state update.
// The kernel intentionally uses no scatter instruction. Dense transforms,
// prefix scan, and indexed reads are SIMD operations. Sparse cache
// metadata writes run in the following parallel_lru_metadata_write kernel.

#include "kernel_operator.h"

namespace {

constexpr uint32_t kBytesPerInt = sizeof(int32_t);
constexpr uint32_t kScalarBlockElements = 32 / kBytesPerInt;

// Vector indices inside the runtime-sized victim-selection scratch arena.
constexpr uint32_t kTopkTokenVector = 0;
constexpr uint32_t kTopkDevicePosVector = 1;
constexpr uint32_t kMissFlagVector = 2;
constexpr uint32_t kScanScratchVector = 3;
constexpr uint32_t kVictimVector = 4;
constexpr uint32_t kGatherOffsetVector = 5;
constexpr uint32_t kVectorScratchVector = 6;

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipes()
{
    const int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

__aicore__ inline void SyncMte2ToVector()
{
    SyncPipes<AscendC::HardEvent::MTE2_V>();
}

__aicore__ inline void SyncMte2ToScalar()
{
    SyncPipes<AscendC::HardEvent::MTE2_S>();
}

__aicore__ inline void SyncVectorToMte2()
{
    SyncPipes<AscendC::HardEvent::V_MTE2>();
}

__aicore__ inline void SyncVectorToMte3()
{
    SyncPipes<AscendC::HardEvent::V_MTE3>();
}

__aicore__ inline void SyncVectorToScalar()
{
    SyncPipes<AscendC::HardEvent::V_S>();
}

__aicore__ inline void SyncScalarToVector()
{
    SyncPipes<AscendC::HardEvent::S_V>();
}

__aicore__ inline void SyncMte3ToScalar()
{
    SyncPipes<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void SyncMte3ToVector()
{
    SyncPipes<AscendC::HardEvent::MTE3_V>();
}

__aicore__ inline void SyncScalarToMte3()
{
    SyncPipes<AscendC::HardEvent::S_MTE3>();
}

template <typename T>
__aicore__ inline void CopyRowIn(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                 uint32_t elementCount)
{
    const uint32_t byteCount =
        elementCount * static_cast<uint32_t>(sizeof(T));
    AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyRowOut(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  uint32_t elementCount)
{
    if (elementCount == 0) {
        return;
    }
    const uint32_t byteCount =
        elementCount * static_cast<uint32_t>(sizeof(T));
    AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

template <uint32_t kCacheCapacity>
class KernelFusedTimestampLruMetadataUpdateWithProbation
{
    // The persistent pairs are already ordered by descending age. Five
    // capacity-sized vectors plus two packed bit masks are sufficient for a
    // fully vectorized stable compaction. After compaction, all vectors above
    // the LRU pair become the seven-vector top-k victim scratch arena.
    static constexpr uint32_t kCompactSlotsOffset = 0;
    static constexpr uint32_t kCompactStampsOffset =
        kCompactSlotsOffset + kCacheCapacity * kBytesPerInt;
    static constexpr uint32_t kCompactPositionMaskOffset =
        kCompactStampsOffset + kCacheCapacity * kBytesPerInt;
    static constexpr uint32_t kCompactGatherOffsetsOffset =
        kCompactPositionMaskOffset + kCacheCapacity * kBytesPerInt;
    static constexpr uint32_t kCompactHitFlagsOffset =
        kCompactGatherOffsetsOffset + kCacheCapacity * kBytesPerInt;
    static constexpr uint32_t kPatternBytes = kCacheCapacity / 8;
    static constexpr uint32_t kPatternWords =
        kPatternBytes / sizeof(uint32_t);
    static constexpr uint32_t kNonHitPatternOffset =
        kCompactHitFlagsOffset + kCacheCapacity * kBytesPerInt;
    static constexpr uint32_t kHitPatternOffset =
        kNonHitPatternOffset + kPatternBytes;
    static constexpr uint32_t kCompactVictimScratchOffset =
        kCompactStampsOffset + kCacheCapacity * kBytesPerInt;
    static_assert(kPatternBytes % 32 == 0,
                  "gather patterns must be UB-aligned");

public:
    __aicore__ inline KernelFusedTimestampLruMetadataUpdateWithProbation() {}

    __aicore__ inline void Init(GM_ADDR reqIndices, GM_ADDR topkIndices,
                                GM_ADDR deviceTokenPos, GM_ADDR hitPositionMask,
                                GM_ADDR deviceLruSlots, GM_ADDR deviceLruSlotStamps,
                                GM_ADDR victimSlots, GM_ADDR missCounts,
                                uint32_t batchSize, uint32_t requestRows,
                                uint32_t topk, uint32_t maxContextLen,
                                uint32_t stampMax, uint32_t probationAge, uint32_t workUbBytes,
                                AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->topk = topk;
        this->maxContextLen = maxContextLen;
        this->stampMax = stampMax;
        this->probationAge = probationAge;

        reqIndicesGm.SetGlobalBuffer((__gm__ int32_t *)reqIndices, batchSize);
        topkIndicesGm.SetGlobalBuffer((__gm__ int32_t *)topkIndices,
                                     static_cast<uint64_t>(batchSize) * topk);
        deviceTokenPosGm.SetGlobalBuffer((__gm__ int32_t *)deviceTokenPos,
                                        static_cast<uint64_t>(batchSize) * topk);
        hitPositionMaskGm.SetGlobalBuffer((__gm__ int32_t *)hitPositionMask,
                                         static_cast<uint64_t>(batchSize) * kCacheCapacity);
        deviceLruSlotsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlots,
                                        static_cast<uint64_t>(requestRows) * kCacheCapacity);
        deviceLruSlotStampsGm.SetGlobalBuffer((__gm__ int32_t *)deviceLruSlotStamps,
                                             static_cast<uint64_t>(requestRows) * kCacheCapacity);
        victimSlotsGm.SetGlobalBuffer((__gm__ int32_t *)victimSlots,
                                     static_cast<uint64_t>(batchSize) * topk);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);

        // The host verifies the selected runtime memory plan against the
        // current platform before launching the kernel.
        pipe->InitBuffer(workBuf, workUbBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreCount = AscendC::GetBlockNum();
        for (uint32_t batchIdx = coreIdx; batchIdx < batchSize; batchIdx += coreCount) {
            ProcessRequest(batchIdx);
        }
    }

private:
    __aicore__ inline uint32_t TopkVectorOffset(uint32_t vectorIndex) const
    {
        return kCompactVictimScratchOffset + vectorIndex * topk * kBytesPerInt;
    }

    __aicore__ inline void ProcessRequest(uint32_t batchIdx)
    {
        AscendC::LocalTensor<int32_t> staging =
            workBuf.GetWithOffset<int32_t>(kScalarBlockElements,
                                           TopkVectorOffset(kVectorScratchVector));
        CopyRowIn(staging, reqIndicesGm[batchIdx], 1);
        SyncMte2ToScalar();
        const int32_t reqId = staging.GetValue(0);
        // Negative sentinel and out-of-range request rows are masked by the
        // caller. Leave their victim output undefined and skip all GM writes.
        if (reqId < 0 || static_cast<uint32_t>(reqId) >= requestRows) {
            return;
        }

        const uint32_t requestRow = static_cast<uint32_t>(reqId);
        BuildUpdatedSlotStampPairs(batchIdx, requestRow);
        const uint32_t missCount = BuildVictimPlan(batchIdx);
        WriteMissCount(batchIdx, missCount);
        WriteLruState(requestRow, missCount);
    }

    // The persistent LRU pairs are already ordered by descending timestamp.
    // Stable-compact non-hits to the front and append hits at age zero. This
    // preserves the old order inside both groups without building float keys,
    // sorting, or allocating a second capacity-sized output pair.
    __aicore__ inline void BuildUpdatedSlotStampPairs(uint32_t batchIdx, uint32_t requestRow)
    {
        AscendC::LocalTensor<int32_t> lruSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactSlotsOffset);
        AscendC::LocalTensor<int32_t> lruStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactStampsOffset);
        AscendC::LocalTensor<int32_t> positionMask =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactPositionMaskOffset);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactGatherOffsetsOffset);
        AscendC::LocalTensor<int32_t> hitFlags =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactHitFlagsOffset);
        AscendC::LocalTensor<uint32_t> nonHitPattern =
            workBuf.GetWithOffset<uint32_t>(kPatternWords, kNonHitPatternOffset);
        AscendC::LocalTensor<uint32_t> hitPattern =
            workBuf.GetWithOffset<uint32_t>(kPatternWords, kHitPatternOffset);

        CopyRowIn(lruSlots, deviceLruSlotsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(lruStamps, deviceLruSlotStampsGm[requestRow * kCacheCapacity], kCacheCapacity);
        CopyRowIn(positionMask, hitPositionMaskGm[batchIdx * kCacheCapacity], kCacheCapacity);
        SyncMte2ToVector();

        // Saturating age increment: min(stamp, stampMax - 1) + 1.
        AscendC::Mins(lruStamps, lruStamps, static_cast<int32_t>(stampMax - 1), kCacheCapacity);
        AscendC::Adds(lruStamps, lruStamps, static_cast<int32_t>(1), kCacheCapacity);

        // Convert physical slots to byte offsets and gather the hit bit in
        // current LRU order. positionMask is guaranteed to contain only 0/1.
        AscendC::Muls(gatherOffsets, lruSlots, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(hitFlags, positionMask, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        // CompareScalar emits one packed bit per LRU entry. GatherMask consumes
        // the same representation and performs a stable vector compaction.
        AscendC::CompareScalar(nonHitPattern.ReinterpretCast<uint8_t>(), hitFlags,
                               static_cast<int32_t>(0), AscendC::CMPMODE::EQ,
                               kCacheCapacity);
        AscendC::CompareScalar(hitPattern.ReinterpretCast<uint8_t>(), hitFlags,
                               static_cast<int32_t>(1), AscendC::CMPMODE::EQ,
                               kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        // Lay the compacted slot groups in one contiguous 2*C source region:
        // positionMask stores non-hits and gatherOffsets stores hits.
        uint64_t nonHitCount64 = 0;
        uint64_t hitCount64 = 0;
        uint64_t stampCount64 = 0;
        AscendC::GatherMask(positionMask, lruSlots, nonHitPattern, true,
                            kCacheCapacity, {1, 1, 0, 0}, nonHitCount64);
        AscendC::GatherMask(gatherOffsets, lruSlots, hitPattern, true,
                            kCacheCapacity, {1, 1, 0, 0}, hitCount64);
        AscendC::GatherMask(hitFlags, lruStamps, nonHitPattern, true,
                            kCacheCapacity, {1, 1, 0, 0}, stampCount64);
        SyncVectorToScalar();
        const uint32_t nonHitCount = static_cast<uint32_t>(nonHitCount64);
        SyncScalarToVector();

        // GatherMask compacted non-hit stamps into hitFlags. Zero its unwritten
        // tail with a vector prefix predicate so hit stamps become MRU age zero.
        AscendC::CreateVecIndex(lruStamps, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Muls(lruStamps, lruStamps, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(lruStamps, lruStamps, static_cast<int32_t>(nonHitCount), kCacheCapacity);
        AscendC::Maxs(lruStamps, lruStamps, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(lruStamps, lruStamps, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Mul(hitFlags, hitFlags, lruStamps, kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        // For output i, read compacted source i for a non-hit and
        // C + (i - nonHitCount) for a hit. The two compact groups are adjacent
        // in UB, so one indexed Gather materializes the final stable order.
        AscendC::CreateVecIndex(lruSlots, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Adds(lruStamps, lruSlots,
                      static_cast<int32_t>(1) - static_cast<int32_t>(nonHitCount),
                      kCacheCapacity);
        AscendC::Maxs(lruStamps, lruStamps, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(lruStamps, lruStamps, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Muls(lruStamps, lruStamps,
                      static_cast<int32_t>(kCacheCapacity - nonHitCount),
                      kCacheCapacity);
        AscendC::Add(lruStamps, lruStamps, lruSlots, kCacheCapacity);
        AscendC::Muls(lruStamps, lruStamps, static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(lruSlots, positionMask, lruStamps.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(lruStamps, hitFlags, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint32_t BuildVictimPlan(uint32_t batchIdx)
    {
        AscendC::LocalTensor<int32_t> topkTokens =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kTopkTokenVector));
        AscendC::LocalTensor<int32_t> devicePos =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kTopkDevicePosVector));
        AscendC::LocalTensor<int32_t> missFlag =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kMissFlagVector));
        AscendC::LocalTensor<int32_t> scanScratch =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kScanScratchVector));
        AscendC::LocalTensor<int32_t> victims =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kVictimVector));
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kGatherOffsetVector));
        AscendC::LocalTensor<int32_t> vectorScratch =
            workBuf.GetWithOffset<int32_t>(topk, TopkVectorOffset(kVectorScratchVector));
        AscendC::LocalTensor<int32_t> lruSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactSlotsOffset);

        // The top-k input buffers reuse the stage-A positionMask and
        // gatherOffsets regions. Ensure the final stage-A Gather has finished
        // reading those regions before MTE2 starts overwriting them.
        SyncVectorToMte2();
        CopyRowIn(topkTokens, topkIndicesGm[batchIdx * topk], topk);
        CopyRowIn(devicePos, deviceTokenPosGm[batchIdx * topk], topk);
        SyncMte2ToVector();

        // missFlag = valid_topk && (device_pos == -1), expressed entirely as
        // clamped int32 SIMD arithmetic.
        AscendC::Adds(missFlag, devicePos, static_cast<int32_t>(1), topk);
        AscendC::Maxs(missFlag, missFlag, static_cast<int32_t>(0), topk);
        AscendC::Mins(missFlag, missFlag, static_cast<int32_t>(1), topk);
        AscendC::Duplicate(scanScratch, static_cast<int32_t>(1), topk);
        AscendC::Sub(missFlag, scanScratch, missFlag, topk);

        AscendC::Adds(scanScratch, topkTokens, static_cast<int32_t>(1), topk);
        AscendC::Maxs(scanScratch, scanScratch, static_cast<int32_t>(0), topk);
        AscendC::Mins(scanScratch, scanScratch, static_cast<int32_t>(1), topk);
        AscendC::Duplicate(vectorScratch, static_cast<int32_t>(maxContextLen), topk);
        AscendC::Sub(vectorScratch, vectorScratch, topkTokens, topk);
        AscendC::Maxs(vectorScratch, vectorScratch, static_cast<int32_t>(0), topk);
        AscendC::Mins(vectorScratch, vectorScratch, static_cast<int32_t>(1), topk);
        AscendC::Min(scanScratch, scanScratch, vectorScratch, topk);
        AscendC::Min(missFlag, missFlag, scanScratch, topk);
        AscendC::PipeBarrier<PIPE_V>();

        // Hillis-Steele inclusive scan. Gather supplies a zero-padded shifted
        // vector on platforms where scatter is unavailable.
        for (uint32_t step = 1; step < topk; step <<= 1) {
            AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), topk);
            AscendC::Adds(gatherOffsets, gatherOffsets, -static_cast<int32_t>(step), topk);
            AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), topk);
            AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), topk);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(scanScratch, missFlag, gatherOffsets.ReinterpretCast<uint32_t>(),
                            static_cast<uint32_t>(0), topk);
            AscendC::Duplicate(scanScratch, static_cast<int32_t>(0), step);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(missFlag, missFlag, scanScratch, topk);
            AscendC::PipeBarrier<PIPE_V>();
        }

        SyncVectorToScalar();
        const uint32_t missCount = static_cast<uint32_t>(missFlag.GetValue(topk - 1));

        // Gather the r-th oldest eligible slot for the r-th miss, then restore
        // -1 in hit/invalid positions without a scatter operation.
        AscendC::Adds(gatherOffsets, missFlag, static_cast<int32_t>(-1), topk);
        AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), topk);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), topk);
        AscendC::Gather(victims, lruSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), topk);
        AscendC::PipeBarrier<PIPE_V>();

        // Recover the original 0/1 miss vector as scan[i] - scan[i-1].
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), topk);
        AscendC::Adds(gatherOffsets, gatherOffsets, static_cast<int32_t>(-1), topk);
        AscendC::Maxs(gatherOffsets, gatherOffsets, static_cast<int32_t>(0), topk);
        AscendC::Muls(gatherOffsets, gatherOffsets, static_cast<int32_t>(kBytesPerInt), topk);
        AscendC::Gather(scanScratch, missFlag, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), topk);
        AscendC::Duplicate(scanScratch, static_cast<int32_t>(0), 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scanScratch, missFlag, scanScratch, topk);
        AscendC::Adds(victims, victims, static_cast<int32_t>(1), topk);
        AscendC::Mul(victims, victims, scanScratch, topk);
        AscendC::Adds(victims, victims, static_cast<int32_t>(-1), topk);
        AscendC::PipeBarrier<PIPE_V>();

        SyncVectorToMte3();
        CopyRowOut(victimSlotsGm[batchIdx * topk], victims, topk);
        return missCount;
    }

    __aicore__ inline void WriteMissCount(uint32_t batchIdx, uint32_t missCount)
    {
        AscendC::LocalTensor<int32_t> staging =
            workBuf.GetWithOffset<int32_t>(kScalarBlockElements,
                                           TopkVectorOffset(kVectorScratchVector));
        SyncVectorToScalar();
        staging.SetValue(0, static_cast<int32_t>(missCount));
        SyncScalarToMte3();
        AscendC::DataCopyExtParams oneIntParams{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(missCountsGm[batchIdx], staging, oneIntParams);
    }

    __aicore__ inline void WriteLruState(uint32_t requestRow, uint32_t missCount)
    {
        AscendC::LocalTensor<int32_t> lruSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactSlotsOffset);
        AscendC::LocalTensor<int32_t> lruStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactStampsOffset);
        AscendC::LocalTensor<int32_t> gatherOffsets =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity, kCompactVictimScratchOffset);
        AscendC::LocalTensor<int32_t> rotatedSlots =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity,
                                           kCompactVictimScratchOffset +
                                               kCacheCapacity * kBytesPerInt);
        AscendC::LocalTensor<int32_t> rotatedStamps =
            workBuf.GetWithOffset<int32_t>(kCacheCapacity,
                                           kCompactVictimScratchOffset +
                                               2 * kCacheCapacity * kBytesPerInt);

        const uint32_t rowOffset = requestRow * kCacheCapacity;
        if (missCount == 0) {
            SyncVectorToMte3();
            CopyRowOut(deviceLruSlotsGm[rowOffset], lruSlots, kCacheCapacity);
            CopyRowOut(deviceLruSlotStampsGm[rowOffset], lruStamps, kCacheCapacity);
            SyncMte3ToScalar();
            return;
        }

        // rotatedStamps overlaps the victim-output source buffer. Drain that
        // MTE3 read before the vector pipeline reuses the shared UB region.
        SyncMte3ToVector();

        uint32_t insertionIndex = kCacheCapacity - missCount;
        if (probationAge > 0) {
            // The non-victim suffix is already sorted by descending age. Find
            // the first entry younger than probationAge; existing entries with
            // the same age remain before the newly filled victims.
            SyncVectorToScalar();
            uint32_t low = missCount;
            uint32_t high = kCacheCapacity;
            while (low < high) {
                const uint32_t mid = low + ((high - low) >> 1);
                if (lruStamps.GetValue(mid) >= static_cast<int32_t>(probationAge)) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            insertionIndex = low - missCount;
            SyncScalarToVector();
        }

        // Victim slots occupy the prefix. Give them probationAge, then build
        // the stable insertion permutation entirely with vector arithmetic:
        //   src = i + (i < insertionIndex) * missCount
        //           - (insertionIndex <= i < insertionIndex + missCount)
        //             * insertionIndex.
        AscendC::Duplicate(lruStamps, static_cast<int32_t>(probationAge), missCount);
        AscendC::CreateVecIndex(gatherOffsets, static_cast<int32_t>(0), kCacheCapacity);

        AscendC::Muls(rotatedSlots, gatherOffsets, static_cast<int32_t>(-1), kCacheCapacity);
        AscendC::Adds(rotatedSlots, rotatedSlots,
                      static_cast<int32_t>(insertionIndex), kCacheCapacity);
        AscendC::Maxs(rotatedSlots, rotatedSlots, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(rotatedSlots, rotatedSlots, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Muls(rotatedSlots, rotatedSlots,
                      static_cast<int32_t>(missCount), kCacheCapacity);
        AscendC::Add(gatherOffsets, gatherOffsets, rotatedSlots, kCacheCapacity);

        AscendC::CreateVecIndex(rotatedSlots,
                                static_cast<int32_t>(1) - static_cast<int32_t>(insertionIndex),
                                kCacheCapacity);
        AscendC::Maxs(rotatedSlots, rotatedSlots, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(rotatedSlots, rotatedSlots, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::CreateVecIndex(
            rotatedStamps,
            static_cast<int32_t>(1) - static_cast<int32_t>(insertionIndex + missCount),
            kCacheCapacity);
        AscendC::Maxs(rotatedStamps, rotatedStamps, static_cast<int32_t>(0), kCacheCapacity);
        AscendC::Mins(rotatedStamps, rotatedStamps, static_cast<int32_t>(1), kCacheCapacity);
        AscendC::Sub(rotatedSlots, rotatedSlots, rotatedStamps, kCacheCapacity);
        AscendC::Muls(rotatedSlots, rotatedSlots,
                      static_cast<int32_t>(insertionIndex), kCacheCapacity);
        AscendC::Sub(gatherOffsets, gatherOffsets, rotatedSlots, kCacheCapacity);
        AscendC::Muls(gatherOffsets, gatherOffsets,
                      static_cast<int32_t>(kBytesPerInt), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(rotatedSlots, lruSlots, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::Gather(rotatedStamps, lruStamps, gatherOffsets.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), kCacheCapacity);
        AscendC::PipeBarrier<PIPE_V>();
        SyncVectorToMte3();

        CopyRowOut(deviceLruSlotsGm[rowOffset], rotatedSlots, kCacheCapacity);
        CopyRowOut(deviceLruSlotStampsGm[rowOffset], rotatedStamps, kCacheCapacity);
        // Drain both full-row writes before this core reuses the shared UB for
        // the next grid-stride request or exits the kernel.
        SyncMte3ToScalar();
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> deviceTokenPosGm;
    AscendC::GlobalTensor<int32_t> hitPositionMaskGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotsGm;
    AscendC::GlobalTensor<int32_t> deviceLruSlotStampsGm;
    AscendC::GlobalTensor<int32_t> victimSlotsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;

    uint32_t batchSize = 0;
    uint32_t requestRows = 0;
    uint32_t topk = 0;
    uint32_t maxContextLen = 0;
    uint32_t stampMax = 0;
    uint32_t probationAge = 0;
};

template <uint32_t kCacheCapacity>
__aicore__ inline void RunFusedTimestampLruMetadataUpdateWithProbation(
    GM_ADDR reqIndices, GM_ADDR topkIndices, GM_ADDR deviceTokenPos,
    GM_ADDR hitPositionMask, GM_ADDR deviceLruSlots,
    GM_ADDR deviceLruSlotStamps, GM_ADDR victimSlots, GM_ADDR missCounts,
    uint32_t batchSize, uint32_t requestRows, uint32_t topk, uint32_t maxContextLen,
    uint32_t stampMax, uint32_t probationAge, uint32_t workUbBytes,
    AscendC::TPipe *pipe)
{
    KernelFusedTimestampLruMetadataUpdateWithProbation<kCacheCapacity> kernel;
    kernel.Init(reqIndices, topkIndices, deviceTokenPos, hitPositionMask,
                deviceLruSlots, deviceLruSlotStamps, victimSlots, missCounts,
                batchSize, requestRows, topk, maxContextLen, stampMax, probationAge,
                workUbBytes, pipe);
    kernel.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fused_timestamp_lru_metadata_update_with_probation(
    GM_ADDR req_indices, GM_ADDR topk_indices, GM_ADDR device_token_pos,
    GM_ADDR hit_position_mask, GM_ADDR device_lru_slots, GM_ADDR device_lru_slot_stamps,
    GM_ADDR victim_slots, GM_ADDR miss_counts, uint32_t batch_size, uint32_t request_rows,
    uint32_t cache_capacity, uint32_t topk, uint32_t max_context_len, uint32_t stamp_max,
    uint32_t probation_age, uint32_t work_ub_bytes)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (cache_capacity == 2048) {
        RunFusedTimestampLruMetadataUpdateWithProbation<2048>(
            req_indices, topk_indices, device_token_pos, hit_position_mask,
            device_lru_slots, device_lru_slot_stamps, victim_slots, miss_counts,
            batch_size, request_rows, topk, max_context_len, stamp_max, probation_age,
            work_ub_bytes, &pipe);
    } else if (cache_capacity == 4096) {
        RunFusedTimestampLruMetadataUpdateWithProbation<4096>(
            req_indices, topk_indices, device_token_pos, hit_position_mask,
            device_lru_slots, device_lru_slot_stamps, victim_slots, miss_counts,
            batch_size, request_rows, topk, max_context_len, stamp_max, probation_age,
            work_ub_bytes, &pipe);
    } else if (cache_capacity == 6144) {
        RunFusedTimestampLruMetadataUpdateWithProbation<6144>(
            req_indices, topk_indices, device_token_pos, hit_position_mask,
            device_lru_slots, device_lru_slot_stamps, victim_slots, miss_counts,
            batch_size, request_rows, topk, max_context_len, stamp_max, probation_age,
            work_ub_bytes, &pipe);
    } else if (cache_capacity == 8192) {
        RunFusedTimestampLruMetadataUpdateWithProbation<8192>(
            req_indices, topk_indices, device_token_pos, hit_position_mask,
            device_lru_slots, device_lru_slot_stamps, victim_slots, miss_counts,
            batch_size, request_rows, topk, max_context_len, stamp_max, probation_age,
            work_ub_bytes, &pipe);
    }
}
