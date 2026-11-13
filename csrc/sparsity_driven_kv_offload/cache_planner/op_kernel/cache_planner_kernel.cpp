#include "kernel_operator.h"

constexpr uint32_t kCapacity = 2048;
constexpr uint32_t kJoinLength = 2 * kCapacity;
constexpr uint32_t kOldTag = 0x80000000U;

class KernelPlanSparseKvCache
{
public:
    __aicore__ inline void Init(GM_ADDR sourceToSlot, GM_ADDR slotToSource, GM_ADDR reqIds, GM_ADDR topkInput,
                                GM_ADDR topkSrcOut, GM_ADDR topkDstOut, GM_ADDR topkMissCounts, GM_ADDR missSrcOut,
                                GM_ADDR missDstOut, GM_ADDR missCounts, uint32_t realReqCount, uint32_t maxContextLen,
                                uint32_t sourceStride, uint32_t cacheCapacity, uint32_t batchSize, AscendC::TPipe *pipe)
    {
        this->realReqCount = realReqCount;
        this->maxContextLen = maxContextLen;
        this->sourceStride = sourceStride;
        this->cacheCapacity = cacheCapacity;
        this->batchSize = batchSize;
        sourceToSlotGm.SetGlobalBuffer((__gm__ int32_t *)sourceToSlot,
                                       static_cast<uint64_t>(realReqCount + 1) * sourceStride);
        slotToSourceGm.SetGlobalBuffer((__gm__ int32_t *)slotToSource,
                                       static_cast<uint64_t>(realReqCount + 1) * cacheCapacity);
        reqIdsGm.SetGlobalBuffer((__gm__ int32_t *)reqIds, batchSize);
        topkInputGm.SetGlobalBuffer((__gm__ int32_t *)topkInput, static_cast<uint64_t>(batchSize) * cacheCapacity);
        topkSrcOutGm.SetGlobalBuffer((__gm__ int32_t *)topkSrcOut, static_cast<uint64_t>(batchSize) * cacheCapacity);
        topkDstOutGm.SetGlobalBuffer((__gm__ int32_t *)topkDstOut, static_cast<uint64_t>(batchSize) * cacheCapacity);
        topkMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts, batchSize);
        missSrcOutGm.SetGlobalBuffer((__gm__ int32_t *)missSrcOut, static_cast<uint64_t>(batchSize) * cacheCapacity);
        missDstOutGm.SetGlobalBuffer((__gm__ int32_t *)missDstOut, static_cast<uint64_t>(batchSize) * cacheCapacity);
        missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
        pipe->InitBuffer(keysBuf, kJoinLength * sizeof(float));
        pipe->InitBuffer(payloadBuf, kJoinLength * sizeof(uint32_t));
        pipe->InitBuffer(sortedBuf, kJoinLength * 2 * sizeof(float));
        pipe->InitBuffer(tmpBuf, kJoinLength * 2 * sizeof(float));
        pipe->InitBuffer(missSourcesBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(hitSourcesBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(hitSlotsBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(freeSlotsBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(countStagingBuf, 32);
        pipe->InitBuffer(topkRowBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(oldOwnersBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(topkSrcOutBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(topkDstOutBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(missSrcOutBuf, kCapacity * sizeof(int32_t));
        pipe->InitBuffer(missDstOutBuf, kCapacity * sizeof(int32_t));
        mte2ToScalarEvent = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::MTE2_S));
        scalarToVectorEvent = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::S_V));
        vectorToScalarEvent = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::V_S));
        scalarToMte3Event = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::S_MTE3));
        mte3ToScalarEvent = static_cast<int32_t>(pipe->FetchEventID(AscendC::HardEvent::MTE3_S));
    }

    __aicore__ inline void WriteCount(AscendC::GlobalTensor<int32_t> &dst, uint32_t index, int32_t value)
    {
        auto staging = countStagingBuf.Get<int32_t>();
        staging.SetValue(0, value);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
        const AscendC::DataCopyExtParams copyParams{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(dst[index], staging, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
    }

    __aicore__ inline void Process()
    {
        auto keys = keysBuf.Get<float>();
        auto payload = payloadBuf.Get<uint32_t>();
        auto sorted = sortedBuf.Get<float>();
        auto tmp = tmpBuf.Get<float>();
        auto missSources = missSourcesBuf.Get<int32_t>();
        auto hitSources = hitSourcesBuf.Get<int32_t>();
        auto hitSlots = hitSlotsBuf.Get<int32_t>();
        auto freeSlots = freeSlotsBuf.Get<int32_t>();
        auto topkRow = topkRowBuf.Get<int32_t>();
        auto oldOwners = oldOwnersBuf.Get<int32_t>();
        auto topkSrcOut = topkSrcOutBuf.Get<int32_t>();
        auto topkDstOut = topkDstOutBuf.Get<int32_t>();
        auto missSrcOut = missSrcOutBuf.Get<int32_t>();
        auto missDstOut = missDstOutBuf.Get<int32_t>();
        for (uint32_t b = AscendC::GetBlockIdx(); b < batchSize; b += AscendC::GetBlockNum()) {
            const int32_t reqId = reqIdsGm.GetValue(b);
            const uint64_t outBase = static_cast<uint64_t>(b) * cacheCapacity;
            // Initialize all output tails in UB. Scalar writes below replace only
            // the populated prefix, so every unused entry remains a sentinel.
            AscendC::Duplicate(topkSrcOut, static_cast<int32_t>(-1), cacheCapacity);
            AscendC::Duplicate(topkDstOut, static_cast<int32_t>(-1), cacheCapacity);
            AscendC::Duplicate(missSrcOut, static_cast<int32_t>(-1), cacheCapacity);
            AscendC::Duplicate(missDstOut, static_cast<int32_t>(-1), cacheCapacity);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent);
            // Invalid/padded requests return before touching either state map.
            if (reqId < 0 || static_cast<uint32_t>(reqId) >= realReqCount) {
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
                AscendC::DataCopy(topkSrcOutGm[outBase], topkSrcOut, kCapacity);
                AscendC::DataCopy(topkDstOutGm[outBase], topkDstOut, kCapacity);
                AscendC::DataCopy(missSrcOutGm[outBase], missSrcOut, kCapacity);
                AscendC::DataCopy(missDstOutGm[outBase], missDstOut, kCapacity);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
                WriteCount(topkMissCountsGm, b, 0);
                WriteCount(missCountsGm, b, 0);
                continue;
            }

            const uint64_t sourceBase = static_cast<uint64_t>(reqId) * sourceStride;
            const uint64_t inverseBase = static_cast<uint64_t>(reqId) * cacheCapacity;
            AscendC::DataCopy(topkRow, topkInputGm[outBase], kCapacity);
            AscendC::DataCopy(oldOwners, slotToSourceGm[inverseBase], kCapacity);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(mte2ToScalarEvent);
            // Sort-join K new source IDs with K current slot owners. The key
            // transform makes descending full sort produce ascending source
            // IDs; keys are exact FP32 integers under the host-validated bound.
            for (uint32_t i = 0; i < cacheCapacity; ++i) {
                const int32_t src = topkRow.GetValue(i);
                const uint32_t owner = static_cast<uint32_t>(oldOwners.GetValue(i));
                const float newKey = (src >= 0 && static_cast<uint32_t>(src) < maxContextLen)
                                         ? static_cast<float>(static_cast<int32_t>(maxContextLen) - src)
                                         : 0.0f;
                // Give each invalid old slot a distinct negative key. This
                // keeps all free slots after valid sources and orders them by
                // slot ID despite Sort not promising a stable order for ties.
                const float oldKey = (owner < maxContextLen)
                                         ? static_cast<float>(static_cast<int32_t>(maxContextLen - owner))
                                         : -static_cast<float>(static_cast<int32_t>(i + 1));
                keys.SetValue(i, newKey);
                payload.SetValue(i, i);
                keys.SetValue(kCapacity + i, oldKey);
                payload.SetValue(kCapacity + i, kOldTag | i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVectorEvent);
            AscendC::Sort<float, true>(sorted, keys, payload, tmp, kJoinLength / 32);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalarEvent);

            uint32_t missCount = 0;
            uint32_t hitCount = 0;
            uint32_t freeCount = 0;
            // Sorted entries are (key,payload) pairs. A valid equal-key old/new
            // pair is a hit; unmatched new IDs are misses; unmatched old owners
            // are safe destinations, including slots whose owner is invalid.
            const auto sortedBits = sorted.ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < kJoinLength;) {
                const float key = sorted.GetValue(2 * i);
                const uint32_t tag = sortedBits.GetValue(2 * i + 1);
                uint32_t runEnd = i + 1;
                while (runEnd < kJoinLength && sorted.GetValue(2 * runEnd) == key)
                    ++runEnd;
                uint32_t newIndex = kCapacity;
                uint32_t oldSlot = kCapacity;
                for (uint32_t j = i; j < runEnd; ++j) {
                    const uint32_t item = sortedBits.GetValue(2 * j + 1);
                    if ((item & kOldTag) != 0)
                        oldSlot = item & ~kOldTag;
                    else
                        newIndex = item;
                }
                if (key > 0.0f) {
                    if (newIndex < kCapacity && oldSlot < kCapacity) {
                        const int32_t src = static_cast<int32_t>(maxContextLen) - static_cast<int32_t>(key);
                        hitSources.SetValue(hitCount, src);
                        hitSlots.SetValue(hitCount++, static_cast<int32_t>(oldSlot));
                    } else if (newIndex < kCapacity) {
                        missSources.SetValue(missCount++, topkRow.GetValue(newIndex));
                    } else if (oldSlot < kCapacity) {
                        freeSlots.SetValue(freeCount++, static_cast<int32_t>(oldSlot));
                    }
                } else if (key < 0.0f) {
                    // Every invalid old payload is an independently emitted
                    // free slot; negative slot keys preserve lowest-slot order.
                    const uint32_t item = sortedBits.GetValue(2 * i + 1);
                    freeSlots.SetValue(freeCount++, static_cast<int32_t>(item & ~kOldTag));
                }
                i = runEnd;
            }
            // At fixed K=capacity, unmatched old slots have exactly one entry
            // per miss. Use the sorted deterministic pairing without a scan.
            for (uint32_t i = 0; i < missCount; ++i) {
                const int32_t src = missSources.GetValue(i);
                const int32_t slot = (i < freeCount) ? freeSlots.GetValue(i) : -1;
                missSrcOut.SetValue(i, src);
                missDstOut.SetValue(i, slot);
                topkSrcOut.SetValue(i, src);
                topkDstOut.SetValue(i, slot);
                if (slot >= 0) {
                    const int32_t oldSrc = oldOwners.GetValue(static_cast<uint32_t>(slot));
                    if (oldSrc >= 0 && static_cast<uint32_t>(oldSrc) < maxContextLen &&
                        sourceToSlotGm.GetValue(sourceBase + static_cast<uint32_t>(oldSrc)) == slot)
                        sourceToSlotGm.SetValue(sourceBase + static_cast<uint32_t>(oldSrc), -1);
                    sourceToSlotGm.SetValue(sourceBase + static_cast<uint32_t>(src), slot);
                    oldOwners.SetValue(static_cast<uint32_t>(slot), src);
                }
            }
            for (uint32_t i = 0; i < hitCount; ++i) {
                topkSrcOut.SetValue(missCount + i, hitSources.GetValue(i));
                topkDstOut.SetValue(missCount + i, hitSlots.GetValue(i));
            }
            // Bulk copy updated maps and outputs after all scalar UB writes.
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(scalarToMte3Event);
            AscendC::DataCopy(slotToSourceGm[inverseBase], oldOwners, kCapacity);
            AscendC::DataCopy(topkSrcOutGm[outBase], topkSrcOut, kCapacity);
            AscendC::DataCopy(topkDstOutGm[outBase], topkDstOut, kCapacity);
            AscendC::DataCopy(missSrcOutGm[outBase], missSrcOut, kCapacity);
            AscendC::DataCopy(missDstOutGm[outBase], missDstOut, kCapacity);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3ToScalarEvent);
            WriteCount(topkMissCountsGm, b, static_cast<int32_t>(missCount));
            WriteCount(missCountsGm, b, static_cast<int32_t>(missCount));
        }
    }

private:
    AscendC::GlobalTensor<int32_t> sourceToSlotGm, slotToSourceGm, reqIdsGm, topkInputGm;
    AscendC::GlobalTensor<int32_t> topkSrcOutGm, topkDstOutGm, topkMissCountsGm;
    AscendC::GlobalTensor<int32_t> missSrcOutGm, missDstOutGm, missCountsGm;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> keysBuf, payloadBuf, sortedBuf, tmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> missSourcesBuf, hitSourcesBuf, hitSlotsBuf, freeSlotsBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topkRowBuf, oldOwnersBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> topkSrcOutBuf, topkDstOutBuf, missSrcOutBuf, missDstOutBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> countStagingBuf;
    int32_t mte2ToScalarEvent = 0;
    int32_t scalarToVectorEvent = 0, vectorToScalarEvent = 0;
    int32_t scalarToMte3Event = 0, mte3ToScalarEvent = 0;
    uint32_t realReqCount = 0, maxContextLen = 0, sourceStride = 0, cacheCapacity = 0, batchSize = 0;
};

extern "C" __global__ __aicore__ void plan_sparse_kv_cache(
    GM_ADDR sourceToSlot, GM_ADDR slotToSource, GM_ADDR reqIds, GM_ADDR topkInput, GM_ADDR topkSrcOut,
    GM_ADDR topkDstOut, GM_ADDR topkMissCounts, GM_ADDR missSrcOut, GM_ADDR missDstOut, GM_ADDR missCounts,
    uint32_t realReqCount, uint32_t maxContextLen, uint32_t sourceStride, uint32_t cacheCapacity, uint32_t batchSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelPlanSparseKvCache kernel;
    AscendC::TPipe pipe;
    kernel.Init(sourceToSlot, slotToSource, reqIds, topkInput, topkSrcOut, topkDstOut, topkMissCounts, missSrcOut,
                missDstOut, missCounts, realReqCount, maxContextLen, sourceStride, cacheCapacity, batchSize, &pipe);
    kernel.Process();
}
