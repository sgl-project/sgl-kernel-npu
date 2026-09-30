// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Parallel sparse metadata writes for timestamp-LRU refill. The preceding
// victim-selection kernel produces victim_slots and miss_counts on the same
// stream. Each request is divided into 64 independent 32-entry tiles so one
// request can use many AIVs when it contains many misses.
//
// The tile path is pipelined through:
//   1. a two-entry VECIN queue for top-k/victim GM-to-UB prefetch;
//   2. reverse-map MTE2 loads plus a vector Gather;
//   3. a two-entry VECOUT queue holding compact sparse-write records.
// CopyOut launches all MTE3 writes for one tile without waiting for them. The
// queue only stalls when an output buffer must be reused, allowing the next
// tile's MTE2/vector work to overlap the previous tile's sparse GM writes.

#include "kernel_operator.h"

namespace {

constexpr uint32_t kTopk = 2048;
constexpr uint32_t kTileElements = 32;
constexpr uint32_t kTilesPerBatch = kTopk / kTileElements;
constexpr uint32_t kBytesPerInt = sizeof(int32_t);
constexpr uint32_t kScalarBlockElements = 32 / kBytesPerInt;
constexpr uint32_t kSlotLineElements = kScalarBlockElements;
constexpr uint32_t kSlotLineCount = kTileElements + 1;
constexpr uint32_t kSentinelLineOffset = kTileElements * kSlotLineElements;
constexpr uint32_t kSentinelLineOffsetBytes = kSentinelLineOffset * kBytesPerInt;
constexpr uint32_t kBufferNum = 2;

constexpr uint32_t kInputTopkOffset = 0;
constexpr uint32_t kInputVictimOffset = kTileElements;
constexpr uint32_t kInputElements = 2 * kTileElements;

// Four-byte DataCopyPad sources are spaced by one 32-byte block so every
// source address remains DMA aligned.
constexpr uint32_t kWriteStagingElements = kTileElements * kScalarBlockElements;
constexpr uint32_t kWriteNewTokenOffset = 0;
constexpr uint32_t kWriteVictimOffset =
    kWriteNewTokenOffset + kWriteStagingElements;
constexpr uint32_t kWriteOldTokenOffset =
    kWriteVictimOffset + kWriteStagingElements;
constexpr uint32_t kWriteRecordElements =
    kWriteOldTokenOffset + kTileElements;

static_assert(kTopk % kTileElements == 0,
              "top-k must divide into complete metadata tiles");
static_assert((kWriteRecordElements * kBytesPerInt) % 32 == 0,
              "write-record buffers must be 32-byte aligned");

template <AscendC::HardEvent event>
__aicore__ inline void SyncPipes()
{
    const int32_t eventId =
        static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

class KernelParallelLruMetadataWrite
{
public:
    __aicore__ inline KernelParallelLruMetadataWrite() {}

    __aicore__ inline void Init(
        GM_ADDR slotMap, GM_ADDR reqIndices, GM_ADDR topkIndices,
        GM_ADDR victimSlots, GM_ADDR missCounts, GM_ADDR deviceSlotTokens,
        uint32_t batchSize, uint32_t requestRows, uint32_t cacheCapacity,
        uint32_t slotMapWidth, uint32_t maxContextLen, AscendC::TPipe *pipe)
    {
        this->batchSize = batchSize;
        this->requestRows = requestRows;
        this->cacheCapacity = cacheCapacity;
        this->slotMapWidth = slotMapWidth;
        this->maxContextLen = maxContextLen;

        slotMapGm.SetGlobalBuffer(
            (__gm__ int32_t *)slotMap,
            static_cast<uint64_t>(requestRows) * slotMapWidth);
        reqIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)reqIndices, batchSize);
        topkIndicesGm.SetGlobalBuffer(
            (__gm__ int32_t *)topkIndices,
            static_cast<uint64_t>(batchSize) * kTopk);
        victimSlotsGm.SetGlobalBuffer(
            (__gm__ int32_t *)victimSlots,
            static_cast<uint64_t>(batchSize) * kTopk);
        missCountsGm.SetGlobalBuffer(
            (__gm__ int32_t *)missCounts, batchSize);
        deviceSlotTokensGm.SetGlobalBuffer(
            (__gm__ int32_t *)deviceSlotTokens,
            static_cast<uint64_t>(requestRows) * cacheCapacity);

        pipe->InitBuffer(
            inputQueue, kBufferNum, kInputElements * kBytesPerInt);
        pipe->InitBuffer(
            writeQueue, kBufferNum, kWriteRecordElements * kBytesPerInt);
        pipe->InitBuffer(
            oldTokenLineBuf,
            kSlotLineCount * kSlotLineElements * kBytesPerInt);
        pipe->InitBuffer(
            oldTokenOffsetBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(
            oldTokenBuf, kTileElements * kBytesPerInt);
        pipe->InitBuffer(minusOneBuf, 32);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<int32_t> minusOne =
            minusOneBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> oldTokenLines =
            oldTokenLineBuf.Get<int32_t>();
        AscendC::Duplicate(
            minusOne, static_cast<int32_t>(-1), kScalarBlockElements);
        AscendC::Duplicate(
            oldTokenLines[kSentinelLineOffset],
            static_cast<int32_t>(-1), kSlotLineElements);
        SyncPipes<AscendC::HardEvent::V_MTE3>();

        uint32_t inputRequestRows[kBufferNum] = {0, 0};
        uint32_t inputHead = 0;
        uint32_t inputTail = 0;
        uint32_t inputCount = 0;

        uint32_t outputRequestRows[kBufferNum] = {0, 0};
        uint32_t outputWriteCounts[kBufferNum] = {0, 0};
        uint32_t outputHead = 0;
        uint32_t outputTail = 0;
        uint32_t outputCount = 0;

        const uint32_t workerIdx = AscendC::GetBlockIdx();
        const uint32_t workerNum = AscendC::GetBlockNum();
        const uint32_t taskCount = batchSize * kTilesPerBatch;

        for (uint32_t task = workerIdx; task < taskCount;
             task += workerNum) {
            uint32_t requestRow = 0;
            uint32_t gmOffset = 0;
            if (!BuildTileTask(task, requestRow, gmOffset)) {
                continue;
            }

            if (inputCount == kBufferNum) {
                if (outputCount == kBufferNum) {
                    CopyOut(
                        outputRequestRows[outputHead],
                        outputWriteCounts[outputHead], minusOne);
                    outputHead = NextQueueIndex(outputHead);
                    --outputCount;
                }

                outputRequestRows[outputTail] =
                    inputRequestRows[inputHead];
                outputWriteCounts[outputTail] =
                    ComputeTile(inputRequestRows[inputHead]);
                outputTail = NextQueueIndex(outputTail);
                ++outputCount;
                inputHead = NextQueueIndex(inputHead);
                --inputCount;
            }

            CopyIn(gmOffset);
            inputRequestRows[inputTail] = requestRow;
            inputTail = NextQueueIndex(inputTail);
            ++inputCount;
        }

        while (inputCount > 0) {
            if (outputCount == kBufferNum) {
                CopyOut(
                    outputRequestRows[outputHead],
                    outputWriteCounts[outputHead], minusOne);
                outputHead = NextQueueIndex(outputHead);
                --outputCount;
            }

            outputRequestRows[outputTail] =
                inputRequestRows[inputHead];
            outputWriteCounts[outputTail] =
                ComputeTile(inputRequestRows[inputHead]);
            outputTail = NextQueueIndex(outputTail);
            ++outputCount;
            inputHead = NextQueueIndex(inputHead);
            --inputCount;
        }

        while (outputCount > 0) {
            CopyOut(
                outputRequestRows[outputHead],
                outputWriteCounts[outputHead], minusOne);
            outputHead = NextQueueIndex(outputHead);
            --outputCount;
        }
    }

private:
    __aicore__ inline uint32_t NextQueueIndex(uint32_t index) const
    {
        return index == kBufferNum - 1 ? 0 : index + 1;
    }

    __aicore__ inline bool BuildTileTask(
        uint32_t task, uint32_t &requestRow, uint32_t &gmOffset)
    {
        const uint32_t batchIdx = task / kTilesPerBatch;
        const int32_t reqId = reqIndicesGm.GetValue(batchIdx);
        if (reqId < 0 ||
            static_cast<uint32_t>(reqId) >= requestRows ||
            missCountsGm.GetValue(batchIdx) == 0) {
            return false;
        }

        requestRow = static_cast<uint32_t>(reqId);
        const uint32_t tileIdx =
            task - batchIdx * kTilesPerBatch;
        gmOffset =
            batchIdx * kTopk + tileIdx * kTileElements;
        return true;
    }

    __aicore__ inline void CopyIn(uint32_t gmOffset)
    {
        AscendC::LocalTensor<int32_t> input =
            inputQueue.AllocTensor<int32_t>();
        AscendC::DataCopy(
            input[kInputTopkOffset],
            topkIndicesGm[gmOffset], kTileElements);
        AscendC::DataCopy(
            input[kInputVictimOffset],
            victimSlotsGm[gmOffset], kTileElements);
        inputQueue.EnQue(input);
    }

    __aicore__ inline uint32_t ComputeTile(uint32_t requestRow)
    {
        AscendC::LocalTensor<int32_t> input =
            inputQueue.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> topkLocal =
            input[kInputTopkOffset];
        AscendC::LocalTensor<int32_t> victimsLocal =
            input[kInputVictimOffset];
        AscendC::LocalTensor<int32_t> oldTokenLines =
            oldTokenLineBuf.Get<int32_t>();
        AscendC::LocalTensor<uint32_t> oldTokenOffsets =
            oldTokenOffsetBuf.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> oldTokens =
            oldTokenBuf.Get<int32_t>();

        const uint32_t slotTokenRowOffset =
            requestRow * cacheCapacity;
        uint32_t validVictimCount = 0;
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const int32_t victim = victimsLocal.GetValue(i);
            if (victim < 0 ||
                static_cast<uint32_t>(victim) >= cacheCapacity) {
                oldTokenOffsets.SetValue(
                    i, kSentinelLineOffsetBytes);
                continue;
            }

            const uint32_t victimU =
                static_cast<uint32_t>(victim);
            ++validVictimCount;
            const uint32_t gmLineBase =
                victimU & ~(kSlotLineElements - 1U);
            const uint32_t lineLane =
                victimU & (kSlotLineElements - 1U);
            const uint32_t ubLineBase =
                i * kSlotLineElements;
            oldTokenOffsets.SetValue(
                i, (ubLineBase + lineLane) * kBytesPerInt);
            AscendC::DataCopy(
                oldTokenLines[ubLineBase],
                deviceSlotTokensGm[
                    slotTokenRowOffset + gmLineBase],
                kSlotLineElements);
        }

        if (validVictimCount == 0) {
            AscendC::LocalTensor<int32_t> emptyWriteRecord =
                writeQueue.AllocTensor<int32_t>();
            writeQueue.EnQue(emptyWriteRecord);
            inputQueue.FreeTensor(input);
            return 0;
        }

        SyncPipes<AscendC::HardEvent::S_V>();
        SyncPipes<AscendC::HardEvent::MTE2_V>();
        AscendC::Gather(
            oldTokens, oldTokenLines, oldTokenOffsets,
            static_cast<uint32_t>(0), kTileElements);
        AscendC::PipeBarrier<PIPE_V>();
        SyncPipes<AscendC::HardEvent::V_S>();

        // Allocate the next output buffer only after the MTE2/Gather stage.
        // If CopyOut just released a buffer, this lets its outstanding MTE3
        // traffic overlap the current tile before queue reuse can stall here.
        AscendC::LocalTensor<int32_t> writeRecord =
            writeQueue.AllocTensor<int32_t>();
        uint32_t writeCount = 0;
        for (uint32_t i = 0; i < kTileElements; ++i) {
            const int32_t victim = victimsLocal.GetValue(i);
            if (victim < 0 ||
                static_cast<uint32_t>(victim) >= cacheCapacity) {
                continue;
            }

            const int32_t newToken = topkLocal.GetValue(i);
            if (newToken < 0 ||
                static_cast<uint32_t>(newToken) >= maxContextLen) {
                continue;
            }

            const uint32_t stagingIndex =
                writeCount * kScalarBlockElements;
            writeRecord.SetValue(
                kWriteNewTokenOffset + stagingIndex, newToken);
            writeRecord.SetValue(
                kWriteVictimOffset + stagingIndex, victim);
            writeRecord.SetValue(
                kWriteOldTokenOffset + writeCount,
                oldTokens.GetValue(i));
            ++writeCount;
        }

        writeQueue.EnQue(writeRecord);
        inputQueue.FreeTensor(input);
        return writeCount;
    }

    __aicore__ inline void CopyOut(
        uint32_t requestRow, uint32_t writeCount,
        const AscendC::LocalTensor<int32_t> &minusOne)
    {
        AscendC::LocalTensor<int32_t> writeRecord =
            writeQueue.DeQue<int32_t>();
        AscendC::DataCopyExtParams oneIntParams{
            1, sizeof(int32_t), 0, 0, 0};
        const uint32_t slotMapRowOffset =
            requestRow * slotMapWidth;
        const uint32_t slotTokenRowOffset =
            requestRow * cacheCapacity;

        for (uint32_t i = 0; i < writeCount; ++i) {
            const uint32_t stagingIndex =
                i * kScalarBlockElements;
            const int32_t newToken = writeRecord.GetValue(
                kWriteNewTokenOffset + stagingIndex);
            const int32_t victim = writeRecord.GetValue(
                kWriteVictimOffset + stagingIndex);
            const int32_t oldToken = writeRecord.GetValue(
                kWriteOldTokenOffset + i);

            if (oldToken >= 0 &&
                static_cast<uint32_t>(oldToken) < maxContextLen) {
                AscendC::DataCopyPad(
                    slotMapGm[
                        slotMapRowOffset +
                        static_cast<uint32_t>(oldToken)],
                    minusOne, oneIntParams);
            }
            AscendC::DataCopyPad(
                deviceSlotTokensGm[
                    slotTokenRowOffset +
                    static_cast<uint32_t>(victim)],
                writeRecord[
                    kWriteNewTokenOffset + stagingIndex],
                oneIntParams);
            AscendC::DataCopyPad(
                slotMapGm[
                    slotMapRowOffset +
                    static_cast<uint32_t>(newToken)],
                writeRecord[
                    kWriteVictimOffset + stagingIndex],
                oneIntParams);
        }

        writeQueue.FreeTensor(writeRecord);
    }

private:
    AscendC::TQue<
        AscendC::TPosition::VECIN, kBufferNum> inputQueue;
    AscendC::TQue<
        AscendC::TPosition::VECOUT, kBufferNum> writeQueue;
    AscendC::TBuf<
        AscendC::TPosition::VECCALC> oldTokenLineBuf;
    AscendC::TBuf<
        AscendC::TPosition::VECCALC> oldTokenOffsetBuf;
    AscendC::TBuf<
        AscendC::TPosition::VECCALC> oldTokenBuf;
    AscendC::TBuf<
        AscendC::TPosition::VECCALC> minusOneBuf;

    AscendC::GlobalTensor<int32_t> slotMapGm;
    AscendC::GlobalTensor<int32_t> reqIndicesGm;
    AscendC::GlobalTensor<int32_t> topkIndicesGm;
    AscendC::GlobalTensor<int32_t> victimSlotsGm;
    AscendC::GlobalTensor<int32_t> missCountsGm;
    AscendC::GlobalTensor<int32_t> deviceSlotTokensGm;

    uint32_t batchSize = 0;
    uint32_t requestRows = 0;
    uint32_t cacheCapacity = 0;
    uint32_t slotMapWidth = 0;
    uint32_t maxContextLen = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void parallel_lru_metadata_write(
    GM_ADDR slot_map, GM_ADDR req_indices, GM_ADDR topk_indices,
    GM_ADDR victim_slots, GM_ADDR miss_counts,
    GM_ADDR device_slot_tokens, uint32_t batch_size,
    uint32_t request_rows, uint32_t cache_capacity,
    uint32_t slot_map_width, uint32_t max_context_len)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelParallelLruMetadataWrite kernel;
    kernel.Init(
        slot_map, req_indices, topk_indices, victim_slots,
        miss_counts, device_slot_tokens, batch_size,
        request_rows, cache_capacity, slot_map_width, max_context_len, &pipe);
    kernel.Process();
}
