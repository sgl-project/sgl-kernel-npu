/**
 * @file unidex_copy_kernel.cpp
 * @brief Indexed row-copy kernel for Ascend AI Cores.
 *
 * For each valid, in-range mapping i:
 *   dst[dst_index[i]] = src[src_index[i]]
 *
 * The source and destination are treated as byte-addressed row buffers. Each
 * logical row occupies blockBytes bytes. Mapping entries are partitioned into
 * contiguous ranges across the launched AI Cores.
 *
 * The kernel copies one complete row through UB at a time. A two-entry queue
 * pipelines GM-to-UB and UB-to-GM transfers. Invalid mask entries, negative
 * indices, and out-of-range indices are skipped without reporting an error.
 *
 * The host wrapper limits blockBytes to 32 KiB and validates that the addressed
 * source and destination extents fit in the kernel's uint32_t offset range.
 * Byte-wise copying keeps the kernel independent of the tensor element type.
 */

#include "kernel_operator.h"

using CopyUnit = uint8_t;
constexpr uint32_t BUFFER_NUM = 2;

class KernelUniDexCopy
{
public:
    __aicore__ inline KernelUniDexCopy() {}

    /**
     * @brief Initialize global buffers, per-core work, and the UB queue.
     *
     * @param src Base address of the source byte buffer in GM.
     * @param dst Base address of the destination byte buffer in GM.
     * @param src_index Source row indices in GM (int64_t, maxCopy entries).
     * @param dst_index Destination row indices in GM (int64_t, maxCopy entries).
     * @param valid_mask Mapping validity flags in GM (bool/uint8_t, maxCopy entries).
     * @param srcRows Number of logical source rows.
     * @param dstRows Number of logical destination rows.
     * @param blockBytes Number of bytes in each source and destination row.
     * @param maxCopy Number of mapping entries to process.
     * @param pipeIn Pipeline used to initialize the copy queue.
     *
     * Each AI Core handles:
     * [blockIdx * copyRowsPerCore,
     *  min(maxCopy, (blockIdx + 1) * copyRowsPerCore)).
     */
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, GM_ADDR src_index, GM_ADDR dst_index, GM_ADDR valid_mask,
                                uint32_t srcRows, uint32_t dstRows, uint32_t blockBytes, uint32_t maxCopy,
                                AscendC::TPipe *pipeIn)
    {
        this->srcRows = srcRows;
        this->dstRows = dstRows;
        this->blockBytes = blockBytes;
        this->maxCopy = maxCopy;
        this->pipe = pipeIn;

        const uint32_t blockNum = AscendC::GetBlockNum();
        this->copyRowsPerCore = (maxCopy + blockNum - 1) / blockNum;

        srcGm.SetGlobalBuffer((__gm__ CopyUnit *)src, srcRows * blockBytes);
        dstGm.SetGlobalBuffer((__gm__ CopyUnit *)dst, dstRows * blockBytes);
        srcIndexGm.SetGlobalBuffer((__gm__ int64_t *)src_index, maxCopy);
        dstIndexGm.SetGlobalBuffer((__gm__ int64_t *)dst_index, maxCopy);
        validMaskGm.SetGlobalBuffer((__gm__ uint8_t *)valid_mask, maxCopy);

        this->alignedBlockBytes = (blockBytes + 31U) & ~31U;
        pipe->InitBuffer(copyQue, BUFFER_NUM, alignedBlockBytes * sizeof(CopyUnit));
    }

    /**
     * @brief Process the mapping range assigned to the current AI Core.
     *
     * Each mapping is checked before its source row is enqueued. When the
     * two-entry queue is full, the oldest row is dequeued and copied to its
     * destination before another source row is enqueued. Remaining rows are
     * flushed after the mapping loop.
     *
     * Work is partitioned by mapping count rather than valid-entry count, so
     * sparse masks can produce uneven useful work across cores.
     */
    __aicore__ inline void Process()
    {
        if (blockBytes == 0) {
            return;
        }

        const uint32_t coreBegin = AscendC::GetBlockIdx() * copyRowsPerCore;
        uint32_t coreEnd = coreBegin + copyRowsPerCore;
        if (coreEnd > maxCopy) {
            coreEnd = maxCopy;
        }

        uint32_t dstOffsets[BUFFER_NUM] = {0, 0};
        uint32_t queueHead = 0;
        uint32_t queueTail = 0;
        uint32_t queued = 0;

        for (uint32_t i = coreBegin; i < coreEnd; ++i) {
            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            if (!BuildCopyTask(i, srcOffset, dstOffset)) {
                continue;
            }

            if (queued == BUFFER_NUM) {
                CopyOut(dstOffsets[queueHead]);
                queueHead = NextQueueIndex(queueHead);
                --queued;
            }

            CopyIn(srcOffset);
            dstOffsets[queueTail] = dstOffset;
            queueTail = NextQueueIndex(queueTail);
            ++queued;
        }

        while (queued > 0) {
            CopyOut(dstOffsets[queueHead]);
            queueHead = NextQueueIndex(queueHead);
            --queued;
        }
    }

private:
    /**
     * @brief Advance a circular queue index.
     * @param index Current queue index.
     * @return The next index, wrapping to zero after BUFFER_NUM - 1.
     */
    __aicore__ inline uint32_t NextQueueIndex(uint32_t index) const
    {
        return index == BUFFER_NUM - 1 ? 0 : index + 1;
    }

    /**
     * @brief Validate one mapping and compute its byte offsets.
     *
     * @param mapIdx Index into src_index, dst_index, and valid_mask.
     * @param[out] srcOffset Source byte offset (src row * blockBytes).
     * @param[out] dstOffset Destination byte offset (dst row * blockBytes).
     * @return true for a valid in-range mapping; false otherwise.
     *
     * A mapping is accepted only when its validity flag is nonzero and both
     * row indices are non-negative and within their respective row counts.
     */
    __aicore__ inline bool BuildCopyTask(uint32_t mapIdx, uint32_t &srcOffset, uint32_t &dstOffset)
    {
        if (validMaskGm.GetValue(mapIdx) == 0) {
            return false;
        }

        const int64_t srcRow = srcIndexGm.GetValue(mapIdx);
        const int64_t dstRow = dstIndexGm.GetValue(mapIdx);
        if (srcRow < 0 || dstRow < 0) {
            return false;
        }
        if (srcRow >= static_cast<int64_t>(srcRows) || dstRow >= static_cast<int64_t>(dstRows)) {
            return false;
        }

        srcOffset = static_cast<uint32_t>(srcRow) * blockBytes;
        dstOffset = static_cast<uint32_t>(dstRow) * blockBytes;
        return true;
    }

    /**
     * @brief Copy one source row from GM to UB and enqueue it.
     * @param srcOffset Source byte offset in GM.
     *
     * Each queue allocation reserves blockBytes rounded up to a multiple of
     * 32 bytes.
     */
    __aicore__ inline void CopyIn(uint32_t srcOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.AllocTensor<CopyUnit>();
        AscendC::DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<CopyUnit> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, srcGm[srcOffset], copyParams, padParams);
        copyQue.EnQue(local);
    }

    /**
     * @brief Dequeue one row, copy it from UB to GM, and free its buffer.
     * @param dstOffset Destination byte offset in GM.
     */
    __aicore__ inline void CopyOut(uint32_t dstOffset)
    {
        AscendC::LocalTensor<CopyUnit> local = copyQue.DeQue<CopyUnit>();
        AscendC::DataCopyExtParams copyParams{1, blockBytes, 0, 0, 0};
        AscendC::DataCopyPad(dstGm[dstOffset], local, copyParams);
        copyQue.FreeTensor(local);
    }

private:
    AscendC::TPipe *pipe = nullptr;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> copyQue;

    AscendC::GlobalTensor<CopyUnit> srcGm;
    AscendC::GlobalTensor<CopyUnit> dstGm;
    AscendC::GlobalTensor<int64_t> srcIndexGm;
    AscendC::GlobalTensor<int64_t> dstIndexGm;
    AscendC::GlobalTensor<uint8_t> validMaskGm;

    uint32_t srcRows = 0;
    uint32_t dstRows = 0;
    uint32_t blockBytes = 0;
    uint32_t alignedBlockBytes = 0;
    uint32_t maxCopy = 0;
    uint32_t copyRowsPerCore = 0;
};

/**
 * @brief Kernel entry point launched by the host through aclrtLaunch.
 *
 * @param src Source byte buffer in GM.
 * @param dst Destination byte buffer in GM.
 * @param src_index Source row indices in GM.
 * @param dst_index Destination row indices in GM.
 * @param valid_mask Mapping validity flags in GM.
 * @param srcRows Number of source rows.
 * @param dstRows Number of destination rows.
 * @param blockBytes Number of bytes per row.
 * @param maxCopy Number of mapping entries.
 *
 * The launch block dimension determines the value returned by GetBlockNum()
 * and therefore the mapping range assigned to each AI Core.
 */
extern "C" __global__ __aicore__ void unidex_copy(GM_ADDR src, GM_ADDR dst, GM_ADDR src_index, GM_ADDR dst_index,
                                                  GM_ADDR valid_mask, uint32_t srcRows, uint32_t dstRows,
                                                  uint32_t blockBytes, uint32_t maxCopy)
{
    AscendC::TPipe pipe;
    KernelUniDexCopy kernel;
    kernel.Init(src, dst, src_index, dst_index, valid_mask, srcRows, dstRows, blockBytes, maxCopy, &pipe);
    kernel.Process();
}
