// BSD 3-Clause License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

#include "kernel_operator.h"
#include "state_exchange_tiling.h"

#define STATE_EXCHANGE_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

// Same UB double-buffer budget as the KV sparse-copy kernel: one entry is
// copied through an 88KB UB slot, entries larger than that are split.
constexpr int64_t STATE_EXCHANGE_UB_BUFFER_SIZE = 88 * 1024;
// DataCopyExtParams::blockLen is a uint16_t, so one DataCopy moves at most
// 65535 bytes; chunk splits at the largest 32-byte aligned value below it.
// The 88KB UB slot still hosts each chunk with room to spare.
constexpr int64_t STATE_EXCHANGE_COPY_CHUNK = 65504;
using optiling::STATE_EXCHANGE_MAX_ENTRIES;
using optiling::STATE_EXCHANGE_META_STRIDE;

// Transfer description from CANN tiling data:
//   num_entries (<= 16): one entry per normalized state component pair,
//       where a pair is device [layers, slots, *payload] /
//       host [slots, layers, 1, *payload] (layer/slot-narrowed views allowed;
//       base pointer and strides carry the offsets).
//   indices_len / direction: shared 1-D int64 slot indices (device memory),
//       direction 1 = H2D, 2 = D2H.
//   start_dim: first participating AIV core index. < 0 selects the
//       acc_offload occupancy (blockDim - 2, i.e. only the last 2 AIV cores
//       copy, keeping the rest free for concurrent compute); 0 lets every
//       launched core participate (perf-sweep override).
//   per entry e (8 metadata fields):
//   [0] device_base  [1] host_base (device-visible address of the host page,
//       i.e. DVA produced by aclrtHostRegister / MF registration)
//   [2] device_slot_pitch  [3] device_layer_pitch   (bytes)
//   [4] host_slot_pitch    [5] host_layer_pitch     (bytes)
//   [6] width (bytes of one (slot, layer) payload row)  [7] layer_count
// Row addressing: device_base + layer*dlp + slot*dsp /
// host_base + slot*hsp + layer*hlp.

// start_dim < 0: reserve the last 2 AIV cores (native acc_offload occupancy).
constexpr int32_t STATE_EXCHANGE_START_DIM_DEFAULT = -1;

/*
 * Fused address-computation + copy kernel for the Mamba state exchange
 * (layer-first device pool <-> page-first host pool).  No (src, dst, len)
 * entry table is built on the host: every AIV core derives the addresses of
 * the (entry, layer, slot) rows it owns directly from the device-resident
 * slot indices and the per-entry layout metadata, so the indices never
 * round-trip through the CPU.
 *
 * Work distribution: the (entry, layer) pairs form `total_units` row groups,
 * each covering `indices_len` slot rows.  Every core walks all units
 * sequentially and round-robins the slot rows within a unit
 * (row += aiv_num), which balances the same total work as a flat global
 * row round-robin without host-side prefix sums.  The copy pipeline is the
 * proven kv_exchange two-stage UB double buffering (CopyIn -> EnQue ->
 * DeQue -> CopyOut); rows wider than the UB buffer are split.
 */
class StateExchangeKernel
{
public:
    STATE_EXCHANGE_AICORE_KERNEL StateExchangeKernel() {}

    // Use the transfer description resolved by CANN tiling.
    STATE_EXCHANGE_AICORE_KERNEL void
    Init(AscendC::TPipe *pipe, __gm__ int64_t *deviceIdx, __gm__ int64_t *hostIdx, int64_t numEntries,
         int64_t indicesLen, int64_t direction, int64_t startDim,
         const int64_t entries[STATE_EXCHANGE_MAX_ENTRIES][STATE_EXCHANGE_META_STRIDE])
    {
        pipe_ = pipe;
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();

        // Active-core window: cores [startDim, blockDim) participate. A
        // negative startDim selects the native acc_offload occupancy --
        // reserve the last 2 AIV cores for the copy so the rest stay free
        // for concurrent compute; 0 lets every launched core participate
        // (perf-sweep override).
        startDim_ = startDim;
        if (startDim_ < 0) {
            startDim_ = aivNum_ > 2 ? static_cast<int64_t>(aivNum_) - 2 : 0;
        }
        if (startDim_ >= static_cast<int64_t>(aivNum_) || aivIndex_ < startDim_) {
            skip_ = true;
            return;
        }

        aivIndex_ = aivIndex_ - static_cast<uint32_t>(startDim_);
        aivNum_ = aivNum_ - static_cast<uint32_t>(startDim_);

        numEntries_ = static_cast<int32_t>(numEntries);
        if (numEntries_ > STATE_EXCHANGE_MAX_ENTRIES) {
            numEntries_ = STATE_EXCHANGE_MAX_ENTRIES;
        }
        indicesLen_ = indicesLen;
        direction_ = direction;
        deviceIdxGm_.SetGlobalBuffer(deviceIdx, static_cast<uint32_t>(indicesLen_));
        hostIdxGm_.SetGlobalBuffer(hostIdx, static_cast<uint32_t>(indicesLen_));

        int64_t units = 0;
        for (int32_t e = 0; e < numEntries_; ++e) {
            entries_[e].deviceBase = entries[e][0];
            entries_[e].hostBase = entries[e][1];
            entries_[e].deviceSlotPitch = entries[e][2];
            entries_[e].deviceLayerPitch = entries[e][3];
            entries_[e].hostSlotPitch = entries[e][4];
            entries_[e].hostLayerPitch = entries[e][5];
            entries_[e].width = entries[e][6];
            entries_[e].layerCount = entries[e][7];
            entryFirst_[e] = units;
            units += entries_[e].layerCount;
        }
        totalUnits_ = units;

        pipe_->InitBuffer(bindQueue_, 2, STATE_EXCHANGE_UB_BUFFER_SIZE);
    }

    STATE_EXCHANGE_AICORE_KERNEL void Process()
    {
        if (skip_ || indicesLen_ <= 0 || numEntries_ <= 0 || totalUnits_ <= 0) {
            return;
        }
        RowWalker walker(this);
        RunPipeline(walker);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    struct Entry {
        int64_t deviceBase = 0;
        int64_t hostBase = 0;
        int64_t deviceSlotPitch = 0;
        int64_t deviceLayerPitch = 0;
        int64_t hostSlotPitch = 0;
        int64_t hostLayerPitch = 0;
        int64_t width = 0;
        int64_t layerCount = 0;
    };

    struct EntryCtx {
        uint64_t src = 0;
        uint64_t dst = 0;
        uint32_t bytes = 0;
    };

    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    STATE_EXCHANGE_AICORE_KERNEL void ResolveUnit(int64_t unit, int32_t &entryIdx, int64_t &layer) const
    {
        int32_t e = 0;
        while (e + 1 < numEntries_ && entryFirst_[e + 1] <= unit) {
            ++e;
        }
        entryIdx = e;
        layer = unit - entryFirst_[e];
    }

    STATE_EXCHANGE_AICORE_KERNEL void CopyIn(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<uint8_t> local = bindQueue_.AllocTensor<uint8_t>();
        AscendC::GlobalTensor<uint8_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(ctx.src), ctx.bytes);
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPadExtParams<uint8_t> padParams{};
        AscendC::DataCopyPad(local, srcGm, copyParams, padParams);
        bindQueue_.EnQue(local);
    }

    STATE_EXCHANGE_AICORE_KERNEL void CopyOut(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<uint8_t> local = bindQueue_.DeQue<uint8_t>();
        AscendC::GlobalTensor<uint8_t> dstGm;
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(ctx.dst), ctx.bytes);
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPad(dstGm, local, copyParams);
        bindQueue_.FreeTensor(local);
    }

    // Build the copy entry for the (unit, slot row, split) position; the entry
    // is split when the remaining width exceeds the UB buffer.
    STATE_EXCHANGE_AICORE_KERNEL bool MakeEntry(EntryCtx &ctx, int64_t unit, int64_t idx, int64_t offset) const
    {
        int32_t entryIdx = 0;
        int64_t layer = 0;
        ResolveUnit(unit, entryIdx, layer);
        const Entry &entry = entries_[entryIdx];
        const int64_t deviceSlot = deviceIdxGm_.GetValue(idx);
        const int64_t hostSlot = hostIdxGm_.GetValue(idx);
        const int64_t deviceRow =
            entry.deviceBase + layer * entry.deviceLayerPitch + deviceSlot * entry.deviceSlotPitch;
        const int64_t hostRow = entry.hostBase + hostSlot * entry.hostSlotPitch + layer * entry.hostLayerPitch;
        const bool isD2H = (direction_ == 2);
        const int64_t srcRow = isD2H ? deviceRow : hostRow;
        const int64_t dstRow = isD2H ? hostRow : deviceRow;
        const int64_t remain = entry.width - offset;
        const int64_t len = STATE_EXCHANGE_COPY_CHUNK < remain ? STATE_EXCHANGE_COPY_CHUNK : remain;
        ctx.src = static_cast<uint64_t>(srcRow + offset);
        ctx.dst = static_cast<uint64_t>(dstRow + offset);
        ctx.bytes = static_cast<uint32_t>(len);
        return ctx.bytes > 0;
    }

    // ------------------------------------------------------------------
    // Row walker: units (entry, layer) sequential, slot rows round-robined
    // across cores, width splits innermost.
    // ------------------------------------------------------------------

    struct RowWalker {
        explicit STATE_EXCHANGE_AICORE_KERNEL RowWalker(StateExchangeKernel *k) : k_(k) {}

        STATE_EXCHANGE_AICORE_KERNEL bool FirstEntry(EntryCtx &ctx)
        {
            // A core participates only in the rows it round-robins; when
            // there are fewer rows than cores this core may own none.
            if (k_->aivIndex_ >= k_->indicesLen_) {
                return false;
            }
            unit_ = 0;
            idx_ = k_->aivIndex_;
            offset_ = 0;
            if (k_->MakeEntry(ctx, unit_, idx_, offset_)) {
                return true;
            }
            return NextEntry(ctx);
        }

        STATE_EXCHANGE_AICORE_KERNEL bool NextEntry(EntryCtx &ctx)
        {
            while (Advance()) {
                if (k_->MakeEntry(ctx, unit_, idx_, offset_)) {
                    return true;
                }
            }
            return false;
        }

    private:
        // Step to the next (split, slot row, unit) position. Slot rows
        // advance by aiv_num within a unit; unit boundaries reset the row
        // cursor back to the core's first row.
        STATE_EXCHANGE_AICORE_KERNEL bool Advance()
        {
            int32_t entryIdx = 0;
            int64_t layer = 0;
            k_->ResolveUnit(unit_, entryIdx, layer);
            const int64_t width = k_->entries_[entryIdx].width;
            if (offset_ + STATE_EXCHANGE_COPY_CHUNK < width) {
                offset_ += STATE_EXCHANGE_COPY_CHUNK;
                return true;
            }
            offset_ = 0;
            idx_ += k_->aivNum_;
            if (idx_ < k_->indicesLen_) {
                return true;
            }
            ++unit_;
            if (unit_ >= k_->totalUnits_) {
                return false;
            }
            idx_ = k_->aivIndex_;
            return true;
        }

        StateExchangeKernel *k_;
        int64_t unit_ = 0;
        int64_t idx_ = 0;
        int64_t offset_ = 0;
    };

    // ------------------------------------------------------------------
    // Mode-independent copy pipeline: two-stage UB double buffering. While
    // the current slot drains to GM the next slot is already filling, so the
    // copy latency of consecutive entries overlaps.
    // ------------------------------------------------------------------

    template <typename Walker>
    STATE_EXCHANGE_AICORE_KERNEL void RunPipeline(Walker &walker)
    {
        EntryCtx cur;
        if (!walker.FirstEntry(cur)) {
            return;
        }
        CopyIn(cur);
        while (true) {
            EntryCtx next;
            const bool hasNext = walker.NextEntry(next);
            if (hasNext) {
                CopyIn(next);
            }
            CopyOut(cur);
            if (!hasNext) {
                break;
            }
            cur = next;
        }
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> bindQueue_;
    AscendC::GlobalTensor<int64_t> deviceIdxGm_;
    AscendC::GlobalTensor<int64_t> hostIdxGm_;

    Entry entries_[STATE_EXCHANGE_MAX_ENTRIES];
    int64_t entryFirst_[STATE_EXCHANGE_MAX_ENTRIES] = {0};
    int64_t totalUnits_ = 0;
    int32_t numEntries_ = 0;
    int64_t indicesLen_ = 0;
    int64_t direction_ = 0;
    int64_t startDim_ = STATE_EXCHANGE_START_DIM_DEFAULT;
    uint32_t aivIndex_ = 0;
    uint32_t aivNum_ = 0;
    bool skip_ = false;
};

extern "C" __global__ __aicore__ void sgl_state_exchange(GM_ADDR deviceIndices, GM_ADDR hostIndices,
                                                         GM_ADDR deviceIndicesRef, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(optiling::SglStateExchangeTilingData);
    GET_TILING_DATA(data, tiling);
    AscendC::TPipe pipe;
    StateExchangeKernel op;
    op.Init(&pipe, reinterpret_cast<__gm__ int64_t *>(deviceIndices), reinterpret_cast<__gm__ int64_t *>(hostIndices),
            data.numEntries, data.indicesLen, data.direction, data.startDim, data.entries);
    op.Process();
}
