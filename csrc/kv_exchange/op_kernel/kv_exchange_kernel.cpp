/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of MemulanPSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
 *
 * Ported from MemFabric_Hybrid acc_offload module
 * (src/acc_offload/csrc/operators/acc_offload_kv_exchange.h/.cpp,
 * branch br_feature_a5_offload) into sgl-kernel-npu.
 */

#include "kernel_operator.h"

#define HYBM_AICORE_KERNEL __attribute__((always_inline)) __aicore__ __inline__

// Same UB double-buffer budget as the KV sparse-copy kernel: one entry is
// copied through an 88KB UB slot, entries larger than that are split.
constexpr int64_t KV_EXCHANGE_UB_BUFFER_SIZE = 88 * 1024;
constexpr int32_t KV_EXCHANGE_MAX_COMPONENTS = 4;
// Transfer description. The kv_exchange entry receives it as flat kernel
// launch parameters (device-validated contract):
//   num_components (<= 4, k/v/index_k/index_k_scale order)
//   num_pages      page_size
//   direction: 1 = H2D, 2 = D2H
//   device_indices / host_indices (int64 token indices, device memory)
//   host_layout_mode: 0 = page_first (page-outer / layer-inner iteration),
//       1 = layer_first (layer-outer / page-inner iteration with peek-ahead
//       merging of contiguous pages)
//   start_dim: first participating AIV core index. < 0 selects the
//       native acc_offload occupancy (blockDim - 2, i.e. only the last
//       2 AIV cores copy, keeping the rest free for concurrent compute);
//       0 means all launched cores participate (perf-sweep override).
//   per component c (9 launch parameters):
//   [0] device_base  [1] host_base (device-visible address of the host page,
//       i.e. DVA produced by aclrtHostRegister on the caller side)
//   [2] device_layer_pitch  [3] device_page_pitch   (bytes)
//   [4] host_page_pitch     [5] host_layer_pitch    (bytes)
//   [6] width (bytes of one (page, layer) row)  [7] layer_lo  [8] layer_hi
// A component with layer_lo >= layer_hi (or width <= 0) transfers nothing.
constexpr int32_t KV_EXCHANGE_META_STRIDE = 9;

// start_dim < 0: reserve the last 2 AIV cores (native acc_offload occupancy).
constexpr int32_t KV_EXCHANGE_START_DIM_DEFAULT = -1;

/*
 * Fused address-computation + copy kernel for the page_first(host) <->
 * layer_first(device) KV cache exchange.  No (src, dst, len) entry table is
 * built on the host: every AIV core derives the addresses of the
 * (page, layer, split) blocks it owns directly from the device-resident page
 * indices and the per-component layout metadata, so the indices never
 * round-trip through the CPU.
 *
 * The copy pipeline is mode independent (RunPipeline: two-stage UB double
 * buffering); only the iteration order differs, so each host layout mode
 * plugs in its own Walker with the same FirstEntry/NextEntry contract:
 *
 *   PageFirstWalker  (mode 0): pages round-robined across AIV cores (outer
 *       loop); each core walks its pages' components and layers sequentially
 *       (inner loop).
 *
 *   LayerFirstWalker (mode 1): layers are the outer loop and pages the inner
 *       loop.  Pages are block-partitioned across cores (not round-robined)
 *       so that contiguous pages in the sorted indices array stay on the same
 *       core.  A peek-ahead scan merges runs of pages whose host AND device
 *       slots are both contiguous into a single large copy, reducing DMA
 *       request overhead.  When no contiguous run is found, the kernel
 *       degrades to per-page copies with the same UB double buffering.
 */
class OffloadKvExchangeKernel
{
public:
    enum HostLayoutMode : int32_t {
        HOST_LAYOUT_PAGE_FIRST = 0,
        HOST_LAYOUT_LAYER_FIRST = 1,
    };

    HYBM_AICORE_KERNEL OffloadKvExchangeKernel() {}

    // Assembly from the flat transfer description. The kv_exchange entry
    // passes every field as a launch parameter, so the description never
    // round-trips through device memory.
    HYBM_AICORE_KERNEL void Init(AscendC::TPipe *pipe, __gm__ int64_t *deviceIdx, __gm__ int64_t *hostIdx,
                                 int64_t numComponents, int64_t numPages, int64_t pageSize, int64_t direction,
                                 int64_t hostLayoutMode, int64_t startDim,
                                 const int64_t comps[KV_EXCHANGE_MAX_COMPONENTS][KV_EXCHANGE_META_STRIDE])
    {
        pipe_ = pipe;
        aivNum_ = AscendC::GetBlockNum();
        aivIndex_ = AscendC::GetBlockIdx();

        // Active-core window: cores [startDim, blockDim) participate. A
        // negative startDim selects the native acc_offload occupancy --
        // reserve the last 2 AIV cores for the copy so the rest stay free
        // for concurrent compute (62 on the 64-core Ascend950, 46 on the
        // 48-core parts); 0 lets every launched core participate
        // (perf-sweep override).
        startDim_ = startDim;
        if (startDim_ < 0) {
            startDim_ = static_cast<int64_t>(aivNum_) - 2;
        }

        if (aivIndex_ < startDim_) {
            skip_ = true;
            return;
        }

        aivIndex_ = aivIndex_ - startDim_;
        aivNum_ = aivNum_ - startDim_;

        numComponents_ = static_cast<int32_t>(numComponents);
        numPages_ = numPages;
        pageSize_ = pageSize;
        direction_ = direction;
        const int64_t indicesLen = numPages_ * pageSize_;
        deviceIdxGm_.SetGlobalBuffer(deviceIdx, static_cast<uint32_t>(indicesLen));
        hostIdxGm_.SetGlobalBuffer(hostIdx, static_cast<uint32_t>(indicesLen));

        hostLayoutMode_ = static_cast<int32_t>(hostLayoutMode);

        for (int32_t c = 0; c < numComponents_ && c < KV_EXCHANGE_MAX_COMPONENTS; ++c) {
            comps_[c].deviceBase = comps[c][0];
            comps_[c].hostBase = comps[c][1];
            comps_[c].deviceLayerPitch = comps[c][2];
            comps_[c].devicePagePitch = comps[c][3];
            comps_[c].hostPagePitch = comps[c][4];
            comps_[c].hostLayerPitch = comps[c][5];
            comps_[c].width = comps[c][6];
            comps_[c].layerLo = comps[c][7];
            comps_[c].layerHi = comps[c][8];
        }
        if (numComponents_ > KV_EXCHANGE_MAX_COMPONENTS) {
            numComponents_ = KV_EXCHANGE_MAX_COMPONENTS;
        }

        pipe_->InitBuffer(bindQueue_, 2, KV_EXCHANGE_UB_BUFFER_SIZE);
    }

    HYBM_AICORE_KERNEL void Process()
    {
        if (skip_ || numPages_ <= 0 || numComponents_ <= 0) {
            return;
        }
        if (hostLayoutMode_ == HOST_LAYOUT_LAYER_FIRST) {
            LayerFirstWalker walker(this);
            RunPipeline(walker);
        } else {
            PageFirstWalker walker(this);
            RunPipeline(walker);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    struct Component {
        int64_t deviceBase = 0;
        int64_t hostBase = 0;
        int64_t deviceLayerPitch = 0;
        int64_t devicePagePitch = 0;
        int64_t hostPagePitch = 0;
        int64_t hostLayerPitch = 0;
        int64_t width = 0;
        int64_t layerLo = 0;
        int64_t layerHi = 0;

        HYBM_AICORE_KERNEL bool Empty() const
        {
            return width <= 0 || layerLo >= layerHi;
        }
    };

    struct EntryCtx {
        uint64_t src = 0;
        uint64_t dst = 0;
        uint32_t bytes = 0;
    };

    // Iterator cursor shared by both walkers: which component / layer / width
    // split is in flight, plus the resolved (devicePage, hostPage) pair of the
    // row being transferred.
    struct Cursor {
        int32_t comp = 0;
        int64_t layer = 0;
        int64_t offset = 0;
        int64_t devicePage = 0;
        int64_t hostPage = 0;
    };

    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    HYBM_AICORE_KERNEL int32_t FirstNonEmptyComp()
    {
        for (int32_t c = 0; c < numComponents_; ++c) {
            if (!comps_[c].Empty()) {
                return c;
            }
        }
        return -1;
    }

    HYBM_AICORE_KERNEL int32_t NextNonEmptyComp(int32_t from)
    {
        int32_t next = from + 1;
        while (next < numComponents_ && comps_[next].Empty()) {
            ++next;
        }
        return next;
    }

    HYBM_AICORE_KERNEL void CopyIn(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<uint8_t> local = bindQueue_.AllocTensor<uint8_t>();
        AscendC::GlobalTensor<uint8_t> srcGm;
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(ctx.src), ctx.bytes);
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPadExtParams<uint8_t> padParams{};
        AscendC::DataCopyPad(local, srcGm, copyParams, padParams);
        bindQueue_.EnQue(local);
    }

    HYBM_AICORE_KERNEL void CopyOut(const EntryCtx &ctx)
    {
        AscendC::LocalTensor<uint8_t> local = bindQueue_.DeQue<uint8_t>();
        AscendC::GlobalTensor<uint8_t> dstGm;
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(ctx.dst), ctx.bytes);
        AscendC::DataCopyExtParams copyParams(1, ctx.bytes, 0, 0, 0);
        AscendC::DataCopyPad(dstGm, local, copyParams);
        bindQueue_.FreeTensor(local);
    }

    // Resolve the host/device page numbers of `page` into the cursor.
    HYBM_AICORE_KERNEL void LoadPages(Cursor &cur, int64_t page)
    {
        const int64_t token = page * pageSize_;
        cur.devicePage = deviceIdxGm_.GetValue(token) / pageSize_;
        cur.hostPage = hostIdxGm_.GetValue(token) / pageSize_;
    }

    // Build the copy entry for the cursor's current row. `rowWidth` is the
    // contiguous byte width at the cursor (one page slot for page_first, the
    // merged page run for layer_first); the entry is split when it exceeds
    // the UB buffer.
    HYBM_AICORE_KERNEL bool MakeEntry(EntryCtx &ctx, const Cursor &cur, int64_t rowWidth)
    {
        const Component &c = comps_[cur.comp];
        const int64_t deviceRow = c.deviceBase + cur.layer * c.deviceLayerPitch + cur.devicePage * c.devicePagePitch;
        const int64_t hostRow = c.hostBase + cur.hostPage * c.hostPagePitch + cur.layer * c.hostLayerPitch;
        const bool isD2H = (direction_ == 2);
        const int64_t srcRow = isD2H ? deviceRow : hostRow;
        const int64_t dstRow = isD2H ? hostRow : deviceRow;
        const int64_t remain = rowWidth - cur.offset;
        const int64_t len = KV_EXCHANGE_UB_BUFFER_SIZE < remain ? KV_EXCHANGE_UB_BUFFER_SIZE : remain;
        ctx.src = static_cast<uint64_t>(srcRow + cur.offset);
        ctx.dst = static_cast<uint64_t>(dstRow + cur.offset);
        ctx.bytes = static_cast<uint32_t>(len);
        return ctx.bytes > 0;
    }

    // ------------------------------------------------------------------
    // Mode 0 walker: page-first host (page-outer / layer-inner, round-robin)
    // ------------------------------------------------------------------

    struct PageFirstWalker {
        explicit HYBM_AICORE_KERNEL PageFirstWalker(OffloadKvExchangeKernel *k) : k_(k) {}

        HYBM_AICORE_KERNEL bool FirstEntry(EntryCtx &ctx)
        {
            cur_.comp = k_->FirstNonEmptyComp();
            if (cur_.comp < 0) {
                return false;
            }
            cur_.layer = k_->comps_[cur_.comp].layerLo;
            page_ = k_->aivIndex_;
            if (page_ >= k_->numPages_) {
                return false;
            }
            k_->LoadPages(cur_, page_);
            if (k_->MakeEntry(ctx, cur_, k_->comps_[cur_.comp].width)) {
                return true;
            }
            return NextEntry(ctx);
        }

        HYBM_AICORE_KERNEL bool NextEntry(EntryCtx &ctx)
        {
            while (Advance()) {
                if (k_->MakeEntry(ctx, cur_, k_->comps_[cur_.comp].width)) {
                    return true;
                }
            }
            return false;
        }

    private:
        // Step to the next (split, layer, component, page) position. Pages are
        // round-robined across cores: page += aivNum_.
        HYBM_AICORE_KERNEL bool Advance()
        {
            const Component &c = k_->comps_[cur_.comp];
            if (cur_.offset + KV_EXCHANGE_UB_BUFFER_SIZE < c.width) {
                cur_.offset += KV_EXCHANGE_UB_BUFFER_SIZE;
                return true;
            }
            cur_.offset = 0;
            if (cur_.layer + 1 < c.layerHi) {
                ++cur_.layer;
                return true;
            }
            const int32_t next = k_->NextNonEmptyComp(cur_.comp);
            if (next < k_->numComponents_) {
                cur_.comp = next;
                cur_.layer = k_->comps_[next].layerLo;
                return true;
            }
            page_ += k_->aivNum_;
            if (page_ >= k_->numPages_) {
                return false;
            }
            cur_.comp = k_->FirstNonEmptyComp();
            cur_.layer = k_->comps_[cur_.comp].layerLo;
            k_->LoadPages(cur_, page_);
            return true;
        }

        OffloadKvExchangeKernel *k_;
        Cursor cur_;
        int64_t page_ = 0;
    };

    // ------------------------------------------------------------------
    // Mode 1 walker: layer-first host (layer-outer / page-inner, block
    // partition with peek-ahead merging)
    // ------------------------------------------------------------------

    struct LayerFirstWalker {
        explicit HYBM_AICORE_KERNEL LayerFirstWalker(OffloadKvExchangeKernel *k) : k_(k)
        {
            // Block-partition the pages across cores so contiguous pages in
            // the sorted indices array stay on the same core.
            const int64_t pagesPerCore = k_->numPages_ / k_->aivNum_;
            const int64_t remainder = k_->numPages_ % k_->aivNum_;
            if (k_->aivIndex_ < remainder) {
                pageStart_ = k_->aivIndex_ * (pagesPerCore + 1);
                pageEnd_ = pageStart_ + pagesPerCore + 1;
            } else {
                pageStart_ = remainder * (pagesPerCore + 1) + (k_->aivIndex_ - remainder) * pagesPerCore;
                pageEnd_ = pageStart_ + pagesPerCore;
            }
        }

        HYBM_AICORE_KERNEL bool FirstEntry(EntryCtx &ctx)
        {
            cur_.comp = k_->FirstNonEmptyComp();
            if (cur_.comp < 0 || pageStart_ >= pageEnd_) {
                return false;
            }
            cur_.layer = k_->comps_[cur_.comp].layerLo;
            pagePos_ = pageStart_;
            cur_.offset = 0;
            LoadMerged();
            return k_->MakeEntry(ctx, cur_, MergedWidth());
        }

        HYBM_AICORE_KERNEL bool NextEntry(EntryCtx &ctx)
        {
            while (Advance()) {
                if (k_->MakeEntry(ctx, cur_, MergedWidth())) {
                    return true;
                }
            }
            return false;
        }

    private:
        HYBM_AICORE_KERNEL int64_t MergedWidth() const
        {
            return mergeCount_ * k_->comps_[cur_.comp].width;
        }

        // Step to the next (split, page-run, layer, component) position: the
        // current merged run first, then the next run, then the next layer,
        // then the next component.
        HYBM_AICORE_KERNEL bool Advance()
        {
            if (cur_.offset + KV_EXCHANGE_UB_BUFFER_SIZE < MergedWidth()) {
                cur_.offset += KV_EXCHANGE_UB_BUFFER_SIZE;
                return true;
            }
            cur_.offset = 0;

            pagePos_ += mergeCount_;
            if (pagePos_ < pageEnd_) {
                LoadMerged();
                return true;
            }

            pagePos_ = pageStart_;
            if (cur_.layer + 1 < k_->comps_[cur_.comp].layerHi) {
                ++cur_.layer;
                LoadMerged();
                return true;
            }

            const int32_t next = k_->NextNonEmptyComp(cur_.comp);
            if (next < k_->numComponents_) {
                cur_.comp = next;
                cur_.layer = k_->comps_[next].layerLo;
                LoadMerged();
                return true;
            }
            return false;
        }

        // Resolve the page pair at pagePos_ and peek ahead: extend mergeCount_
        // while both the host and device page numbers stay contiguous, so the
        // whole run is transferred as one wide row.
        HYBM_AICORE_KERNEL void LoadMerged()
        {
            k_->LoadPages(cur_, pagePos_);
            mergeCount_ = 1;
            while (pagePos_ + mergeCount_ < pageEnd_) {
                const int64_t nextToken = (pagePos_ + mergeCount_) * k_->pageSize_;
                const int64_t nextHost = k_->hostIdxGm_.GetValue(nextToken) / k_->pageSize_;
                const int64_t nextDevice = k_->deviceIdxGm_.GetValue(nextToken) / k_->pageSize_;
                if (nextHost != cur_.hostPage + mergeCount_ || nextDevice != cur_.devicePage + mergeCount_) {
                    break;
                }
                ++mergeCount_;
            }
        }

        OffloadKvExchangeKernel *k_;
        Cursor cur_;
        int64_t pageStart_ = 0;
        int64_t pageEnd_ = 0;
        int64_t pagePos_ = 0;
        int64_t mergeCount_ = 1;
    };

    // ------------------------------------------------------------------
    // Mode-independent copy pipeline: two-stage UB double buffering. While
    // the current slot drains to GM the next slot is already filling, so the
    // copy latency of consecutive entries overlaps.
    // ------------------------------------------------------------------

    template <typename Walker>
    HYBM_AICORE_KERNEL void RunPipeline(Walker &walker)
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

    Component comps_[KV_EXCHANGE_MAX_COMPONENTS];
    int32_t numComponents_ = 0;
    int64_t numPages_ = 0;
    int64_t pageSize_ = 0;
    int64_t direction_ = 0;
    int64_t startDim_ = KV_EXCHANGE_START_DIM_DEFAULT;
    int32_t hostLayoutMode_ = HOST_LAYOUT_PAGE_FIRST;
    uint32_t aivIndex_ = 0;
    uint32_t aivNum_ = 0;
    bool skip_ = false;
};

// Per-component launch parameters (9 scalars each, same field order as the
// legacy meta slots); c is the component index 0..3.
#define KV_EXCHANGE_COMPONENT_PARAMS(c)                                                        \
    int64_t deviceBase##c, int64_t hostBase##c, int64_t deviceLayerPitch##c, int64_t devicePagePitch##c, \
        int64_t hostPagePitch##c, int64_t hostLayerPitch##c, int64_t width##c, int64_t layerLo##c,  \
        int64_t layerHi##c

#define KV_EXCHANGE_COMPONENT_ARGS(c)                                                          \
    deviceBase##c, hostBase##c, deviceLayerPitch##c, devicePagePitch##c, hostPagePitch##c,       \
        hostLayerPitch##c, width##c, layerLo##c, layerHi##c

/*
 * Production entry: the whole transfer description travels inside the kernel
 * launch parameter block. No device-resident meta, no pinned host staging and
 * no separate H2D meta upload remain on the submission path.
 */
extern "C" __global__ __aicore__ void kv_exchange(GM_ADDR device_indices, GM_ADDR host_indices,
                                                int64_t num_components, int64_t num_pages, int64_t page_size,
                                                int64_t direction, int64_t host_layout, int64_t start_dim,
                                                KV_EXCHANGE_COMPONENT_PARAMS(0), KV_EXCHANGE_COMPONENT_PARAMS(1),
                                                KV_EXCHANGE_COMPONENT_PARAMS(2), KV_EXCHANGE_COMPONENT_PARAMS(3))
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    OffloadKvExchangeKernel op;
    const int64_t comps[KV_EXCHANGE_MAX_COMPONENTS][KV_EXCHANGE_META_STRIDE] = {
        {KV_EXCHANGE_COMPONENT_ARGS(0)},
        {KV_EXCHANGE_COMPONENT_ARGS(1)},
        {KV_EXCHANGE_COMPONENT_ARGS(2)},
        {KV_EXCHANGE_COMPONENT_ARGS(3)},
    };
    op.Init(&pipe, reinterpret_cast<__gm__ int64_t *>(device_indices),
            reinterpret_cast<__gm__ int64_t *>(host_indices), num_components, num_pages, page_size, direction,
            host_layout, start_dim, comps);
    op.Process();
}
