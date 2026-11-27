/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_quant_scatter_copy_sparse_flash_attention_copy.h
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_COPY_H
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_COPY_H

#include "kernel_operator.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_schedule.h"

namespace fused_quant_scatter_copy_sfa {
using namespace AscendC;

__aicore__ inline int64_t RowOffsetInRequest(GlobalTensor<int32_t> &table, uint32_t width, uint32_t blocks,
                                             uint64_t tableRow, int32_t token)
{
    if (token < 0 || static_cast<uint32_t>(token) / CACHE_BLOCK_SIZE >= width) return -1;
    const int32_t block = table.GetValue(tableRow + token / CACHE_BLOCK_SIZE);
    if (block < 0 || static_cast<uint32_t>(block) >= blocks) return -1;
    return (static_cast<int64_t>(block) * CACHE_BLOCK_SIZE + token % CACHE_BLOCK_SIZE) * PACKED_KV_ROW_BYTES;
}

__aicore__ inline int64_t RowOffset(GlobalTensor<int32_t> &table, uint32_t width, uint32_t blocks, uint32_t batch,
                                    int32_t token)
{
    return RowOffsetInRequest(table, width, blocks, static_cast<uint64_t>(batch) * width, token);
}

// The gate is evaluated from device metadata so graph replay may select a
// different path. A heterogeneous batch may contain any mixture of MTP0..15.
// Fully-visible steady rows let source-aware SFA consume ordinary misses from
// Host while hit tiles are running. LIM appends at most Q rewritten residents
// to the request list, so only its final Q entries need to be resident before
// attention. Cold rows and any route outside the proven fused-gather range
// retain the complete-copy path.
__aicore__ inline PackedCopyMode SelectCopyMode(GM_ADDR missCounts, GM_ADDR topkMissCounts, GM_ADDR ends,
                                                GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens,
                                                const FusedQuantScatterCopySparseFlashAttentionTilingData &td,
                                                bool fullVisible, bool residentEnabled, bool nativeLimPlan,
                                                bool &hasShort)
{
    GlobalTensor<int32_t> c, m, q, kv, cache;
    c.SetGlobalBuffer((__gm__ int32_t *)missCounts);
    m.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
    q.SetGlobalBuffer((__gm__ int32_t *)ends);
    kv.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKv);
    cache.SetGlobalBuffer((__gm__ int32_t *)numCacheTokens);
    PackedCopyMode mode = PackedCopyMode::NONE;
    int32_t previous = 0;
    for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
        const int32_t n = c.GetValue(b);
        const int32_t end = q.GetValue(b);
        const int32_t actualKvLength = kv.GetValue(b);
        const int32_t cacheTokenCount = cache.GetValue(b);
        ASSERT(n >= 0 && n <= COPY_CAPACITY);
        ASSERT(actualKvLength >= 0 && cacheTokenCount >= 0);
        ASSERT(cacheTokenCount != 0 || n == 0);
        hasShort |= actualKvLength > 0 && cacheTokenCount == 0;
        ASSERT(IsValidQuerySegment(previous, end, static_cast<int32_t>(td.baseParams.qSeqSize)));
        // Only the paired native producer guarantees a contiguous miss prefix.
        int32_t maxRouteMiss = 0;
        for (int32_t t = previous; t < end; ++t) {
            const int32_t miss = m.GetValue(t);
            ASSERT(miss >= 0 && miss <= TOPK_CAPACITY && (n != 0 || miss == 0));
            if (miss > maxRouteMiss) maxRouteMiss = miss;
        }
        mode =
            UpdatePackedCopyMode(mode, n, cacheTokenCount, maxRouteMiss, fullVisible, residentEnabled, nativeLimPlan);
        previous = end;
    }
    ASSERT(previous == td.baseParams.qSeqSize);
    return mode;
}

struct PackedResidentInputs {
    GM_ADDR current = nullptr;
    GM_ADDR positions = nullptr;
    GM_ADDR valid = nullptr;
    GM_ADDR requestSlots = nullptr;
    GM_ADDR reqToToken = nullptr;
    GM_ADDR residentEnd = nullptr;
    uint32_t requestRows = 0;
    uint32_t requestContext = 0;
};

// Share the same double-buffered DMA across short history, current KV,
// and union entries. Each lane completes its pending writes before attention.
class PackedRowTransfer
{
public:
    __aicore__ inline void Init(TPipe &pipe, GM_ADDR hbm)
    {
        hbm_.SetGlobalBuffer((__gm__ int8_t *)hbm);
        pipe.InitBuffer(storage_, 2 * ALIGNED_KV_ROW_BYTES);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            readReady_[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
            writeDone_[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());
        }
    }

    __aicore__ inline void Push(GlobalTensor<int8_t> &source, int64_t from, int64_t to)
    {
        ASSERT(from >= 0 && to >= 0);
        if (from < 0 || to < 0) return;
        const uint32_t slot = issued_ % 2;
        if (wrote_[slot]) {
            WaitFlag<HardEvent::MTE3_MTE2>(writeDone_[slot]);
            wrote_[slot] = false;
        }
        auto buf = storage_.Get<int8_t>()[slot * ALIGNED_KV_ROW_BYTES];
        DataCopyPad(buf, source[from], DataCopyExtParams{1, PACKED_KV_ROW_BYTES, 0, 0, 0},
                    DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
        pendingDst_[slot] = to;
        SetFlag<HardEvent::MTE2_MTE3>(readReady_[slot]);
        ++issued_;
        if (issued_ - finished_ == 2) WriteNext();
    }

    __aicore__ inline void Finish(TPipe &pipe)
    {
        while (finished_ < issued_)
            WriteNext();
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (wrote_[slot]) WaitFlag<HardEvent::MTE3_MTE2>(writeDone_[slot]);
            pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(readReady_[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_MTE2>(writeDone_[slot]);
        }
    }

private:
    __aicore__ inline void WriteNext()
    {
        const uint32_t slot = finished_ % 2;
        WaitFlag<HardEvent::MTE2_MTE3>(readReady_[slot]);
        DataCopyPad(hbm_[pendingDst_[slot]], storage_.Get<int8_t>()[slot * ALIGNED_KV_ROW_BYTES],
                    DataCopyExtParams{1, PACKED_KV_ROW_BYTES, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(writeDone_[slot]);
        wrote_[slot] = true;
        ++finished_;
    }

    GlobalTensor<int8_t> hbm_;
    TBuf<TPosition::VECIN> storage_;
    event_t readReady_[2], writeDone_[2];
    bool wrote_[2] = {false, false};
    int64_t pendingDst_[2];
    uint32_t issued_ = 0, finished_ = 0;
};

class PackedResidentStage
{
public:
    __aicore__ inline void Init(const PackedResidentInputs &inputs, GM_ADDR ends, GM_ADDR lengths, GM_ADDR cacheTokens,
                                TPipe &pipe)
    {
        requestRows_ = inputs.requestRows;
        context_ = inputs.requestContext;
        current.SetGlobalBuffer((__gm__ int8_t *)inputs.current);
        positions_.SetGlobalBuffer((__gm__ int64_t *)inputs.positions);
        valid_.SetGlobalBuffer((__gm__ uint8_t *)inputs.valid);
        slots_.SetGlobalBuffer((__gm__ int32_t *)inputs.requestSlots);
        address_.SetGlobalBuffer((__gm__ int32_t *)inputs.reqToToken);
        residentEnd_.SetGlobalBuffer((__gm__ int32_t *)inputs.residentEnd);
        queryEnds_.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengths_.SetGlobalBuffer((__gm__ int32_t *)lengths);
        cacheTokens_.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        pipe_ = &pipe;
        pipe_->InitBuffer(scalarBuf_, 32U);
        scalarEvent_ = static_cast<event_t>(pipe_->AllocEventID<HardEvent::S_MTE3>());
    }

    __aicore__ inline bool Enabled() const
    {
        return requestRows_ != 0;
    }

    __aicore__ inline void QueryRange(uint32_t batch, int32_t &start, int32_t &end)
    {
        start = batch == 0 ? 0 : queryEnds_.GetValue(batch - 1);
        end = queryEnds_.GetValue(batch);
    }

    __aicore__ inline int64_t FirstPosition(int32_t start)
    {
        return Enabled() ? positions_.GetValue(start) : -1;
    }

    // Resolve a requested source against the current layer's publication.
    // Incoming misses and retained rewritten residents share the same plan.
    __aicore__ inline int32_t CurrentRow(int32_t start, int32_t end, int64_t first, int32_t source)
    {
        if (first < 0 || source < first || source - first >= end - start) return -1;
        const int32_t row = start + static_cast<int32_t>(source - first);
        return valid_.GetValue(row) && positions_.GetValue(row) == source ? row : -1;
    }

    __aicore__ inline void CopyShort(uint32_t batch, int32_t start, int32_t end, uint32_t lanes,
                                     GlobalTensor<int8_t> &host, GlobalTensor<int32_t> &hbmTable,
                                     const FusedQuantScatterCopySparseFlashAttentionTilingData &td,
                                     PackedRowTransfer &transfer)
    {
        const int32_t request = slots_.GetValue(batch);
        const int32_t length = lengths_.GetValue(batch);
        const int32_t first = length - (end - start);
        const uint32_t pages = td.baseParams.maxBlockNumPerBatch;
        if (!LiveRequest(request, length) || cacheTokens_.GetValue(batch) != 0 || first < 0 ||
            static_cast<uint32_t>(length) > context_ || static_cast<uint32_t>(length) > pages * CACHE_BLOCK_SIZE)
            return;
        const int32_t old = residentEnd_.GetValue(request);
        const int32_t begin = old < 0 ? 0 : (old < first ? old : first);
        const int64_t hostRows = static_cast<int64_t>(td.dramPhysicalBlocks) * CACHE_BLOCK_SIZE;
        // Graph padding and long requests exit before touching the host address
        // table. Real short rows read only the missing historical interval.
        for (int32_t pos = begin + GetBlockIdx(); pos < first; pos += lanes) {
            const int32_t source = address_.GetValue(static_cast<uint64_t>(request) * context_ + pos);
            ASSERT(source >= 0 && source < hostRows);
            if (source < 0 || source >= hostRows) continue;
            transfer.Push(host, static_cast<int64_t>(source) * PACKED_KV_ROW_BYTES,
                          RowOffset(hbmTable, pages, td.hbmPhysicalBlocks, batch, pos));
        }
        for (int32_t row = start + GetBlockIdx(); row < end; row += lanes) {
            const int64_t pos = positions_.GetValue(row);
            if (valid_.GetValue(row) && pos >= first && pos < length) {
                transfer.Push(current, static_cast<int64_t>(row) * PACKED_KV_ROW_BYTES,
                              RowOffset(hbmTable, pages, td.hbmPhysicalBlocks, batch, static_cast<int32_t>(pos)));
            }
        }
    }

    __aicore__ inline void PublishExtents(const FusedQuantScatterCopySparseFlashAttentionTilingData &td)
    {
        // Publish after the copy barrier; every lane above reads the old
        // extent. One publisher also avoids false sharing between requests.
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) return;
            LocalTensor<int32_t> scalar = scalarBuf_.Get<int32_t>();
            DataCopyParams one{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
            for (uint32_t batch = 0; batch < td.baseParams.batchSize; ++batch) {
                const int32_t request = slots_.GetValue(batch);
                const int32_t length = lengths_.GetValue(batch);
                if (!LiveRequest(request, length)) continue;
                int32_t extent = -1;
                if (cacheTokens_.GetValue(batch) == 0) {
                    int32_t start, end;
                    QueryRange(batch, start, end);
                    const int32_t first = length - (end - start);
                    bool contiguous =
                        first >= 0 && static_cast<uint32_t>(length) <= context_ &&
                        static_cast<uint32_t>(length) <= td.baseParams.maxBlockNumPerBatch * CACHE_BLOCK_SIZE;
                    for (int32_t row = start; contiguous && row < end; ++row) {
                        contiguous = valid_.GetValue(row) && positions_.GetValue(row) == first + row - start;
                    }
                    if (contiguous) extent = length;
                }
                if (residentEnd_.GetValue(request) != extent) {
                    scalar.SetValue(0, extent);
                    SetFlag<HardEvent::S_MTE3>(scalarEvent_);
                    WaitFlag<HardEvent::S_MTE3>(scalarEvent_);
                    DataCopyPad(residentEnd_[request], scalar, one);
                }
            }
        }
    }

    GlobalTensor<int8_t> current;

private:
    __aicore__ inline bool LiveRequest(int32_t request, int32_t length)
    {
        return request > 0 && static_cast<uint32_t>(request) < requestRows_ && length > 0;
    }

    GlobalTensor<int32_t> slots_, address_, residentEnd_, queryEnds_, lengths_, cacheTokens_;
    GlobalTensor<int64_t> positions_;
    GlobalTensor<uint8_t> valid_;
    uint32_t requestRows_ = 0, context_ = 0;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> scalarBuf_;
    event_t scalarEvent_ = static_cast<event_t>(-1);
};

__aicore__ inline void StageKv(TPipe &pipe, GM_ADDR hbm, GM_ADDR dram, GM_ADDR ht, GM_ADDR dt, GM_ADDR missSrcIds,
                               GM_ADDR missDstSlots, GM_ADDR missCounts,
                               const FusedQuantScatterCopySparseFlashAttentionTilingData &td,
                               PackedResidentStage &resident, PackedCopyMode copyMode, bool copyShort)
{
    if ASCEND_IS_AIV {
        GlobalTensor<int8_t> host;
        GlobalTensor<int32_t> htable, dtable, sources, dests, sizes;
        host.SetGlobalBuffer((__gm__ int8_t *)dram);
        htable.SetGlobalBuffer((__gm__ int32_t *)ht);
        dtable.SetGlobalBuffer((__gm__ int32_t *)dt);
        sources.SetGlobalBuffer((__gm__ int32_t *)missSrcIds);
        dests.SetGlobalBuffer((__gm__ int32_t *)missDstSlots);
        sizes.SetGlobalBuffer((__gm__ int32_t *)missCounts);
        TPipe stagePipe;
        PackedRowTransfer transfer;
        transfer.Init(stagePipe, hbm);
        const uint32_t lanes = td.singleCoreParams.usedCoreNum * 2;
        for (uint32_t b = 0; b < td.baseParams.batchSize; ++b) {
            int32_t start = 0, end = 0;
            if (resident.Enabled()) resident.QueryRange(b, start, end);
            if (copyShort) resident.CopyShort(b, start, end, lanes, host, htable, td, transfer);
            if (copyMode == PackedCopyMode::NONE) continue;
            const int32_t n = sizes.GetValue(b);
            if (n <= 0) continue;
            const int64_t first = resident.FirstPosition(start);
            const uint32_t refreshCount = static_cast<uint32_t>(end - start);
            const uint32_t copyStart =
                copyMode == PackedCopyMode::REFRESH_SUFFIX && static_cast<uint32_t>(n) > refreshCount
                    ? static_cast<uint32_t>(n) - refreshCount
                    : 0U;
            for (uint32_t i = copyStart + GetBlockIdx(); i < static_cast<uint32_t>(n); i += lanes) {
                const uint64_t p = static_cast<uint64_t>(b) * COPY_CAPACITY + i;
                const int32_t source = sources.GetValue(p);
                const int64_t to =
                    RowOffset(htable, td.baseParams.maxBlockNumPerBatch, td.hbmPhysicalBlocks, b, dests.GetValue(p));
                const int32_t row = resident.CurrentRow(start, end, first, source);
                if (row >= 0) {
                    transfer.Push(resident.current, static_cast<int64_t>(row) * PACKED_KV_ROW_BYTES, to);
                } else {
                    transfer.Push(host, RowOffset(dtable, td.dramMaxBlocks, td.dramPhysicalBlocks, b, source), to);
                }
            }
        }
        transfer.Finish(stagePipe);
    }
    SyncAll<false>();
}

}  // namespace fused_quant_scatter_copy_sfa

#endif
