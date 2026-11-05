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
 * \file mega_moe_base.h
 * \brief
 */

#ifndef MEGA_MOE_BASE_H
#define MEGA_MOE_BASE_H

#include "lib/std/tuple.h"
#include "mega_moe_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_apt.h"
#include "mega_moe_workspace_info.h"

using namespace AscendC;

// A peer can enter the next barrier before this rank has consumed its
// current publication, overwriting the single signal slot.  In a serial
// collective it can lead by at most one barrier: the next wait still needs
// this rank's publication.  Compare sequence numbers modulo 2^32 so both
// an overtaken publication and counter wrap are handled without accepting
// an older signal (the distance must remain below half the sequence range).
__aicore__ inline bool IsCrossRankBarrierReached(uint32_t observed, uint32_t expected)
{
    return observed - expected < 0x80000000U;
}

// Bounded exponential backoff with a per-core stagger for GM readiness
// spins.  One cadence step matching the join already tuned on
// chainLenReadyFlag: a 100-cycle floor keeps the ready-on-first-probe
// common case (decode) fast, and after a few misses the inter-probe gap
// grows (capped at 800) and is staggered per core, so many cores waiting
// on one GM word stop saturating the L2/NoC bank and delaying the
// producer's publish -- a convex feedback that otherwise turns a
// routing-skewed tail into a several-fold latency spike.  Callers keep
// their own exit predicate; only the poll cadence changes.
__aicore__ inline void GmSpinBackoffStep(uint32_t &pollMisses, int64_t &pollDelayCycles)
{
    ++pollMisses;
    if (pollMisses > 8U) {
        pollDelayCycles = pollDelayCycles < 800 ? pollDelayCycles << 1 : 800;
    }
    int64_t staggerCycles = pollMisses > 8U ? static_cast<int64_t>((GetBlockIdx() & 3U) * 16U) : 0;
    int64_t startCycle = GetSystemCycle();
    while (GetSystemCycle() - startCycle < pollDelayCycles + staggerCycles) {
    }
}

struct Mc2MoeContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer[1024];
};

struct GMMAddrInfo {
    GM_ADDR aGlobal;
    GM_ADDR bGlobal;
    GM_ADDR aScaleGlobal;
    GM_ADDR bScaleGlobal;
    GM_ADDR outputGlobal;
    __gm__ int32_t *flagSwigluToGmm2;
    __gm__ int32_t *dispatchDoneCores;
    int32_t dispatchReadyTarget;
    __gm__ int32_t *dispatchTileReady;
    __gm__ int32_t *flagGmm2Ready;
};

// Legacy readiness-sweep token-owner descriptor helpers are retained for
// source-level tests; the production kernel does not call that protocol.  The experiment stores
// one descriptor per local contribution in the
// existing [token, expert] int32 workspace.  Pack (canonical row, expert,
// top-k slot) into one word so Dispatch keeps the same one-descriptor-store
// cost as the C3 expert-major path.  The capability check is shape-generic:
// unsupported widths simply retain the expert-major protocol.
__aicore__ inline uint32_t GetTokenOwnerFieldBits(uint32_t valueCount)
{
    // Bit-width of (valueCount - 1) via a fixed-depth binary search.  The
    // old one-bit-per-iteration shift loop was inlined at dozens of call
    // sites and became one of the largest single-line code-size contributors
    // in the kernel image; this form is branch-free and identical for every
    // input (0 maps to 0 bits).
    uint32_t value = valueCount > 0U ? valueCount - 1U : 0U;
    uint32_t bits = 0U;
    if (value > 0xFFFFU) {
        bits += 16U;
        value >>= 16U;
    }
    if (value > 0xFFU) {
        bits += 8U;
        value >>= 8U;
    }
    if (value > 0xFU) {
        bits += 4U;
        value >>= 4U;
    }
    if (value > 3U) {
        bits += 2U;
        value >>= 2U;
    }
    if (value > 1U) {
        bits += 1U;
        value >>= 1U;
    }
    return bits + value;
}

// Combine consumes one metadata word per canonical contribution.  When the
// runtime widths fit in a signed int32, reuse stepIdx to carry source rank,
// source token row and original top-k slot together with the chain step.
// Dispatch knows all four values already; packing them removes Combine's
// per-row divide, irregular
// expandedRowIdx read and its source-segment cursor without adding workspace.
__aicore__ inline bool CanPackCombineRouteMetadata(uint32_t worldSize, uint32_t m, uint32_t topK)
{
    if (worldSize == 0U || m == 0U || topK == 0U || m > UINT32_MAX / worldSize) {
        return false;
    }
    uint32_t rankBits = GetTokenOwnerFieldBits(worldSize);
    uint32_t tokenRowBits = GetTokenOwnerFieldBits(m);
    uint32_t topKBits = GetTokenOwnerFieldBits(topK);
    return rankBits + tokenRowBits + 2U * topKBits <= 31U;
}

__aicore__ inline bool ShouldPackCombineRouteMetadata(uint32_t worldSize, uint32_t m, uint32_t topK,
                                                      uint64_t maxOutputSize, uint32_t workerCount)
{
    // Packing trades a few Dispatch shifts for two irregular Combine reads
    // and a segment walk.  Require at least four routed rows per physical
    // worker so decode/small launches retain the lower fixed-cost protocol.
    if (!CanPackCombineRouteMetadata(worldSize, m, topK) || workerCount == 0U || workerCount > UINT32_MAX / 4U) {
        return false;
    }
    return maxOutputSize >= static_cast<uint64_t>(workerCount * 4U);
}

__aicore__ inline int32_t PackCombineRouteMetadata(uint32_t sourceRank, uint32_t tokenRow, uint32_t topKSlot,
                                                   uint32_t stepK, uint32_t tokenRowBits, uint32_t topKBits)
{
    uint32_t topKPayloadBits = 2U * topKBits;
    return static_cast<int32_t>((sourceRank << (tokenRowBits + topKPayloadBits)) | (tokenRow << topKPayloadBits) |
                                (topKSlot << topKBits) | stepK);
}

__aicore__ inline uint32_t GetTokenOwnerDescriptorPayloadBits(uint32_t expertPerRank, uint32_t topK)
{
    return GetTokenOwnerFieldBits(expertPerRank) + GetTokenOwnerFieldBits(topK);
}

__aicore__ inline bool CanPackTokenOwnerDescriptor(uint32_t expertPerRank, uint32_t topK, uint64_t maxOutputSize)
{
    if (topK == 0U || expertPerRank < topK || maxOutputSize == 0U) {
        return false;
    }
    uint32_t payloadBits = GetTokenOwnerDescriptorPayloadBits(expertPerRank, topK);
    if (payloadBits >= 32U) {
        return false;
    }
    // -1 is the invalid-slot sentinel in the shared descriptor workspace, so
    // every valid packed descriptor must remain non-negative when read back
    // as int32_t.
    uint64_t maxCanonicalRow = static_cast<uint64_t>(INT32_MAX) >> payloadBits;
    return maxOutputSize - 1U <= maxCanonicalRow;
}

// The production token-local Dispatch plan is indexed by the original
// [source token, top-k slot], so the descriptor itself only needs the
// schedule position, the expert-local 256-row tile, and the canonical
// expert-major destination row.  Keeping every bit-width runtime-derived
// makes the capability independent of named shapes and physical core count.
__aicore__ inline uint32_t GetDispatchFanoutPayloadBits(uint32_t expertPerRank, uint32_t maxTilesPerExpert)
{
    return GetTokenOwnerFieldBits(expertPerRank) + GetTokenOwnerFieldBits(maxTilesPerExpert);
}

__aicore__ inline bool CanPackDispatchFanoutDescriptor(uint32_t expertPerRank, uint32_t maxTilesPerExpert,
                                                       uint64_t maxOutputSize)
{
    if (expertPerRank == 0U || maxTilesPerExpert == 0U || maxOutputSize == 0U) {
        return false;
    }
    uint32_t payloadBits = GetDispatchFanoutPayloadBits(expertPerRank, maxTilesPerExpert);
    if (payloadBits >= 31U) {
        return false;
    }
    uint64_t maxCanonicalRow = static_cast<uint64_t>(INT32_MAX) >> payloadBits;
    return maxOutputSize - 1U <= maxCanonicalRow;
}

__aicore__ inline uint32_t GetDispatchFanoutTouchedCapacity(uint32_t topK)
{
    return topK > DISPATCH_FANOUT_READY_BATCH_ROUTES ? topK : DISPATCH_FANOUT_READY_BATCH_ROUTES;
}

__aicore__ inline bool CanFitDispatchFanoutInUb(uint32_t sourceRowBytes, uint32_t expertPerRank,
                                                uint32_t maxTilesPerExpert, uint32_t topK, uint32_t sourceTokens,
                                                uint32_t sourceRankCount, uint32_t workerCount)
{
    constexpr uint32_t maxUbInts = DISPATCH_FANOUT_UB_BYTES / sizeof(int32_t);
    if (workerCount == 0U || sourceRankCount == 0U || sourceTokens % sourceRankCount != 0U) {
        return false;
    }
    uint32_t touchedCapacity = GetDispatchFanoutTouchedCapacity(topK);
    uint32_t tokensPerSource = sourceTokens / sourceRankCount;
    uint32_t maxOwnedSourceRows = tokensPerSource / workerCount + (tokensPerSource % workerCount != 0U ? 1U : 0U);
    // Dispatch owns the same Q/R row interval in every source rank.  This can
    // be up to (sourceRankCount-1) tokens wider than ceil(world*m/workers), so
    // prove the actual source-striped slice rather than the old flat bound.
    uint32_t maxOwnedTokens = maxOwnedSourceRows * sourceRankCount;
    if (maxTilesPerExpert == 0U || expertPerRank > maxUbInts / maxTilesPerExpert || topK > maxUbInts ||
        touchedCapacity > maxUbInts - topK || maxOwnedTokens > maxUbInts - topK - touchedCapacity) {
        return false;
    }
    uint32_t alignedOwnedTokens =
        Ops::Base::CeilAlign(maxOwnedTokens, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)));
    if (alignedOwnedTokens > (maxUbInts - topK - touchedCapacity) / 2U) {
        return false;
    }
    uint32_t tokenArrayBase =
        Ops::Base::CeilAlign(2U * topK + touchedCapacity, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)));
    uint32_t fixedScratchEntries = tokenArrayBase + 2U * alignedOwnedTokens;
    if (fixedScratchEntries > maxUbInts) {
        return false;
    }
    uint32_t remainingScratchEntries = maxUbInts - fixedScratchEntries;
    // Cache every descriptor owned by this AIV after its first GM read.  The
    // payload traversal then sorts from scalar-addressable UB instead of
    // rereading topK GM slots for every token.  Keep the capacity proof
    // subtractive so large runtime topK/source-token values cannot overflow.
    if (topK != 0U && maxOwnedTokens > remainingScratchEntries / topK) {
        return false;
    }
    // The runtime token slice can start at any top-k slot.  Reserve at most
    // one int32 cacheline prefix so the aligned descriptor body can land at
    // an aligned UB address while scalar head/tail entries keep their logical
    // zero-based indexing.
    constexpr uint32_t descriptorAlignEntries = ALIGN_32 / sizeof(int32_t);
    uint32_t descriptorCacheEntries = maxOwnedTokens * topK;
    if (remainingScratchEntries < descriptorAlignEntries - 1U ||
        descriptorCacheEntries > remainingScratchEntries - (descriptorAlignEntries - 1U)) {
        return false;
    }
    descriptorCacheEntries += descriptorAlignEntries - 1U;
    uint32_t counterEntries = expertPerRank * maxTilesPerExpert;
    uint32_t slotBytes = Ops::Base::CeilAlign(sourceRowBytes, static_cast<uint32_t>(ALIGN_32));
    uint32_t countBytes =
        Ops::Base::CeilAlign(counterEntries * static_cast<uint32_t>(sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t scratchBytes =
        Ops::Base::CeilAlign((fixedScratchEntries + descriptorCacheEntries) * static_cast<uint32_t>(sizeof(int32_t)),
                             static_cast<uint32_t>(ALIGN_32));
    if (slotBytes > DISPATCH_FANOUT_UB_BYTES / DISPATCH_FANOUT_BUFFER_NUM) {
        return false;
    }
    // Keep the two tables physically independent: countBytes is the live
    // tile-ready accumulator, while the conservative second-table allowance
    // covers the runtime priority heads. Subtractive capacity checks avoid
    // both device-side wide arithmetic and unsigned sum overflow.
    uint32_t remainingBytes = DISPATCH_FANOUT_UB_BYTES - DISPATCH_FANOUT_BUFFER_NUM * slotBytes;
    if (countBytes > remainingBytes) {
        return false;
    }
    remainingBytes -= countBytes;
    if (countBytes > remainingBytes) {
        return false;
    }
    remainingBytes -= countBytes;
    return scratchBytes <= remainingBytes;
}

// A fanout launch pays one descriptor build/read per route.  Enable it only
// when the runtime route population proves both enough duplicate payload
// reads and enough work per physical AIV to amortize that plan.  Tiny launches
// therefore fall through naturally without naming BS, H, topK, or a device.
__aicore__ inline bool HasProfitableDispatchFanoutWork(uint64_t localRoutes, uint64_t sourceTokens,
                                                       uint32_t workerCount)
{
    if (workerCount == 0U || localRoutes <= sourceTokens) {
        return false;
    }
    uint64_t duplicateRoutes = localRoutes - sourceTokens;
    uint32_t minimumRoutes = workerCount <= UINT32_MAX / 8U ? workerCount * 8U : UINT32_MAX;
    uint32_t minimumDuplicates = workerCount <= UINT32_MAX / 2U ? workerCount * 2U : UINT32_MAX;
    return localRoutes >= minimumRoutes && duplicateRoutes >= minimumDuplicates;
}

// C7 bulk Combine only needs the canonical expert-major row and the original
// top-k slot.  The physical expert is implicit in the schedule-ordered append
// position and is not needed once the complete GMM2 stage is visible.  Keeping
// this descriptor narrower than the legacy experiment makes the capability
// check independent of expert count and preserves one scalar store per route.
__aicore__ inline uint32_t GetBulkTokenChainPayloadBits(uint32_t topK)
{
    return GetTokenOwnerFieldBits(topK);
}

__aicore__ inline bool CanPackBulkTokenChainDescriptor(uint32_t topK, uint64_t maxOutputSize)
{
    if (topK == 0U || maxOutputSize == 0U) {
        return false;
    }
    uint32_t payloadBits = GetBulkTokenChainPayloadBits(topK);
    if (payloadBits >= 32U) {
        return false;
    }
    uint64_t maxCanonicalRow = static_cast<uint64_t>(UINT32_MAX >> payloadBits);
    return maxOutputSize - 1U <= maxCanonicalRow;
}

// GMM2 can publish one 256-row M-group for a bounded one/two-wave grid, and
// for an explicitly enabled long expert in a real hot/cold schedule.
// Both producer and consumer must use this exact predicate: a mismatch would
// make Combine wait on a counter that GMM2 never publishes.
__aicore__ inline uint32_t GetGmm2MGroupCount(uint32_t rows)
{
    return Ops::Base::CeilDiv(rows, EXPERT_SCHEDULE_M_TILE_ROWS);
}

// Count only groups with enough rows to amortize an independent ready/consume
// protocol.  A tiny ragged tail (for example row 513 in [256, 256, 1]) is
// deliberately excluded from independent worker-group ownership instead of
// being treated as a third unit of useful parallelism.
__aicore__ inline uint32_t GetMaterialMGroupCount(uint32_t rows)
{
    uint32_t fullGroups = rows / EXPERT_SCHEDULE_M_TILE_ROWS;
    uint32_t tailRows = rows % EXPERT_SCHEDULE_M_TILE_ROWS;
    constexpr uint32_t MIN_MATERIAL_TAIL_ROWS = EXPERT_SCHEDULE_M_TILE_ROWS / 4U;
    return fullGroups + (tailRows >= MIN_MATERIAL_TAIL_ROWS ? 1U : 0U);
}

__aicore__ inline uint32_t GetGmm2NTileCount(uint32_t columns)
{
    return Ops::Base::CeilDiv(columns, GMM2_READY_N_TILE_COLUMNS);
}

// The aggregate ready counter remains in lane0.  Stripe streaming therefore
// uses only the 15 spare lanes in the existing GMM2 cacheline and falls back
// automatically for wider outputs.  This is a layout-capacity predicate, not
// a model-shape or device-SKU special case, and is shared by producer/consumer.
__aicore__ inline bool CanUseGmm2StripeReady(uint32_t columns)
{
    uint32_t nTiles = GetGmm2NTileCount(columns);
    return nTiles > 1U && nTiles <= GMM2_STRIPE_READY_CAPACITY;
}

// Full-A stripe streaming is useful only when the material M-group grid keeps
// every AIC occupied and the complete reduction tile is resident in the six
// K256 A stages.  AIC and AIV call this same helper, preventing a producer /
// consumer mode mismatch from turning a performance fallback into a deadlock.
__aicore__ inline bool CanUseFullAGmm2StripeReady(uint32_t outputColumns, uint32_t reductionColumns,
                                                  uint32_t totalMGroups, uint32_t blockNum)
{
    if (blockNum == 0U || totalMGroups < blockNum || !CanUseGmm2StripeReady(outputColumns)) {
        return false;
    }
    return Ops::Base::CeilDiv(reductionColumns, GMM2_FULL_A_K_TILE_COLUMNS) == GMM2_FULL_A_K_STAGES;
}

// Expert-major keeps its whole-token dependency counter. Stream aligned
// 512-column batches only when the resident producer can publish the existing
// per-N256 lanes. Publication follows completed output stores and is independent
// of the number of L1 A stages: both rotating A3 and Full-A producers support it.
// Both sides use this predicate; no catalog shape/EP is used.
constexpr uint32_t EXPERT_COMBINE_STRIPE_COLUMNS = 2U * GMM2_READY_N_TILE_COLUMNS;
__aicore__ inline bool CanUseExpertMajorGmm2StripeReady(uint32_t outputColumns, uint32_t reductionColumns,
                                                        uint32_t totalMGroups, uint32_t blockNum)
{
    return blockNum > 0U && totalMGroups >= blockNum && reductionColumns > 0U &&
           outputColumns >= 2U * EXPERT_COMBINE_STRIPE_COLUMNS && outputColumns % EXPERT_COMBINE_STRIPE_COLUMNS == 0U &&
           CanUseGmm2StripeReady(outputColumns);
}

__aicore__ inline uint32_t GetGmm2ReadyWaveCount(uint32_t rows, uint32_t columns, uint32_t blockNum,
                                                 bool allowLongStreaming = false)
{
    uint32_t mGroups = GetGmm2MGroupCount(rows);
    uint32_t nTiles = GetGmm2NTileCount(columns);
    uint64_t tileCount = static_cast<uint64_t>(mGroups) * nTiles;
    if (tileCount == 0U || blockNum == 0U || nTiles > blockNum) {
        return 0U;
    }
    if (tileCount <= blockNum) {
        return 1U;
    }
    // A sparse ordinary second wave puts FIX_S + AtomicAdd in every producer
    // tile but exposes little overlap. Dense two-wave mode therefore requires
    // half-wave occupancy and a material final M-group; tiny tails use the
    // separate hybrid prefix below.
    uint32_t secondWaveTiles = static_cast<uint32_t>(tileCount - blockNum);
    uint32_t minimumSecondWaveTiles = (blockNum + 1U) / 2U;
    uint32_t finalMGroupRows = rows % EXPERT_SCHEDULE_M_TILE_ROWS;
    if (finalMGroupRows == 0U) {
        finalMGroupRows = EXPERT_SCHEDULE_M_TILE_ROWS;
    }
    uint32_t materialMGroups = GetMaterialMGroupCount(rows);
    // Hybrid ragged mode: schedule all complete/material groups in the first
    // wave, then leave the tiny tail in the ordinary suffix.  Combine uses
    // whole-expert quotient ownership for this mode, so only the core owning
    // the tail row waits for its late counter; the tail is never assigned an
    // independent worker group.
    bool hasTinyTail = materialMGroups >= 2U && materialMGroups < mGroups;
    if (hasTinyTail && static_cast<uint64_t>(materialMGroups) * nTiles <= blockNum) {
        return 2U;
    }
    // Ordinary two-wave streaming requires a material suffix. Tiny tails have
    // already returned through hybrid mode and are never considered here.
    if (tileCount <= static_cast<uint64_t>(blockNum) * 2U && secondWaveTiles >= minimumSecondWaveTiles &&
        finalMGroupRows * 4U >= EXPERT_SCHEDULE_M_TILE_ROWS) {
        return 2U;
    }
    // A real hot/cold schedule has enough independent expert work to amortize
    // one completion vote per output tile. Expose a complete long hot expert
    // one M-group at a time instead of making Combine wait for the full expert.
    // Unscheduled identity traversal retains the lower-overhead full-ready
    // protocol for long experts.
    if (allowLongStreaming && materialMGroups >= 3U) {
        return 3U;
    }
    return 0U;
}

// Number of complete leading M-groups that can finish within one physical
// AIC wave.  GroupMatmul2 places this prefix first only for a multi-wave
// group-ready expert; the remaining grid keeps the established Nz swizzle.
// The tile count and startBlock phase are unchanged.
__aicore__ inline uint32_t GetGmm2ReadyPrefixGroupCount(uint32_t rows, uint32_t columns, uint32_t blockNum,
                                                        uint32_t readyWaveCount)
{
    if (readyWaveCount < 2U || blockNum == 0U) {
        return 0U;
    }
    uint32_t mGroups = GetGmm2MGroupCount(rows);
    uint32_t nTiles = GetGmm2NTileCount(columns);
    if (mGroups <= 1U || nTiles == 0U || nTiles > blockNum) {
        return 0U;
    }
    uint32_t prefixGroups = blockNum / nTiles;
    uint32_t completeGroups = mGroups - 1U;
    return prefixGroups < completeGroups ? prefixGroups : completeGroups;
}

__aicore__ inline bool UseGmm2MGroupReady(uint32_t rows, uint32_t columns, uint32_t blockNum,
                                          bool allowLongStreaming = false)
{
    return GetGmm2ReadyWaveCount(rows, columns, blockNum, allowLongStreaming) != 0U;
}

struct PeermemInfo {
    GM_ADDR ptrBase;
    GM_ADDR ptrTokenPerExpert;
    GM_ADDR ptrExpertTokenCumsum;
    GM_ADDR offsetExpandedRowIdx;
    GM_ADDR offsetExpertProb;
    GM_ADDR ptrA0;  // initrouting结果，包括data+scale
    GM_ADDR ptrD;

    __aicore__ inline PeermemInfo() = default;
    __aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        ptrBase = base;
        int64_t offset = PEERMEM_DATA_OFFSET;
        ptrTokenPerExpert = base + offset;
        offset += tilingData->epWorldSize *
                  Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize * tilingData->expertPerRank),
                                       (int64_t)ALIGN_128) *
                  sizeof(int32_t);
        ptrExpertTokenCumsum = base + offset;
        offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->expertPerRank,
                                       (int64_t)ALIGN_128) *
                  sizeof(int32_t);
        offsetExpandedRowIdx = base + offset;
        offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->m * tilingData->topK,
                                       static_cast<int64_t>(ALIGN_128)) *
                  sizeof(int32_t);
        offsetExpertProb = base + offset;
        offset += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->epWorldSize) * tilingData->m * tilingData->topK,
                                       static_cast<int64_t>(ALIGN_256)) *
                  sizeof(int16_t);
        ptrA0 = base + offset;
        offset += tilingData->m * GetA0RowStrideByMode(tilingData->k, tilingData->noQuant) * sizeof(int8_t);
        ptrD = base + offset;
    }
};

struct Params {
    GM_ADDR aGmAddr;
    GM_ADDR expertIdxGmAddr;
    GM_ADDR bGmAddr;
    GM_ADDR bScaleGmAddr;
    GM_ADDR b2GmAddr;
    GM_ADDR b2ScaleGmAddr;
    GM_ADDR probsGmAddr;
    GM_ADDR y2GmAddr;
    GM_ADDR expertTokenNumsOutGmAddr;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;
    MegaMoeTilingData *tilingData;
};

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

#endif
