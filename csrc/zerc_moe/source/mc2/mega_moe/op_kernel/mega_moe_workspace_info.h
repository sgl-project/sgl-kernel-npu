/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_workspace_info.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t *
#define HOST_DEVICE
#endif

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_RESERVED_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t MAX_FFTS_USER_FLAG = 7;
// AIV1 owns the same physical UB bank used by the AIC GMM2 Fixpipe path.
// The forward slot transfers ownership after Dispatch and after every active
// expert's Combine. The reverse slot transfers the bank back only after the
// paired AIC has destroyed its GMM2 context. Both directions therefore form a
// strict one-credit ping-pong rather than allowing either side to touch the
// bank while the other side still owns it. +FLAG_ID_MAX selects sub-block 1;
// offset 1 maps to forward raw flag 7 and reverse raw flag 5. Raw flags 8-10
// are reserved for barriers.
constexpr uint16_t GMM2_UB_READY_FLAG = 1;
constexpr uint16_t GMM2_UB_RELEASE_FLAG = 1;
static_assert(AIV_SYNC_AIC_FLAG + GMM2_UB_READY_FLAG <= MAX_FFTS_USER_FLAG,
              "GMM2 UB ownership handoff must use an ordinary FFTS flag");
static_assert(AIC_SYNC_AIV_FLAG + GMM2_UB_RELEASE_FLAG <= MAX_FFTS_USER_FLAG,
              "GMM2 UB ownership release must use an ordinary FFTS flag");
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t PEERMEM_DATA_OFFSET = 1024 * 60;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int32_t INT_CACHELINE = 16;
// chainLenReady lane0 retains its established generic/owner-plan meaning.
// Core-wave uses one otherwise-unused lane to join the final Dispatch work
// without pulling AIC GMM2 into a mixed-core phase barrier.
constexpr uint32_t CORE_WAVE_DISPATCH_READY_OFFSET = 1U;
static_assert(CORE_WAVE_DISPATCH_READY_OFFSET < static_cast<uint32_t>(INT_CACHELINE),
              "core-wave Dispatch completion must stay inside chainLenReady");
// Every 256-row M-group owns disjoint cachelines for its two producer phases:
// Dispatch row completion, then GMM2 N-tile completion.  Each phase uses
// lane0 as its aggregate counter.  Keeping the phases physically separate
// prevents a Dispatch count from looking like an already-ready GMM2 tile.
constexpr uint32_t DISPATCH_TILE_READY_CACHELINES = 2U;
constexpr uint32_t DISPATCH_TILE_READY_STRIDE = DISPATCH_TILE_READY_CACHELINES * static_cast<uint32_t>(INT_CACHELINE);
constexpr uint32_t DISPATCH_READY_OFFSET = 0U;
constexpr uint32_t GMM2_READY_OFFSET = static_cast<uint32_t>(INT_CACHELINE);
// GMM2 lane0 keeps the established whole-M-group completion counter so every
// legacy consumer remains a valid fallback.  The remaining lanes identify
// individual 256-column stripes: lane 1+t is published only after stripe t's
// Fixpipe stores are globally visible.  This reuses already-reset workspace;
// no launch-persistent state or extra allocation is introduced.
constexpr uint32_t GMM2_STRIPE_READY_OFFSET = GMM2_READY_OFFSET + 1U;
constexpr uint32_t GMM2_STRIPE_READY_CAPACITY = static_cast<uint32_t>(INT_CACHELINE) - 1U;
constexpr uint32_t GMM2_READY_N_TILE_COLUMNS = 256U;
constexpr uint32_t GMM2_FULL_A_K_TILE_COLUMNS = 256U;
constexpr uint32_t GMM2_FULL_A_K_STAGES = 6U;
constexpr uint32_t BF16_GMM2_L1_K_COLUMNS = 256U;
constexpr uint32_t BF16_GMM2_L1_A_STAGES = 2U;

HOST_DEVICE constexpr bool CanUseBf16Gmm2L1Reuse(uint32_t columns, uint32_t reduction)
{
    // Both existing A stages must hold the complete K sweep. With one K
    // stage, the ping-pong cursor would move to the other slot on reuse.
    return reduction > BF16_GMM2_L1_K_COLUMNS && reduction <= BF16_GMM2_L1_K_COLUMNS * BF16_GMM2_L1_A_STAGES &&
           columns >= 2U * GMM2_READY_N_TILE_COLUMNS && columns % GMM2_READY_N_TILE_COLUMNS == 0U;
}

struct Bf16Gmm2NLocalTile {
    uint32_t mGroup = 0U;
    uint32_t nBegin = 0U;
    uint32_t tileCount = 0U;
};

HOST_DEVICE constexpr Bf16Gmm2NLocalTile GetBf16Gmm2NLocalTile(uint32_t rows, uint32_t columns, uint32_t cores,
                                                               uint32_t logicalCore)
{
    if (cores == 0U || logicalCore >= cores || rows == 0U || rows % EXPERT_SCHEDULE_M_TILE_ROWS != 0U ||
        columns == 0U || columns % GMM2_READY_N_TILE_COLUMNS != 0U) {
        return {};
    }
    const uint32_t groups = rows / EXPERT_SCHEDULE_M_TILE_ROWS;
    const uint32_t nTiles = columns / GMM2_READY_N_TILE_COLUMNS;
    const uint64_t tiles = static_cast<uint64_t>(groups) * nTiles;
    if (groups >= cores || tiles <= cores || tiles > 2ULL * cores) {
        return {};
    }
    const uint32_t workers = cores / groups;
    const uint32_t wideGroups = cores % groups;
    if (nTiles > 2U * workers) {
        return {};
    }

    // Keep exactly the same one/two-tile workload on each logical core as
    // the strided scheduler. Only tile coordinates change: two-tile owners
    // receive adjacent N tiles of one M group, retaining A in both L1 slots.
    const uint32_t pairs = static_cast<uint32_t>(tiles - cores);
    const uint32_t widePairs = nTiles - (workers + 1U);
    const uint32_t narrowPairs = nTiles - workers;
    if (logicalCore < pairs) {
        const uint32_t widePairCores = wideGroups * widePairs;
        if (logicalCore < widePairCores) {
            return {logicalCore / widePairs, 2U * (logicalCore % widePairs), 2U};
        }
        const uint32_t local = logicalCore - widePairCores;
        return {wideGroups + local / narrowPairs, 2U * (local % narrowPairs), 2U};
    }
    const uint32_t single = logicalCore - pairs;
    const uint32_t wideSingles = 2U * (workers + 1U) - nTiles;
    const uint32_t narrowSingles = 2U * workers - nTiles;
    const uint32_t wideSingleCores = wideGroups * wideSingles;
    if (single < wideSingleCores) {
        return {single / wideSingles, 2U * widePairs + single % wideSingles, 1U};
    }
    const uint32_t local = single - wideSingleCores;
    return {wideGroups + local / narrowSingles, 2U * narrowPairs + local % narrowSingles, 1U};
}

// The ragged-N tiling key uses the last, otherwise unused stripe lane as a
// monotonic bitset. Lane0 remains the whole-M-group count for every fallback.
// All active lanes, including this one, are cleared by the existing reset.
constexpr uint32_t GMM2_STRIPE_MASK_OFFSET = GMM2_STRIPE_READY_OFFSET + GMM2_STRIPE_READY_CAPACITY - 1U;
HOST_DEVICE constexpr bool CanUseRaggedGmm2StripeMask(uint32_t outputColumns, uint32_t reductionColumns)
{
    // A half N tile is naturally 128-column/vector aligned but carries a
    // separate release/probe/DMA setup in the per-stripe path. Require a
    // material N sweep and reserve the last ready lane; no EP/BS/E or SKU
    // value participates. The reduction bound is the existing A6 capacity.
    uint32_t nTiles =
        outputColumns / GMM2_READY_N_TILE_COLUMNS + (outputColumns % GMM2_READY_N_TILE_COLUMNS != 0U ? 1U : 0U);
    return outputColumns % GMM2_READY_N_TILE_COLUMNS == GMM2_READY_N_TILE_COLUMNS / 2U && nTiles >= 8U &&
           nTiles < GMM2_STRIPE_READY_CAPACITY &&
           reductionColumns > (GMM2_FULL_A_K_STAGES - 1U) * GMM2_FULL_A_K_TILE_COLUMNS &&
           reductionColumns <= GMM2_FULL_A_K_STAGES * GMM2_FULL_A_K_TILE_COLUMNS;
}
HOST_DEVICE constexpr bool ShouldPublishRaggedGmm2Batch(uint32_t nTile, uint32_t jobTileEnd, uint32_t totalNTiles,
                                                        uint32_t pendingTiles)
{
    // Preserve first-column latency. Thereafter amortize release over two
    // tiles and fold an otherwise isolated final half-tile into that batch.
    // A split job always flushes at its own end: never wait for another AIC.
    return nTile == 0U || nTile + 1U == jobTileEnd || (pendingTiles >= 2U && nTile + 2U < totalNTiles);
}
static_assert(GMM2_STRIPE_READY_OFFSET + GMM2_STRIPE_READY_CAPACITY ==
                  GMM2_READY_OFFSET + static_cast<uint32_t>(INT_CACHELINE),
              "GMM2 stripe readiness must fit the existing second cacheline");
static_assert(GMM2_READY_OFFSET + static_cast<uint32_t>(INT_CACHELINE) == DISPATCH_TILE_READY_STRIDE,
              "Dispatch and GMM2 ready lines must exactly fill one M-group stride");
static_assert(GMM2_STRIPE_MASK_OFFSET < DISPATCH_TILE_READY_STRIDE && GMM2_STRIPE_READY_CAPACITY < 31U,
              "The ready mask must fit its reset lane and a positive int32_t");
// Dispatch completion is one cache-line counter per expert.  A previous
// unique-lane experiment made one AIV scan every worker lane before opening a
// second gate.  That serialized every expert boundary and showed up directly
// as AIV Scalar time, so retain the established post-MTE3 AtomicAdd counter.
constexpr uint32_t DISPATCH_DONE_STRIDE = static_cast<uint32_t>(INT_CACHELINE);
// Keep the established generic/reset/Combine capacity unchanged so enabling
// the C5 Dispatch optimization cannot silently retile unrelated paths.
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
// MegaMoe is compiled only for Ascend950 (arch 3510), but the physical
// 248-KiB capacity need not be occupied by this stage.  The proven 192-KiB
// arena fits the two-row ring, compact tables and descriptor cache for the
// supported shapes while avoiding an unnecessary full-UB reservation.
constexpr uint32_t DISPATCH_FANOUT_UB_BYTES = BUFFER_ALIGN;
static_assert(DISPATCH_FANOUT_UB_BYTES >= BUFFER_ALIGN &&
                  DISPATCH_FANOUT_UB_BYTES % static_cast<uint32_t>(ALIGN_32) == 0U,
              "Ascend950 token-local Dispatch arena must cover legacy UB and stay block aligned");
constexpr uint32_t DISPATCH_COPY_MAX_BUFFER_NUM = 4U;
constexpr uint32_t DISPATCH_COPY_FALLBACK_BUFFER_NUM = 2U;
// Token-local Dispatch keeps two source rows in flight.  A four-row ring can
// enqueue dozens of scattered data/scale MTE3 stores for one K16 token before
// the first slot is reclaimed, delaying ready publication and contending with
// GMM1 MTE2.  The remainder of
// CopyTempBuf_ is reused as a launch-local expert/tile count table plus a
// short touched-bucket list; no additional UB or workspace is reserved.
constexpr uint32_t DISPATCH_FANOUT_BUFFER_NUM = 2U;
constexpr uint32_t DISPATCH_FANOUT_READY_BATCH_ROUTES = 128U;
static_assert(BUFFER_ALIGN % DISPATCH_COPY_MAX_BUFFER_NUM == 0U &&
                  BUFFER_ALIGN % DISPATCH_COPY_FALLBACK_BUFFER_NUM == 0U,
              "Every adaptive Dispatch ring must divide the fixed UB buffer");
constexpr uint32_t DISPATCH_COPY_MIN_SLOT_BYTES = BUFFER_ALIGN / DISPATCH_COPY_MAX_BUFFER_NUM;
constexpr uint32_t DISPATCH_COPY_FALLBACK_SLOT_BYTES = BUFFER_ALIGN / DISPATCH_COPY_FALLBACK_BUFFER_NUM;
static_assert(DISPATCH_COPY_MIN_SLOT_BYTES % static_cast<uint32_t>(ALIGN_32) == 0U,
              "Every adaptive Dispatch slot must remain GM-block aligned");
static_assert(DISPATCH_COPY_FALLBACK_SLOT_BYTES % static_cast<uint32_t>(ALIGN_32) == 0U,
              "Every fallback Dispatch slot must remain GM-block aligned");
static_assert(DISPATCH_COPY_MAX_BUFFER_NUM <= 4U, "Adaptive Dispatch uses only MTE3_MTE2 event ids 0 through 3");
static_assert(DISPATCH_FANOUT_BUFFER_NUM <= 2U, "Token-local Dispatch uses only MTE3_MTE2 event ids 0 and 1");
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
// Aggregate header appended to the expert-plan scratch table.  The plan
// builder publishes these once per launch, next to the endpoint table it
// already writes; every value applies GetExpertRange's exact maxOutputSize
// clamp so consumers replace their per-expert rescans with single reads.
constexpr uint32_t EXPERT_PLAN_HEADER_TOTAL_MGROUPS = 0U;
constexpr uint32_t EXPERT_PLAN_HEADER_TOTAL_FULL_MGROUPS = 1U;
constexpr uint32_t EXPERT_PLAN_HEADER_TOTAL_TAIL_ROWS = 2U;
constexpr uint32_t EXPERT_PLAN_HEADER_MAX_ACTIVE_TILES = 3U;
constexpr uint32_t EXPERT_PLAN_HEADER_INTS = 4U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 4U;  // 2行累加缓冲 + 2行乒乓输入, 支持Unpermute输出两行批量写
constexpr int64_t SIZE_INT_8 = 1U;
// a0(peermem内Mxfp8DynamicQuant输出, data+scale同行存放)的行步长, 与master分支
// mega_moe_wave.h同款约定: Align256(token) + Align32(scale)。
// - 数据段必须补齐到256B: quant的ComputeFp8Data每循环固定写4*64B=256B(尾部溢出段
//   按掩码补0), cols非256倍数时溢出段[cols, CeilAlign(cols,256))会覆盖紧贴cols
//   放置的scale(H=3200实测: 100B scale全被写成0), scale必须放在溢出段之后。
//   dispatch_v2::QuantDynamicMxFp8同样把scale放在outLocal[Align256(axisH_)]处。
// - scale段补齐到32B: 保证行内scale的GM地址与GM->UB搬运长度满足32B对齐
//   (非对齐GM->UB搬运在本芯片实测结果为全0, 与isPad无关)。
// H=2048/3072/4096/5120/7168均为256倍数且scaleLen为32倍数, 行步长与紧排(k+scaleLen)完全一致。
HOST_DEVICE int64_t GetA0RowStride(int64_t k)
{
    // Ops::Base::CeilDiv 是同型模板, 除数需与 k 同为 int64_t (混型无法推导重载)。
    int64_t scaleLen = Ops::Base::CeilDiv(k, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int64_t scaleLenAlign = Ops::Base::CeilAlign(scaleLen, ALIGN_32);
    return Ops::Base::CeilAlign(k, ALIGN_256) + scaleLenAlign;
}

// a0行内scale段的起始偏移, 即补齐到256B后的数据段长度。
HOST_DEVICE int64_t GetA0ScaleOffset(int64_t k)
{
    return Ops::Base::CeilAlign(k, ALIGN_256);
}
constexpr int64_t SIZE_INT_16 = 2U;
constexpr int64_t SIZE_INT_32 = 4U;

// BF16 (no-quant) a0 行只有数据段: k 个 BF16 元素补齐到 32B。host 要求
// H % 16 == 0, 因此行步长恰为 2H 字节, 与 x 的行布局一致, GM->UB 按行对齐。
HOST_DEVICE int64_t GetA0RowStrideBf16(int64_t k)
{
    return Ops::Base::CeilAlign(k * SIZE_INT_16, ALIGN_32);
}

HOST_DEVICE int64_t GetA0RowStrideByMode(int64_t k, bool noQuant)
{
    return noQuant ? GetA0RowStrideBf16(k) : GetA0RowStride(k);
}
}  // namespace

struct WorkspaceInfo {
    GM_ADDR ptrA;
    GM_ADDR ptrAScale;
    GM_ADDR ptrA2;
    GM_ADDR ptrA2Scale;

    GM_ADDR ptrFlagSwiGluToGmm2;
    GM_ADDR ptrDispatchDoneCores;
    // [expert][256-row tile][2 cache lines].  The first line publishes
    // Dispatch row visibility (aggregate gate plus optional contributor
    // lanes); the second independently publishes GMM2 N-tile completion.
    // Core-wave resets only this launch's clamped per-expert tile prefix;
    // unused capacity may retain stale values but is never addressed.
    GM_ADDR ptrDispatchTileReady;
    GM_ADDR ptrFirstTokenPos;
    // [maxOutputSize] int32: raw expert-major stepK, or the capability-gated
    // packed (sourceRank, tokenRow, topKSlot, stepK) at the canonical row.
    GM_ADDR ptrStepIdx;
    // [worldSize*m][topK] int32: token-local Dispatch inverse route map.
    // Each original top-k slot stores schedule position, expert-local 256-row
    // tile, and canonical expert-major row. Unsupported or fallback launches
    // leave this internal workspace unused by the production payload path.
    GM_ADDR ptrTokenChainDescriptors;
    GM_ADDR ptrPosLen;    // [w][m] int32: token local-chain length / append cursor
    GM_ADDR ptrPosDone;   // [w][m] int32: completed expert-major steps
    GM_ADDR ptrChainLen;  // [w][m] int32: expert-major本地链长, AllGather后由AIV0预计算
    // int32: generic=AIV0 chainLen票; core-wave=AIV1全Dispatch完成票
    GM_ADDR ptrChainLenReadyFlag;
    GM_ADDR ptrCombineAccum;  // 合并缓冲区: GMM2直接写入 + combine累加到firstPos
    GM_ADDR ptrGmm2ReadyFlag;
    GM_ADDR ptrExpertOrder;  // [e] int32: schedule position -> physical local expert id
    // [1+2e+header] int32: slot0 publishes schedule mode; slot1..E publish
    // physical-expert local cumulative endpoints.  The next E slots publish
    // physical-expert -> schedule-position for K16 SIMT Dispatch planning;
    // other pipeline specializations leave that suffix unused.  The trailing
    // EXPERT_PLAN_HEADER_INTS slots publish launch aggregates (M-group
    // totals, tail rows, max per-expert tiles) with GetExpertRange's
    // maxOutputSize clamp.
    GM_ADDR ptrExpertOrderScratch;

    // 共享专家（不参与 Dispatch/Combine）：输入（data/scale 分离，与 ptrA/ptrAScale
    // 同构）、activation（与 ptrA2/ptrA2Scale 同构）、GMM2 输出行（BF16），以及每个
    // 共享专家一个就绪标志。未启用时全部为零长度区域，保证既有布局逐字节不变。
    GM_ADDR ptrSharedInputData;
    GM_ADDR ptrSharedInputScale;
    GM_ADDR ptrSharedActivation;
    GM_ADDR ptrSharedActivationScale;
    GM_ADDR ptrSharedOutput;
    GM_ADDR ptrSharedReadyFlag;

    GM_ADDR ptrInitRoutingWorkspace;
    int64_t workspaceSize;

    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        int64_t scaleLen = 0;
        // MXFP8 dispatches one-byte activations plus MX scales; no-quant
        // dispatches BF16 activations and needs no scale regions.
        int64_t actElemBytes = tilingData->noQuant ? SIZE_INT_16 : SIZE_INT_8;
        workspaceSize = 0;
        ptrInitRoutingWorkspace = base + workspaceSize;
        ptrA = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(actElemBytes * tilingData->maxOutputSize * tilingData->k, ALIGN_512);
        ptrAScale = base + workspaceSize;
        if (!tilingData->noQuant) {
            scaleLen =
                Ops::Base::CeilDiv(tilingData->k, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * scaleLen, ALIGN_512);
        }
        ptrA2 = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(actElemBytes * tilingData->maxOutputSize * tilingData->n / SWIGLU_N_HALF, ALIGN_512);
        ptrA2Scale = base + workspaceSize;
        if (!tilingData->noQuant) {
            scaleLen = Ops::Base::CeilDiv(tilingData->n / SWIGLU_N_HALF, static_cast<uint32_t>(MXFP_DIVISOR_SIZE)) *
                       MXFP_MULTI_BASE_SIZE;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * scaleLen, ALIGN_512);
        }

        ptrFirstTokenPos = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize * tilingData->m, ALIGN_512);
        ptrStepIdx = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->maxOutputSize, ALIGN_512);
        ptrTokenChainDescriptors = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_32 * static_cast<int64_t>(tilingData->epWorldSize) *
                                     static_cast<int64_t>(tilingData->m) * static_cast<int64_t>(tilingData->topK),
                                 ALIGN_512);
        ptrPosLen = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize * tilingData->m, ALIGN_512);
        ptrPosDone = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize * tilingData->m, ALIGN_512);
        ptrChainLen = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->epWorldSize * tilingData->m, ALIGN_512);
        ptrChainLenReadyFlag = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * INT_CACHELINE, ALIGN_128);
        ptrCombineAccum = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_16 * tilingData->maxOutputSize * tilingData->k,
                                              ALIGN_512);  // BF16存储，合并缓冲区: GMM2直接写入 + combine累加到firstPos

        ptrFlagSwiGluToGmm2 = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE, ALIGN_128);
        ptrDispatchDoneCores = base + workspaceSize;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->expertPerRank * DISPATCH_DONE_STRIDE, ALIGN_128);
        ptrDispatchTileReady = base + workspaceSize;
        int64_t maxDispatchTilesPerExpert = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize),
                                                               static_cast<int64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * static_cast<int64_t>(tilingData->expertPerRank) *
                                                  maxDispatchTilesPerExpert * DISPATCH_TILE_READY_STRIDE,
                                              ALIGN_128);
        ptrGmm2ReadyFlag = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * tilingData->expertPerRank * INT_CACHELINE, ALIGN_128);
        ptrExpertOrder = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * static_cast<int64_t>(tilingData->expertPerRank), ALIGN_512);
        ptrExpertOrderScratch = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_32 * (2 * static_cast<int64_t>(tilingData->expertPerRank) + 1 +
                                                             static_cast<int64_t>(EXPERT_PLAN_HEADER_INTS)),
                                              ALIGN_512);

        // 共享专家输入：本卡全部 token 的量化结果，data/scale 分离存放（GMM1 的 A
        // 需要连续行步长，与 a0 的行内交错布局不同），由 PrepareSharedExpertInput
        // 从 a0 去交错拷入。BF16 no-quant 不经过此区域：x 本身就是 [m,k] 连续
        // BF16 行，共享 GMM1 直接读 x，无需去交错也无需拷贝。
        ptrSharedInputData = base + workspaceSize;
        if (!tilingData->noQuant && tilingData->sharedExpertNum > 0U) {
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * static_cast<int64_t>(tilingData->m) * static_cast<int64_t>(tilingData->k), ALIGN_512);
        }
        ptrSharedInputScale = base + workspaceSize;
        if (!tilingData->noQuant && tilingData->sharedExpertNum > 0U) {
            const int64_t sharedInputScaleLen =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->k), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                MXFP_MULTI_BASE_SIZE;
            workspaceSize +=
                Ops::Base::CeilAlign(SIZE_INT_8 * static_cast<int64_t>(tilingData->m) * sharedInputScaleLen, ALIGN_512);
        }
        // 共享专家 activation：与 ptrA2/ptrA2Scale 同构（data 与 scale 分块），
        // 因为共享 epilogue 复用路由 epilogue 的 Init/UpdateGlobalAddr 约定。
        // BF16 无 scale 区，data 按 2 字节/元素计。
        ptrSharedActivation = base + workspaceSize;
        if (tilingData->sharedExpertNum > 0U) {
            const int64_t sharedK = static_cast<int64_t>(tilingData->sharedN) / SWIGLU_N_HALF;
            workspaceSize += Ops::Base::CeilAlign(actElemBytes * static_cast<int64_t>(tilingData->m) * sharedK *
                                                      static_cast<int64_t>(tilingData->sharedExpertNum),
                                                  ALIGN_512);
        }
        ptrSharedActivationScale = base + workspaceSize;
        if (!tilingData->noQuant && tilingData->sharedExpertNum > 0U) {
            const int64_t sharedScaleLen = Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->sharedN) / SWIGLU_N_HALF,
                                                              static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                                           MXFP_MULTI_BASE_SIZE;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * static_cast<int64_t>(tilingData->m) * sharedScaleLen *
                                                      static_cast<int64_t>(tilingData->sharedExpertNum),
                                                  ALIGN_512);
        }
        // GMM2 输出 [sharedExpertNum][m][k] BF16, 供 Unpermute 累加。
        ptrSharedOutput = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_16 * static_cast<int64_t>(tilingData->m) *
                                                  static_cast<int64_t>(tilingData->k) *
                                                  static_cast<int64_t>(tilingData->sharedExpertNum),
                                              ALIGN_512);
        // 每个共享专家占三个标志行：[0, S) 为 GMM1 tile ready（AIC->AIV），
        // [S, 2S) 为 GMM2 完成（AIC->Unpermute 累加），[2S, 3S) 的首行作为共享输入
        // 就绪标志（AIV->AIC）。
        ptrSharedReadyFlag = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_32 * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE * 3, ALIGN_128);
    }
};
#endif  // MEGA_MOE_WORKSPACE_INFO_H
