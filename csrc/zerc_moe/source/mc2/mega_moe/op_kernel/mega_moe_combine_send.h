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
 * \file mega_moe_combine_send.h
 * \brief Combine逻辑：AIV全核按Dispatch/GMM相同的expert顺序并行处理token。
 *        1. 每个expert的连续行按AIV核分成连续区间。
 *        2. 每次最多预取8行 payload，依赖与发布固定按4行检查。
 *        3. 按token依赖: 每个任务等待 posDone[T]==k (本token的前k步完成),
 *        保证BF16位精确累加顺序。UB容量允许时使用奇偶双 accumulator，
 *        仅在下一行依赖已就绪时提前发射前驱GM->UB；未就绪不阻塞当前行。
 *        4. 成本受控的整expert按256行M-group等待GMM2，并把AIV按group行数加权分核。
 *        core-wave 在全量 Dispatch 后直接复用最终 posLen 作为 chainLen，
 *        generic 路径仍由 AIV0 预计算；两者均在首次读取前等待各自的
 *        metadata-ready 票收齐。
 *        逐expert消费无SyncAll，所有核可并行流转，仅等待必要的前序依赖。
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H
#include "catlass/gemm_coord.hpp"
#include "kernel_operator.h"
#include "mega_moe_base.h"
// 须自含 AscendTimerV2_device.hpp: 本文件被 mega_moe_catlass_impl.h 先于其
// timer 头 include, 展开时须已可见 AscendTimerDevice/Overwrite/枚举。
#include "AscendTimerV2_device.hpp"
using namespace AscendC;

namespace MegaMoeCombineImpl {

__aicore__ inline void WaitForGmm2MGroupReady(__gm__ int32_t *readyAddr, int32_t atomicReadyTarget)
{
    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (AscendC::ReadGmByPassDCache(readyAddr) != atomicReadyTarget) {
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
    }
}

template <bool PublishAiv1UbOwnership>
__aicore__ inline void AcquireAiv1UbForCombine()
{
    if constexpr (PublishAiv1UbOwnership) {
        if (GetSubBlockIdx() == 1U) {
            // The paired AIC publishes this only after Gmm2Context has been
            // destroyed, so no Fixpipe Resource/TPipe state can still own the
            // physical AIV1 UB bank when Combine starts using it.
            WaitForCube(GMM2_UB_RELEASE_FLAG);
        }
    }
}

template <bool PublishAiv1UbOwnership>
__aicore__ inline void ReleaseAiv1UbAfterCombine()
{
    if constexpr (PublishAiv1UbOwnership) {
        if (GetSubBlockIdx() == 1U) {
            // CATLASS GMM2 context construction reinitializes the physical
            // AIV1 UB bank. Drain this core's final Combine use before handing
            // the bank back to its paired AIC. GMM2(e+1) consumes this credit
            // only after GMM1(e+1), retaining useful overlap without a global
            // barrier or an all-core atomic counter. The paired AIC cannot
            // construct the next GMM2 context until it consumes this credit.
            PipeBarrier<PIPE_ALL>();
            NotifyCube(GMM2_UB_READY_FLAG);
        }
    }
}

__aicore__ inline void WaitForPosDoneStep(__gm__ int32_t *posDonePtr, int32_t tokenInAllRank, int32_t stepK,
                                          int32_t publishCnt)
{
    __gm__ int32_t *dependency = posDonePtr + tokenInAllRank;
    int32_t observed = ReadGmByPassDCache(dependency);
    if (publishCnt == 1 && observed != stepK) {
        // Decode/small ranges often become ready between adjacent probes.
        observed = ReadGmByPassDCache(dependency);
    }
    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (observed != stepK) {
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
        observed = ReadGmByPassDCache(dependency);
    }
}

__aicore__ inline bool IsPosDoneStepReady(__gm__ int32_t *posDonePtr, int32_t tokenInAllRank, int32_t stepK)
{
    return ReadGmByPassDCache(posDonePtr + tokenInAllRank) == stepK;
}

__aicore__ inline bool CanUseCombinePredecessorPipeline(uint32_t combineRowBytes, uint32_t maxCombineBatch,
                                                        uint32_t routeMetaBytes)
{
    // Keep the existing payload batch unchanged.  The optimized path needs
    // one additional accumulator row plus the aligned route-metadata tail.
    // Division-first checks avoid overflow and device-side wide multiplies.
    if (combineRowBytes == 0U || maxCombineBatch < 2U || combineRowBytes % static_cast<uint32_t>(ALIGN_32) != 0U) {
        return false;
    }
    uint32_t requiredRows = 2U + maxCombineBatch;
    if (requiredRows > BUFFER_ALIGN / combineRowBytes) {
        return false;
    }
    uint32_t payloadEnd = requiredRows * combineRowBytes;
    uint32_t routeMetaOffset = Ops::Base::CeilAlign(payloadEnd, static_cast<uint32_t>(ALIGN_32));
    return routeMetaOffset <= BUFFER_ALIGN && routeMetaBytes <= BUFFER_ALIGN - routeMetaOffset;
}

template <typename ElementC, bool ColumnSlice = false>
__aicore__ inline void ProcessCombineRowRange(
    uint32_t rankId, GlobalTensor<ElementC> combineAccumGmGlobal, GlobalTensor<int32_t> &expandedRowIdx,
    GlobalTensor<bfloat16_t> &distProbs, GlobalTensor<int32_t> &tokenPerExpertWin, __gm__ int32_t *firstTokenPosPtr,
    __gm__ int32_t *stepIdxPtr, __gm__ int32_t *chainLenPtr, __gm__ int32_t *posDonePtr, int32_t globalExpertIdx,
    int32_t expertTokenStart, int32_t rowStart, int32_t rowEnd, int32_t combineBatch, int32_t n, uint32_t worldSize,
    uint32_t m, uint32_t topK, bool packedRouteMetadata, int32_t expertAlign, uint64_t remoteDOffset,
    bool useRouteMetadataCache, bool usePredecessorPipeline, const AscendC::TEventID *combineMte2vEvent,
    const AscendC::TEventID *accumGuardEvent, LocalTensor<ElementC> combineAccumUb,
    LocalTensor<ElementC> combineBatchUb, LocalTensor<int32_t> combineRouteMetaUb, DataCopyExtParams copyParamsBf16,
    DataCopyPadExtParams<ElementC> padParamsBf16, int32_t columnBegin = 0, int32_t columnCount = 0,
    bool publishWholeRow = true)
{
    constexpr int32_t PUBLISH_BATCH = 4;
    if (rowStart >= rowEnd) {
        return;
    }
    int32_t vectorColumns = n;
    bool publishCompletedRows = true;
    if constexpr (ColumnSlice) {
        vectorColumns = columnCount;
        publishCompletedRows = publishWholeRow;
        copyParamsBf16.blockLen = static_cast<uint32_t>(vectorColumns * sizeof(ElementC));
    } else {
        columnBegin = 0;
    }

    uint32_t routeTokenRowBits = GetTokenOwnerFieldBits(m);
    uint32_t routeTokenRowMask = routeTokenRowBits == 0U ? 0U : (UINT32_MAX >> (32U - routeTokenRowBits));
    uint32_t routeTopKBits = GetTokenOwnerFieldBits(topK);
    uint32_t routeTopKMask = routeTopKBits == 0U ? 0U : (UINT32_MAX >> (32U - routeTopKBits));

    // Locate the source-rank segment containing this independent row range.
    // Use a bounded scan so empty leading/trailing source segments cannot make
    // the cursor walk outside the runtime world.
    int32_t curSourceRank = 0;
    int32_t curSegPre = 0;
    int32_t curSegCount = 0;
    int32_t skipped = rowStart;
    if (!packedRouteMetadata) {
        for (; curSourceRank < static_cast<int32_t>(worldSize); ++curSourceRank) {
            curSegPre = globalExpertIdx == 0
                            ? 0
                            : tokenPerExpertWin.GetValue(curSourceRank * expertAlign + globalExpertIdx - 1);
            int32_t curSegEnd = tokenPerExpertWin.GetValue(curSourceRank * expertAlign + globalExpertIdx);
            curSegCount = curSegEnd - curSegPre;
            if (skipped < curSegCount) {
                break;
            }
            skipped -= curSegCount;
        }
        if (curSourceRank >= static_cast<int32_t>(worldSize)) {
            return;
        }
    }
    int32_t segRemaining = packedRouteMetadata ? 0 : curSegCount - skipped;
    int32_t expandedIdxPos =
        packedRouteMetadata ? 0 : curSourceRank * static_cast<int32_t>(m * topK) + curSegPre + skipped;
    GlobalTensor<ElementC> gmRemoteD;
    int32_t remoteDSourceRank = -1;

    for (int32_t copyStart = rowStart; copyStart < rowEnd; copyStart += combineBatch) {
        int32_t copyEnd = copyStart + combineBatch < rowEnd ? copyStart + combineBatch : rowEnd;
        int32_t copyCnt = copyEnd - copyStart;
        constexpr int64_t ROUTE_META_ALIGN_WORDS = static_cast<int64_t>(ALIGN_32 / sizeof(int32_t));
        constexpr int64_t ROUTE_META_CACHE_WORDS = 16;
        constexpr int32_t ROUTE_META_CACHE_MIN_ROWS = 4;
        int64_t routeWinBegin = 0;
        bool cacheThisCopy = false;
        if (useRouteMetadataCache && copyCnt >= ROUTE_META_CACHE_MIN_ROWS) {
            int64_t routeBegin = static_cast<int64_t>(expertTokenStart) + copyStart;
            int64_t routeEnd = static_cast<int64_t>(expertTokenStart) + copyEnd;
            routeWinBegin = routeBegin / ROUTE_META_ALIGN_WORDS * ROUTE_META_ALIGN_WORDS;
            int64_t routeWinEnd =
                (routeEnd + ROUTE_META_ALIGN_WORDS - 1) / ROUTE_META_ALIGN_WORDS * ROUTE_META_ALIGN_WORDS;
            int64_t routeWordCount = routeWinEnd - routeWinBegin;
            if (routeWordCount > 0 && routeWordCount <= ROUTE_META_CACHE_WORDS) {
                GlobalTensor<int32_t> routeMetadataWin;
                routeMetadataWin.SetGlobalBuffer(stepIdxPtr + routeWinBegin);
                DataCopyExtParams routeMetaCopyParams{1U, static_cast<uint32_t>(routeWordCount * sizeof(int32_t)), 0U,
                                                      0U, 0U};
                DataCopyPad(combineRouteMetaUb, routeMetadataWin, routeMetaCopyParams,
                            DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
                // The metadata window is consumed by scalar GetValue below.
                // Complete this small DMA synchronously so no event survives
                // an early return from malformed raw-route metadata.
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                cacheThisCopy = true;
            }
        }
        DataCopyExtParams batchCopyParams{1U, static_cast<uint32_t>(copyCnt * n * sizeof(ElementC)), 0U, 0U, 0U};
        if constexpr (ColumnSlice) {
            // GM retains the full hidden-width row stride; UB packs only
            // this completed column interval. All intervals are 32B aligned.
            batchCopyParams.blockCount = static_cast<uint16_t>(copyCnt);
            batchCopyParams.blockLen = static_cast<uint32_t>(vectorColumns * sizeof(ElementC));
            batchCopyParams.srcStride = static_cast<uint32_t>((n - vectorColumns) * sizeof(ElementC));
        }
        DataCopyPad(combineBatchUb,
                    combineAccumGmGlobal[static_cast<int64_t>(expertTokenStart + copyStart) * n + columnBegin],
                    batchCopyParams, padParamsBf16);
        bool payloadBatchJoined = false;
        if (usePredecessorPipeline) {
            // Keep the payload DMA independent from the two predecessor
            // fetches.  Waiting for slot R must never accidentally drain the
            // event that makes the whole payload batch visible.
            SetFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[0]);
        }

        for (int32_t publishStart = copyStart; publishStart < copyEnd; publishStart += PUBLISH_BATCH) {
            int32_t publishEnd = publishStart + PUBLISH_BATCH < copyEnd ? publishStart + PUBLISH_BATCH : copyEnd;
            int32_t publishCnt = publishEnd - publishStart;
            int32_t publishSourceRank[PUBLISH_BATCH];
            int32_t publishTokenRow[PUBLISH_BATCH];
            int32_t publishTokenInAllRank[PUBLISH_BATCH];
            int32_t publishStepK[PUBLISH_BATCH];
            int32_t publishFirstPos[PUBLISH_BATCH];
            bfloat16_t publishProb[PUBLISH_BATCH];
            uint32_t tailMask = 0U;
            uint32_t posDonePublishMask = 0U;

            // Preload stable route/chain metadata before dependency waits.
            for (int32_t R = publishStart; R < publishEnd; ++R) {
                int32_t publishIdx = R - publishStart;
                int64_t routeMetadataIdx = static_cast<int64_t>(expertTokenStart) + R;
                int32_t routeMetadata =
                    cacheThisCopy ? combineRouteMetaUb.GetValue(static_cast<uint32_t>(routeMetadataIdx - routeWinBegin))
                                  : ReadGmByPassDCache(stepIdxPtr + routeMetadataIdx);
                if (packedRouteMetadata) {
                    uint32_t packed = static_cast<uint32_t>(routeMetadata);
                    uint32_t stepK = routeTopKBits == 0U ? 0U : packed & routeTopKMask;
                    uint32_t topKSlot = routeTopKBits == 0U ? 0U : (packed >> routeTopKBits) & routeTopKMask;
                    uint32_t tokenRow =
                        routeTokenRowBits == 0U ? 0U : (packed >> (2U * routeTopKBits)) & routeTokenRowMask;
                    uint32_t sourceRank = packed >> (routeTokenRowBits + 2U * routeTopKBits);
                    uint32_t tokenInAllRank = sourceRank * m + tokenRow;
                    publishSourceRank[publishIdx] = static_cast<int32_t>(sourceRank);
                    publishTokenRow[publishIdx] = static_cast<int32_t>(tokenRow);
                    publishTokenInAllRank[publishIdx] = static_cast<int32_t>(tokenInAllRank);
                    uint32_t probabilityRow = topK == 8U ? tokenInAllRank << 3U
                                                         : (topK == 16U ? tokenInAllRank << 4U : tokenInAllRank * topK);
                    publishProb[publishIdx] = distProbs.GetValue(probabilityRow + topKSlot);
                    publishStepK[publishIdx] = static_cast<int32_t>(stepK);
                } else {
                    publishSourceRank[publishIdx] = curSourceRank;
                    int32_t expandedIdx = expandedRowIdx.GetValue(expandedIdxPos);
                    ++expandedIdxPos;
                    int32_t tokenRow =
                        topK == 8U ? expandedIdx >> 3
                                   : (topK == 16U ? expandedIdx >> 4 : expandedIdx / static_cast<int32_t>(topK));
                    publishTokenRow[publishIdx] = tokenRow;
                    int32_t tokenInAllRank = curSourceRank * static_cast<int32_t>(m) + tokenRow;
                    publishTokenInAllRank[publishIdx] = tokenInAllRank;
                    publishProb[publishIdx] =
                        distProbs.GetValue(curSourceRank * static_cast<int32_t>(m * topK) + expandedIdx);
                    publishStepK[publishIdx] = routeMetadata;
                }
                int32_t tokenInAllRank = publishTokenInAllRank[publishIdx];
                int32_t chainLen = ReadGmByPassDCache(chainLenPtr + tokenInAllRank);
                // Dispatch defines the first contribution's canonical row as
                // firstTokenPos.  Reconstruct that value locally for step0;
                // only later contributions need the random GM lookup.
                publishFirstPos[publishIdx] = expertTokenStart + R;
                if (publishStepK[publishIdx] > 0) {
                    publishFirstPos[publishIdx] = ReadGmByPassDCache(firstTokenPosPtr + tokenInAllRank);
                }
                uint32_t publishBit = 1U << publishIdx;
                if (publishStepK[publishIdx] == chainLen - 1) {
                    tailMask |= publishBit;
                }
                if (publishStepK[publishIdx] + 1 < chainLen) {
                    posDonePublishMask |= publishBit;
                }

                if (!packedRouteMetadata && --segRemaining == 0 && R + 1 < rowEnd) {
                    do {
                        ++curSourceRank;
                        if (curSourceRank >= static_cast<int32_t>(worldSize)) {
                            // The raw-route cursor is defensive-only, but its
                            // early return must still consume the outstanding
                            // payload event from this copy batch.
                            if (!payloadBatchJoined) {
                                if (usePredecessorPipeline) {
                                    WaitFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[0]);
                                } else {
                                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                                }
                                payloadBatchJoined = true;
                            }
                            return;
                        }
                        curSegPre = globalExpertIdx == 0
                                        ? 0
                                        : tokenPerExpertWin.GetValue(curSourceRank * expertAlign + globalExpertIdx - 1);
                        curSegCount =
                            tokenPerExpertWin.GetValue(curSourceRank * expertAlign + globalExpertIdx) - curSegPre;
                    } while (curSegCount == 0);
                    segRemaining = curSegCount;
                    expandedIdxPos = curSourceRank * static_cast<int32_t>(m * topK) + curSegPre;
                }
            }

            int32_t R = publishStart;
            bool fetchLive[2] = {false, false};
            bool guardLive[2] = {false, false};
            while (R < publishEnd) {
                int32_t publishIdx = R - publishStart;
                int32_t slot = R & 1;
                if (publishStepK[publishIdx] > 0) {
                    if (usePredecessorPipeline) {
                        // A speculative fetch from the previous iteration may
                        // already own this slot.  Otherwise wait only for the
                        // current row, then issue its predecessor read.
                        if (!fetchLive[slot]) {
                            WaitForPosDoneStep(posDonePtr, publishTokenInAllRank[publishIdx], publishStepK[publishIdx],
                                               publishCnt);
                            if (guardLive[slot]) {
                                WaitFlag<AscendC::HardEvent::V_MTE2>(accumGuardEvent[slot]);
                                guardLive[slot] = false;
                            }
                            DataCopyPad(combineAccumUb[slot * vectorColumns],
                                        combineAccumGmGlobal[static_cast<int64_t>(publishFirstPos[publishIdx]) * n +
                                                             columnBegin],
                                        copyParamsBf16, padParamsBf16);
                            SetFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[1 + slot]);
                            fetchLive[slot] = true;
                        }
                    } else {
                        WaitForPosDoneStep(posDonePtr, publishTokenInAllRank[publishIdx], publishStepK[publishIdx],
                                           publishCnt);
                        SyncFunc<AscendC::HardEvent::V_MTE2>();
                        DataCopyPad(
                            combineAccumUb,
                            combineAccumGmGlobal[static_cast<int64_t>(publishFirstPos[publishIdx]) * n + columnBegin],
                            copyParamsBf16, padParamsBf16);
                    }
                }
                if (!payloadBatchJoined && R == copyStart) {
                    if (usePredecessorPipeline) {
                        WaitFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[0]);
                    } else {
                        SyncFunc<AscendC::HardEvent::MTE2_V>();
                    }
                    payloadBatchJoined = true;
                }
                LocalTensor<ElementC> rowUb = combineBatchUb[static_cast<int64_t>(R - copyStart) * vectorColumns];
                Muls(rowUb, rowUb, publishProb[publishIdx], vectorColumns);
                PipeBarrier<PIPE_V>();

                // Probe the next dependency once.  A ready predecessor is
                // fetched while the current row's vector work is in flight;
                // an unready future row never stalls the current Add.
                if (usePredecessorPipeline && R + 1 < publishEnd) {
                    int32_t nextIdx = publishIdx + 1;
                    int32_t nextSlot = (R + 1) & 1;
                    if (publishStepK[nextIdx] > 0 && !fetchLive[nextSlot] &&
                        IsPosDoneStepReady(posDonePtr, publishTokenInAllRank[nextIdx], publishStepK[nextIdx])) {
                        if (guardLive[nextSlot]) {
                            WaitFlag<AscendC::HardEvent::V_MTE2>(accumGuardEvent[nextSlot]);
                            guardLive[nextSlot] = false;
                        }
                        DataCopyPad(
                            combineAccumUb[nextSlot * vectorColumns],
                            combineAccumGmGlobal[static_cast<int64_t>(publishFirstPos[nextIdx]) * n + columnBegin],
                            copyParamsBf16, padParamsBf16);
                        SetFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[1 + nextSlot]);
                        fetchLive[nextSlot] = true;
                    }
                }

                if (publishStepK[publishIdx] > 0) {
                    if (usePredecessorPipeline) {
                        WaitFlag<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[1 + slot]);
                        fetchLive[slot] = false;
                        Add(rowUb, rowUb, combineAccumUb[slot * vectorColumns], vectorColumns);
                        PipeBarrier<PIPE_V>();
                        SetFlag<AscendC::HardEvent::V_MTE2>(accumGuardEvent[slot]);
                        guardLive[slot] = true;
                    } else {
                        SyncFunc<AscendC::HardEvent::MTE2_V>();
                        Add(rowUb, rowUb, combineAccumUb, vectorColumns);
                        PipeBarrier<PIPE_V>();
                    }
                }
                ++R;
            }

            SyncFunc<AscendC::HardEvent::V_MTE3>();
            if (usePredecessorPipeline) {
                // V_MTE3 proves all Adds are drained.  Consume every live
                // guard now so no stale event can satisfy a later slot reuse
                // or survive into the next expert/launch.
                for (int32_t slot = 0; slot < 2; ++slot) {
                    if (guardLive[slot]) {
                        WaitFlag<AscendC::HardEvent::V_MTE2>(accumGuardEvent[slot]);
                        guardLive[slot] = false;
                    }
                }
            }
            for (int32_t publishIdx = 0; publishIdx < publishCnt; ++publishIdx) {
                int32_t copyIdx = publishStart - copyStart + publishIdx;
                LocalTensor<ElementC> rowUb = combineBatchUb[static_cast<int64_t>(copyIdx) * vectorColumns];
                if ((tailMask & (1U << publishIdx)) != 0U) {
                    int32_t outputSourceRank = publishSourceRank[publishIdx];
                    if (outputSourceRank != remoteDSourceRank) {
                        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(
                            GetRankWinAddrWithOffset(outputSourceRank, remoteDOffset)));
                        remoteDSourceRank = outputSourceRank;
                    }
                    int64_t gmDstOffset =
                        (static_cast<int64_t>(rankId) * static_cast<int64_t>(m) + publishTokenRow[publishIdx]) * n +
                        columnBegin;
                    DataCopyPad(gmRemoteD[gmDstOffset], rowUb, copyParamsBf16);
                } else {
                    DataCopyPad(
                        combineAccumGmGlobal[static_cast<int64_t>(publishFirstPos[publishIdx]) * n + columnBegin],
                        rowUb, copyParamsBf16);
                }
            }
            if (publishEnd == copyEnd) {
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }

            // Only a successor observes the accumulator through posDone.
            // Fence and publish batches containing such an intermediate row;
            // an all-tail batch has no scalar consumer and is already drained
            // by MTE3_MTE2 before UB reuse plus PIPE_ALL at stage completion.
            if (posDonePublishMask != 0U && publishCompletedRows) {
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                for (int32_t publishIdx = 0; publishIdx < publishCnt; ++publishIdx) {
                    if ((posDonePublishMask & (1U << publishIdx)) != 0U) {
                        // Each (token, step) has exactly one contribution.
                        // The writer already observed posDone==step above, so
                        // MTE3_S above orders the payload before this
                        // bypass-cache scalar store, avoiding a read-modify-
                        // write atomic on the shared line.
                        WriteGmByPassDCache(posDonePtr + publishTokenInAllRank[publishIdx],
                                            publishStepK[publishIdx] + 1);
                    }
                }
            }
        }
    }
}

struct CombineRowRange {
    int32_t rowStart;
    int32_t rowEnd;
    int32_t combineBatch;
    bool useMGroupReady;
};

__aicore__ inline bool ClampCombineExpertRows(int32_t expertTokenStart, uint64_t maxOutputSize, int32_t &rowsThisExpert)
{
    if (expertTokenStart < 0 || rowsThisExpert <= 0 || static_cast<uint64_t>(expertTokenStart) >= maxOutputSize) {
        return false;
    }
    uint64_t remainingRows = maxOutputSize - static_cast<uint64_t>(expertTokenStart);
    if (static_cast<uint64_t>(rowsThisExpert) > remainingRows) {
        rowsThisExpert = static_cast<int32_t>(remainingRows);
    }
    return rowsThisExpert > 0;
}

// Shared by ordinary traversal and the one-shot startup lookahead. Keep the
// same physical owner, batch budget, material-tail gate and weighted partition.
template <bool EnableMGroupReady, bool ForceEveryMGroupReady>
__aicore__ inline CombineRowRange ResolveCombineRowRange(int32_t rowsThisExpert, uint32_t expertPerRank,
                                                         uint32_t totalAivCores, uint32_t aivCoreIdx,
                                                         int32_t maxCombineBatch, int32_t n, uint32_t blockNum,
                                                         uint32_t mGroupsPerWave, bool expertScheduleEnabled)
{
    constexpr int32_t MAX_COPY_BATCH = 8;
    constexpr int32_t DEPENDENCY_FIRST_COMBINE_BATCH = 4;
    constexpr int32_t MANY_EXPERT_DEPENDENCY_ROWS_MIN = 384;
    constexpr int32_t MANY_EXPERT_DEPENDENCY_ROWS_MAX = 512;
    int32_t maxRowsPerCore = Ops::Base::CeilDiv(rowsThisExpert, static_cast<int32_t>(totalAivCores));

    int32_t combineBatch = maxCombineBatch;
    // Profiling shows that the many-expert two-M-group regime is
    // dependency limited, while colder/hotter experts benefit from a
    // wider copy.  Compact expert sets keep the eight-row short-expert
    // path and use four rows only when each AIV owns a long range.
    bool dependencySensitiveManyExpert = expertPerRank > 8U && rowsThisExpert >= MANY_EXPERT_DEPENDENCY_ROWS_MIN &&
                                         rowsThisExpert <= MANY_EXPERT_DEPENDENCY_ROWS_MAX &&
                                         maxRowsPerCore > MAX_COPY_BATCH;
    bool compactExpertLongRows = expertPerRank <= 8U && maxRowsPerCore > MAX_COPY_BATCH;
    if (dependencySensitiveManyExpert || compactExpertLongRows) {
        combineBatch = combineBatch < DEPENDENCY_FIRST_COMBINE_BATCH ? combineBatch : DEPENDENCY_FIRST_COMBINE_BATCH;
    }

    uint32_t expertMGroups = GetGmm2MGroupCount(static_cast<uint32_t>(rowsThisExpert));
    bool useMGroupReady = false;
    uint32_t readyWaveCount = 0U;
    if constexpr (ForceEveryMGroupReady) {
        // C8's explicit global scheduler publishes every material
        // M-group after all of its N stripes are visible.  It has no
        // wave-budget fallback, so the consumer must use the matching
        // group counter for short, long and ragged experts alike.
        useMGroupReady = true;
        readyWaveCount = expertMGroups;
    } else if constexpr (EnableMGroupReady) {
        // The producer uses the same M-group budget gate.  Never make the
        // consumer wait for votes when an over-budget expert intentionally
        // publishes only its full-expert flag.
        bool allowLongStreaming = expertScheduleEnabled && expertMGroups <= mGroupsPerWave;
        if (expertMGroups <= mGroupsPerWave) {
            readyWaveCount = GetGmm2ReadyWaveCount(static_cast<uint32_t>(rowsThisExpert), static_cast<uint32_t>(n),
                                                   blockNum, allowLongStreaming);
            useMGroupReady = readyWaveCount != 0U;
        }
    }
    // Ready-group partitioning is enabled only when there are independent
    // groups to overlap, every group can own at least one core, and the
    // weighted group's slowest core issues no more DMAs than the
    // whole-expert partition.  Unlike the former all-cores-per-group
    // scheme, a ragged tail needs only its proportional worker subset;
    // requiring it to cover every AIV would disable the target 513-row
    // expert and recreate the cross-group boundary wait.
    uint32_t wholeRangeBatches = static_cast<uint32_t>(Ops::Base::CeilDiv(maxRowsPerCore, combineBatch));
    uint32_t logicalCore = aivCoreIdx;
    bool canAssignOneGroupPerCore = useMGroupReady && readyWaveCount >= 2U && expertMGroups > 1U &&
                                    GetMaterialMGroupCount(static_cast<uint32_t>(rowsThisExpert)) == expertMGroups &&
                                    expertMGroups <= totalAivCores;
    uint32_t extraGroupWorkers = canAssignOneGroupPerCore ? totalAivCores - expertMGroups : 0U;
    uint32_t groupedRangeBatches = 0U;
    uint32_t groupedMaxRowsPerCore = 0U;
    uint32_t nextGroupWorkerBegin = 0U;
    uint32_t ownedMGroup = expertMGroups;
    uint32_t ownedGroupRows = 0U;
    uint32_t ownedGroupWorkerBegin = 0U;
    uint32_t ownedGroupWorkerCount = 0U;
    if (canAssignOneGroupPerCore) {
        for (uint32_t mGroup = 0U; mGroup < expertMGroups; ++mGroup) {
            uint32_t groupRowStart = mGroup * EXPERT_SCHEDULE_M_TILE_ROWS;
            uint32_t groupRows = static_cast<uint32_t>(rowsThisExpert) - groupRowStart;
            groupRows = groupRows < EXPERT_SCHEDULE_M_TILE_ROWS ? groupRows : EXPERT_SCHEDULE_M_TILE_ROWS;
            // Give every ready group one core, then distribute the
            // remaining cores in proportion to its row count.  Prefix
            // boundaries are adjacent, cover every logical core exactly
            // once, and cannot leave a sparse-but-valid tail unowned.
            uint32_t groupWorkerBegin = nextGroupWorkerBegin;
            uint32_t groupWorkerEnd =
                (mGroup + 1U == expertMGroups)
                    ? totalAivCores
                    : mGroup + 1U +
                          static_cast<uint32_t>(static_cast<uint64_t>(groupRowStart + groupRows) * extraGroupWorkers /
                                                static_cast<uint32_t>(rowsThisExpert));
            nextGroupWorkerBegin = groupWorkerEnd;
            uint32_t groupWorkerCount = groupWorkerEnd - groupWorkerBegin;
            uint32_t maxGroupRowsPerCore = Ops::Base::CeilDiv(groupRows, groupWorkerCount);
            groupedMaxRowsPerCore =
                maxGroupRowsPerCore > groupedMaxRowsPerCore ? maxGroupRowsPerCore : groupedMaxRowsPerCore;
            uint32_t groupRangeBatches = Ops::Base::CeilDiv(maxGroupRowsPerCore, static_cast<uint32_t>(combineBatch));
            groupedRangeBatches = groupRangeBatches > groupedRangeBatches ? groupRangeBatches : groupedRangeBatches;
            if (logicalCore >= groupWorkerBegin && logicalCore < groupWorkerEnd) {
                ownedMGroup = mGroup;
                ownedGroupRows = groupRows;
                ownedGroupWorkerBegin = groupWorkerBegin;
                ownedGroupWorkerCount = groupWorkerCount;
            }
        }
    }
    uint32_t allowedGroupedRowsPerCore = static_cast<uint32_t>(maxRowsPerCore) + (readyWaveCount >= 3U ? 1U : 0U);
    bool partitionByReadyGroup = canAssignOneGroupPerCore && ownedMGroup < expertMGroups &&
                                 groupedMaxRowsPerCore <= allowedGroupedRowsPerCore &&
                                 groupedRangeBatches <= wholeRangeBatches;
    if (partitionByReadyGroup) {
        uint32_t groupRowStart = ownedMGroup * EXPERT_SCHEDULE_M_TILE_ROWS;
        uint32_t groupWorkerIdx = logicalCore - ownedGroupWorkerBegin;
        uint32_t rowsPerCore = ownedGroupRows / ownedGroupWorkerCount;
        uint32_t extraRows = ownedGroupRows % ownedGroupWorkerCount;
        uint32_t localGroupStart =
            groupWorkerIdx * rowsPerCore + (groupWorkerIdx < extraRows ? groupWorkerIdx : extraRows);
        uint32_t localRows = rowsPerCore + (groupWorkerIdx < extraRows ? 1U : 0U);
        int32_t rowStart = static_cast<int32_t>(groupRowStart + localGroupStart);
        return {rowStart, rowStart + static_cast<int32_t>(localRows), combineBatch, useMGroupReady};
    }
    uint32_t baseRows = static_cast<uint32_t>(rowsThisExpert) / totalAivCores;
    uint32_t extraRows = static_cast<uint32_t>(rowsThisExpert) % totalAivCores;
    uint32_t localStart = logicalCore * baseRows + (logicalCore < extraRows ? logicalCore : extraRows);
    uint32_t localRows = baseRows + (logicalCore < extraRows ? 1U : 0U);
    int32_t rowStart = static_cast<int32_t>(localStart);
    return {rowStart, rowStart + static_cast<int32_t>(localRows), combineBatch, useMGroupReady};
}

__aicore__ inline bool AreCombineRowGroupsReady(__gm__ int32_t *expertReadyBase, int32_t rowStart, int32_t rowEnd,
                                                int32_t readyTarget)
{
    if (rowStart >= rowEnd) {
        return false;
    }
    uint32_t firstMGroup = static_cast<uint32_t>(rowStart) / EXPERT_SCHEDULE_M_TILE_ROWS;
    uint32_t lastMGroup = static_cast<uint32_t>(rowEnd - 1) / EXPERT_SCHEDULE_M_TILE_ROWS;
    for (uint32_t group = firstMGroup; group <= lastMGroup; ++group) {
        if (ReadGmByPassDCache(expertReadyBase + static_cast<uint64_t>(group) * DISPATCH_TILE_READY_STRIDE +
                               GMM2_READY_OFFSET) != readyTarget) {
            return false;
        }
    }
    return true;
}

// Stop at the first unavailable dependency, not at an arbitrary batch size:
// an earlier ready prefix must still run, including when the next row is a
// later contribution of the same token that needs this prefix's publication.
__aicore__ inline int32_t GetReadyCombinePrefixEnd(__gm__ int32_t *stepIdxPtr, __gm__ int32_t *posDonePtr,
                                                   __gm__ int32_t *expertReadyBase, int32_t readyTarget,
                                                   int32_t expertTokenStart, int32_t rowStart, int32_t rowEnd,
                                                   uint32_t m, uint32_t topK, uint32_t worldSize)
{
    uint32_t tokenBits = GetTokenOwnerFieldBits(m);
    uint32_t slotBits = GetTokenOwnerFieldBits(topK);
    uint32_t tokenMask = tokenBits == 0U ? 0U : UINT32_MAX >> (32U - tokenBits);
    uint32_t slotMask = slotBits == 0U ? 0U : UINT32_MAX >> (32U - slotBits);
    uint32_t previousGroup = UINT32_MAX;
    int32_t row = rowStart;
    for (; row < rowEnd; ++row) {
        uint32_t group = static_cast<uint32_t>(row) / EXPERT_SCHEDULE_M_TILE_ROWS;
        if (group != previousGroup) {
            if (ReadGmByPassDCache(expertReadyBase + static_cast<uint64_t>(group) * DISPATCH_TILE_READY_STRIDE +
                                   GMM2_READY_OFFSET) != readyTarget) {
                break;
            }
            previousGroup = group;
        }
        uint32_t packed =
            static_cast<uint32_t>(ReadGmByPassDCache(stepIdxPtr + static_cast<int64_t>(expertTokenStart) + row));
        uint32_t step = packed & slotMask;
        uint32_t tokenRow = (packed >> (2U * slotBits)) & tokenMask;
        uint32_t sourceRank = packed >> (tokenBits + 2U * slotBits);
        if (sourceRank >= worldSize || tokenRow >= m || step >= topK ||
            (step > 0U && ReadGmByPassDCache(posDonePtr + static_cast<uint64_t>(sourceRank) * m + tokenRow) !=
                              static_cast<int32_t>(step))) {
            break;
        }
    }
    return row;
}

__aicore__ inline bool AreExpertMajorColumnTilesReady(__gm__ int32_t *groupReadyBase, uint32_t firstTile,
                                                      uint32_t endTile)
{
    for (uint32_t tile = firstTile; tile < endTile; ++tile) {
        if (ReadGmByPassDCache(groupReadyBase + GMM2_STRIPE_READY_OFFSET + tile) != 1) {
            return false;
        }
    }
    return true;
}

__aicore__ inline int32_t WaitForExpertMajorColumnPrefix(__gm__ int32_t *groupReadyBase, int32_t columnBegin,
                                                         int32_t columns)
{
    uint32_t nTiles = GetGmm2NTileCount(static_cast<uint32_t>(columns));
    uint32_t beginTile = static_cast<uint32_t>(columnBegin) / GMM2_READY_N_TILE_COLUMNS;
    uint32_t endTile = beginTile + 2U;
    // The shared gate requires full 512-column intervals. Split producer jobs
    // can complete out of order, so aggregate >=2 or just the last tile's
    // flag is insufficient: every tile in this interval must be visible.
    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (!AreExpertMajorColumnTilesReady(groupReadyBase, beginTile, endTile)) {
        if (ReadGmByPassDCache(groupReadyBase + GMM2_READY_OFFSET) == static_cast<int32_t>(nTiles)) {
            return columns;
        }
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
    }
    // Coalesce only the remaining suffix. Once a prefix has been written,
    // whole-group readiness never permits rerunning that prefix's Muls/Add.
    while (endTile < nTiles && AreExpertMajorColumnTilesReady(groupReadyBase, endTile, endTile + 2U)) {
        endTile += 2U;
    }
    return static_cast<int32_t>(endTile * GMM2_READY_N_TILE_COLUMNS);
}

__aicore__ inline int32_t GetExpertMajorPublicationEnd(int32_t rangeStart, int32_t row, int32_t combineBatch,
                                                       int32_t rangeEnd)
{
    // Anchor both copy batches and four-row publications to the ORIGINAL
    // range, including after a 256-row group boundary split. Restarting the
    // four-row count at the split can merge two old publications and make a
    // repeated-token successor wait for an unpublished predecessor in itself.
    constexpr int32_t PUBLISH_ROWS = 4;
    int32_t copyStart = rangeStart + ((row - rangeStart) / combineBatch) * combineBatch;
    int32_t end = copyStart + (((row - copyStart) / PUBLISH_ROWS) + 1) * PUBLISH_ROWS;
    end = end < copyStart + combineBatch ? end : copyStart + combineBatch;
    return end < rangeEnd ? end : rangeEnd;
}

template <typename ElementC, bool EnableMGroupReady = false, bool WaitForFinalChainLen = false,
          bool PublishAiv1UbOwnership = false, bool ForceEveryMGroupReady = false,
          bool DisablePredecessorPipeline = false, bool EnableStartupLookahead = false,
          bool EnableColumnStreaming = false>
__aicore__ inline void CombineTokensParallel(
    uint32_t rankId, GlobalTensor<ElementC> combineAccumGmGlobal, const Params &params,
    GlobalTensor<int32_t> &expandedRowIdx, GlobalTensor<bfloat16_t> &distProbs,
    GlobalTensor<int32_t> &tokenPerExpertWin, GlobalTensor<int32_t> &expertTokenCumsum,
    __gm__ int32_t *firstTokenPosPtr, __gm__ int32_t *stepIdxPtr, __gm__ int32_t *chainLenPtr,
    __gm__ int32_t *chainLenReadyFlag, __gm__ int32_t *posDonePtr, __gm__ int32_t *gmm2ReadyFlagBasePtr,
    __gm__ int32_t *expertOrderPtr, bool expertScheduleEnabled, uint32_t scheduleBegin, uint32_t scheduleEnd,
    uint32_t mGroupsPerWave, uint32_t combineCoreCount, uint32_t combineCoreIdx, AscendTimerDevice &timer)
{
    static_assert(!ForceEveryMGroupReady || EnableMGroupReady,
                  "forcing every M-group ready requires the ready protocol");
    int32_t n = static_cast<int32_t>(params.tilingData->k);  // GMM2输出维度 = k_
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t m = params.tilingData->m;
    uint32_t topK = params.tilingData->topK;
    uint32_t blockNum = GetBlockNum();
    int32_t expertAlign = Ops::Base::CeilAlign(static_cast<int64_t>(worldSize * expertPerRank), (int64_t)ALIGN_128);
    uint64_t maxOutputSize = params.tilingData->maxOutputSize;
    bool packedRouteMetadata = ShouldPackCombineRouteMetadata(worldSize, m, topK, maxOutputSize, blockNum);

    constexpr int32_t MAX_COPY_BATCH = 8;
    uint32_t combineRowBytes = static_cast<uint32_t>(n) * sizeof(ElementC);
    int32_t ubRows = static_cast<int32_t>(BUFFER_ALIGN / combineRowBytes);
    int32_t maxCombineBatch = ubRows > 1 ? ubRows - 1 : 1;
    maxCombineBatch = maxCombineBatch < MAX_COPY_BATCH ? maxCombineBatch : MAX_COPY_BATCH;
    constexpr uint32_t ROUTE_META_CACHE_WORDS = 16U;
    constexpr uint32_t ROUTE_META_CACHE_BYTES = ROUTE_META_CACHE_WORDS * sizeof(int32_t);
    // The predecessor pipeline amortizes its five persistent events and
    // per-batch flag choreography only when a core owns enough rows;
    // sparse-decode callers disable it at compile time so the whole
    // orchestration folds away and the fallback SyncFunc protocol (same
    // accumulation order, same GM addresses) is used.
    bool usePredecessorPipeline = false;
    if constexpr (!DisablePredecessorPipeline) {
        usePredecessorPipeline = CanUseCombinePredecessorPipeline(
            combineRowBytes, static_cast<uint32_t>(maxCombineBatch), ROUTE_META_CACHE_BYTES);
    }
    uint32_t accumulatorRows = usePredecessorPipeline ? 2U : 1U;
    uint32_t combinePayloadRows = accumulatorRows + static_cast<uint32_t>(maxCombineBatch);
    bool combinePayloadFits = combineRowBytes != 0U && combinePayloadRows <= BUFFER_ALIGN / combineRowBytes;
    uint32_t combinePayloadEnd = combinePayloadFits ? combinePayloadRows * combineRowBytes : BUFFER_ALIGN;
    uint32_t combineRouteMetaOffset =
        combinePayloadFits ? Ops::Base::CeilAlign(combinePayloadEnd, static_cast<uint32_t>(ALIGN_32)) : BUFFER_ALIGN;
    bool useRouteMetadataCache = combinePayloadFits && combineRouteMetaOffset <= BUFFER_ALIGN &&
                                 ROUTE_META_CACHE_BYTES <= BUFFER_ALIGN - combineRouteMetaOffset;
    LocalTensor<ElementC> combineAccumUb{AscendC::TPosition::VECIN, 0, accumulatorRows * static_cast<uint32_t>(n)};
    LocalTensor<ElementC> combineBatchUb{AscendC::TPosition::VECIN, accumulatorRows * combineRowBytes,
                                         static_cast<uint32_t>(maxCombineBatch * n)};
    LocalTensor<int32_t> combineRouteMetaUb{
        AscendC::TPosition::VECIN, useRouteMetadataCache ? combineRouteMetaOffset : 0U, ROUTE_META_CACHE_WORDS};

    DataCopyExtParams copyParamsBf16{1U, static_cast<uint32_t>(n * sizeof(ElementC)), 0U, 0U, 0U};
    DataCopyPadExtParams<ElementC> padParamsBf16{false, 0U, 0U, 0U};
    uint64_t remoteDOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;

    // 计算本rank的任务范围。expertPerRank是动态值，不能使用固定长度的局部数组。
    __gm__ int32_t *expertCumsumPtr = const_cast<__gm__ int32_t *>(expertTokenCumsum.GetPhyAddr());
    __gm__ int32_t *expertPlanPtr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrExpertOrderScratch);
    bool useExpertPlan = false;
    if constexpr (EnableMGroupReady) {
        useExpertPlan = expertPerRank >= EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS;
    }
    int32_t rankBaseRow =
        useExpertPlan ? 0 : ((rankId == 0) ? 0 : ReadGmByPassDCache(expertCumsumPtr + rankId * expertPerRank - 1));
    int32_t rankEndRow = useExpertPlan ? ReadGmByPassDCache(expertPlanPtr + expertPerRank)
                                       : ReadGmByPassDCache(expertCumsumPtr + (rankId + 1) * expertPerRank - 1);
    if (rankEndRow <= rankBaseRow) {
        return;
    }
    // Generic waits for AIV0's precomputed chain length.  Interleaved
    // core-wave waits for AIV1's single final-Dispatch publication before it
    // reuses immutable posLen.  Complete staging already crossed a mixed-core
    // phase boundary and compiles this wait out.
    if constexpr (!EnableMGroupReady || WaitForFinalChainLen) {
        uint32_t pollMisses = 0U;
        int64_t pollDelayCycles = 100;
        while (AscendC::ReadGmByPassDCache(chainLenReadyFlag) < static_cast<int32_t>(blockNum)) {
            GmSpinBackoffStep(pollMisses, pollDelayCycles);
        }
    }
    uint32_t totalAivCores = combineCoreCount;
    uint32_t aivCoreIdx = combineCoreIdx;

    // The optimized path owns one event for the payload batch and one for
    // each predecessor slot.  V_MTE2 guards prevent MTE2 from overwriting a
    // slot until its previous Add has stopped reading it.  Allocate only when
    // the generic UB gate enabled the two-row layout; the fallback keeps the
    // original SyncFunc protocol and consumes no persistent event IDs.
    AscendC::TEventID combineMte2vEvent[3] = {EVENT_ID0, EVENT_ID0, EVENT_ID0};
    AscendC::TEventID accumGuardEvent[2] = {EVENT_ID0, EVENT_ID0};
    if (usePredecessorPipeline) {
        combineMte2vEvent[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        combineMte2vEvent[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        combineMte2vEvent[2] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
        accumGuardEvent[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
        accumGuardEvent[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
    }

    // Every expert is quotient/remainder partitioned.  The optional ready-group
    // planner below can use the same exact partition inside a 256-row producer
    // group when its DMA cost model is favorable.
    // Identity traversal carries the previous physical end forward and keeps
    // the cumulative-C4 one-read rolling-cumsum fast path.  Scheduled
    // traversal resolves each physical expert independently because adjacent
    // schedule positions are not adjacent in the cumsum array.
    int32_t expertTokenStartGlobal = rankBaseRow;
    if (!expertScheduleEnabled && scheduleBegin > 0U) {
        int32_t previousGlobalExpert = static_cast<int32_t>(rankId * expertPerRank + scheduleBegin - 1U);
        expertTokenStartGlobal = ReadGmByPassDCache(expertCumsumPtr + previousGlobalExpert);
    }
    // Preserve physical ownership across the complete expert traversal.  A
    // scheduled per-expert rotation destroys token-chain locality and makes
    // posDone/accumulator ownership bounce between AIV cache domains.
    __gm__ int32_t *dispatchTileReadyBasePtr =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrDispatchTileReady);
    uint32_t nTiles = GetGmm2NTileCount(static_cast<uint32_t>(n));
    uint64_t maxTilesPerExpert = Ops::Base::CeilDiv(maxOutputSize, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
    uint64_t dispatchTileReadyExpertStride = maxTilesPerExpert * DISPATCH_TILE_READY_STRIDE;
    bool canUseColumnStreaming = false;
    if constexpr (EnableColumnStreaming && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
        if (useExpertPlan && packedRouteMetadata) {
            const uint32_t headerBase = 1U + 2U * expertPerRank;
            uint32_t totalMGroups = static_cast<uint32_t>(
                ReadGmByPassDCache(expertPlanPtr + headerBase + EXPERT_PLAN_HEADER_TOTAL_MGROUPS));
            canUseColumnStreaming = CanUseExpertMajorGmm2StripeReady(
                static_cast<uint32_t>(n), params.tilingData->n / SWIGLU_N_HALF, totalMGroups, blockNum);
        }
    }
    // Reserve only private UB slack. No progress state is initialized/read
    // unless the first nonempty range actually misses its producer readiness.
    uint32_t startupProgressOffset = combineRouteMetaOffset + (useRouteMetadataCache ? ROUTE_META_CACHE_BYTES : 0U);
    bool canUseStartupLookahead = false;
    if constexpr (EnableStartupLookahead && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
        canUseStartupLookahead = packedRouteMetadata && useExpertPlan &&
                                 (expertScheduleEnabled || scheduleBegin == 0U) && combinePayloadFits &&
                                 startupProgressOffset <= BUFFER_ALIGN &&
                                 expertPerRank <= (BUFFER_ALIGN - startupProgressOffset) / sizeof(int32_t);
    }
    uint32_t startupProgressWords =
        canUseStartupLookahead ? Ops::Base::CeilAlign(expertPerRank, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)))
                               : 0U;
    LocalTensor<int32_t> startupProgressUb{AscendC::TPosition::VECIN,
                                           canUseStartupLookahead ? startupProgressOffset : 0U, startupProgressWords};
    // Column streaming handles the current expert's producer stall directly.
    // Do not stack the one-shot future-expert scheduler on its partial rows.
    bool startupPending = canUseStartupLookahead && !canUseColumnStreaming;
    bool startupProgressInitialized = false;
    uint32_t startupResumeEnd = 0U;
    for (uint32_t scheduleIdx = scheduleBegin; scheduleIdx < scheduleEnd; ++scheduleIdx) {
        uint32_t e = expertScheduleEnabled
                         ? static_cast<uint32_t>(AscendC::ReadGmByPassDCache(expertOrderPtr + scheduleIdx))
                         : scheduleIdx;
        int32_t globalExpertIdx = static_cast<int32_t>(rankId * expertPerRank + e);
        int32_t currentExpertStartGlobal = expertTokenStartGlobal;
        int32_t expertTokenEndGlobal = 0;
        if (useExpertPlan) {
            if (expertScheduleEnabled) {
                currentExpertStartGlobal = e == 0U ? 0 : ReadGmByPassDCache(expertPlanPtr + e);
            }
            expertTokenEndGlobal = ReadGmByPassDCache(expertPlanPtr + 1U + e);
        } else if (expertScheduleEnabled) {
            currentExpertStartGlobal =
                globalExpertIdx == 0 ? 0 : ReadGmByPassDCache(expertCumsumPtr + globalExpertIdx - 1);
            expertTokenEndGlobal = expertTokenCumsum.GetValue(globalExpertIdx);
        } else {
            expertTokenEndGlobal = expertTokenCumsum.GetValue(globalExpertIdx);
        }
        int32_t expertTokenStart = currentExpertStartGlobal - rankBaseRow;
        int32_t rowsThisExpert = expertTokenEndGlobal - currentExpertStartGlobal;
        if (!expertScheduleEnabled) {
            expertTokenStartGlobal = expertTokenEndGlobal;
        }
        // Dispatch/GMM clamp the canonical local row space to maxOutputSize.
        // Apply the identical clamp before waiting on this expert's ready flag;
        // a fully truncated expert is never launched and therefore never
        // publishes readiness.
        if (!ClampCombineExpertRows(expertTokenStart, maxOutputSize, rowsThisExpert)) {
            continue;
        }

        // Every active expert performs exactly one reverse acquire and one
        // forward release on each physical AIV1/AIC pair. Acquire before any
        // LocalTensor payload access; all later continue paths release once.
        AcquireAiv1UbForCombine<PublishAiv1UbOwnership>();

        CombineRowRange range = ResolveCombineRowRange<EnableMGroupReady, ForceEveryMGroupReady>(
            rowsThisExpert, expertPerRank, totalAivCores, aivCoreIdx, maxCombineBatch, n, blockNum, mGroupsPerWave,
            expertScheduleEnabled);
        int32_t combineBatch = range.combineBatch;
        bool useMGroupReady = range.useMGroupReady;
        int32_t rowStart = range.rowStart;
        int32_t rowEnd = range.rowEnd;
        if constexpr (EnableStartupLookahead && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
            if (startupProgressInitialized) {
                if (scheduleIdx < startupResumeEnd) {
                    rowStart += startupProgressUb.GetValue(e);
                } else {
                    // Past the last prefetched expert, retain no steady-state
                    // progress lookup even when the startup probe missed.
                    startupProgressInitialized = false;
                }
            }
        }
        if (rowStart >= rowEnd) {
            ReleaseAiv1UbAfterCombine<PublishAiv1UbOwnership>();
            continue;
        }

        bool rangeAlreadyReady = false;
        if constexpr (EnableStartupLookahead && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
            if (startupPending) {
                startupPending = false;
                __gm__ int32_t *firstExpertReadyBase =
                    dispatchTileReadyBasePtr + static_cast<uint64_t>(e) * dispatchTileReadyExpertStride;
                rangeAlreadyReady =
                    AreCombineRowGroupsReady(firstExpertReadyBase, rowStart, rowEnd, static_cast<int32_t>(nTiles));
                if (!rangeAlreadyReady) {
                    Duplicate(startupProgressUb, 0, startupProgressWords);
                    SyncFunc<AscendC::HardEvent::V_S>();
                    startupProgressInitialized = true;
                    // One startup window, not a steady-state scheduler. A
                    // newly-ready first range ends the scan immediately.
                    constexpr uint32_t MAX_STARTUP_LOOKAHEAD_EXPERTS = 8U;
                    for (uint32_t lookahead = 1U;
                         lookahead <= MAX_STARTUP_LOOKAHEAD_EXPERTS && lookahead < scheduleEnd - scheduleIdx;
                         ++lookahead) {
                        rangeAlreadyReady = AreCombineRowGroupsReady(firstExpertReadyBase, rowStart, rowEnd,
                                                                     static_cast<int32_t>(nTiles));
                        if (rangeAlreadyReady) {
                            break;
                        }
                        uint32_t candidate =
                            expertScheduleEnabled
                                ? static_cast<uint32_t>(ReadGmByPassDCache(expertOrderPtr + scheduleIdx + lookahead))
                                : scheduleIdx + lookahead;
                        if (candidate >= expertPerRank) {
                            continue;
                        }
                        int32_t candidateStart = candidate == 0U ? 0 : ReadGmByPassDCache(expertPlanPtr + candidate);
                        int32_t candidateRows = ReadGmByPassDCache(expertPlanPtr + candidate + 1U) - candidateStart;
                        if (!ClampCombineExpertRows(candidateStart, maxOutputSize, candidateRows)) {
                            continue;
                        }
                        CombineRowRange candidateRange =
                            ResolveCombineRowRange<EnableMGroupReady, ForceEveryMGroupReady>(
                                candidateRows, expertPerRank, totalAivCores, aivCoreIdx, maxCombineBatch, n, blockNum,
                                mGroupsPerWave, expertScheduleEnabled);
                        int32_t candidateEnd = candidateRange.rowEnd;
                        if (candidateEnd - candidateRange.rowStart > candidateRange.combineBatch) {
                            candidateEnd = candidateRange.rowStart + candidateRange.combineBatch;
                        }
                        __gm__ int32_t *candidateReadyBase =
                            dispatchTileReadyBasePtr + static_cast<uint64_t>(candidate) * dispatchTileReadyExpertStride;
                        candidateEnd = GetReadyCombinePrefixEnd(
                            stepIdxPtr, posDonePtr, candidateReadyBase, static_cast<int32_t>(nTiles), candidateStart,
                            candidateRange.rowStart, candidateEnd, m, topK, worldSize);
                        if (candidateEnd <= candidateRange.rowStart) {
                            continue;
                        }
                        TIMER_BLOCK(timer.Tik(AscendTimer::AIV_COMBINE, candidate));
                        ProcessCombineRowRange<ElementC>(
                            rankId, combineAccumGmGlobal, expandedRowIdx, distProbs, tokenPerExpertWin,
                            firstTokenPosPtr, stepIdxPtr, chainLenPtr, posDonePtr,
                            static_cast<int32_t>(rankId * expertPerRank + candidate), candidateStart,
                            candidateRange.rowStart, candidateEnd, candidateRange.combineBatch, n, worldSize, m, topK,
                            packedRouteMetadata, expertAlign, remoteDOffset, useRouteMetadataCache,
                            usePredecessorPipeline, combineMte2vEvent, accumGuardEvent, combineAccumUb, combineBatchUb,
                            combineRouteMetaUb, copyParamsBf16, padParamsBf16);
                        TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, candidate));
                        startupProgressUb.SetValue(candidate, candidateEnd - candidateRange.rowStart);
                        startupResumeEnd = scheduleIdx + lookahead + 1U;
                    }
                }
            }
        }

        if constexpr (EnableColumnStreaming && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
            __gm__ int32_t *expertReadyBase =
                dispatchTileReadyBasePtr + static_cast<uint64_t>(e) * dispatchTileReadyExpertStride;
            if (canUseColumnStreaming &&
                !AreCombineRowGroupsReady(expertReadyBase, rowStart, rowEnd, static_cast<int32_t>(nTiles))) {
                bool streamTimerStarted = false;
                int32_t streamRow = rowStart;
                while (streamRow < rowEnd) {
                    uint32_t mGroup = static_cast<uint32_t>(streamRow) / EXPERT_SCHEDULE_M_TILE_ROWS;
                    __gm__ int32_t *groupReadyBase =
                        expertReadyBase + static_cast<uint64_t>(mGroup) * DISPATCH_TILE_READY_STRIDE;
                    int32_t groupEnd = static_cast<int32_t>((mGroup + 1U) * EXPERT_SCHEDULE_M_TILE_ROWS);
                    groupEnd = groupEnd < rowEnd ? groupEnd : rowEnd;
                    int32_t streamEnd = GetExpertMajorPublicationEnd(rowStart, streamRow, combineBatch, rowEnd);
                    streamEnd = streamEnd < groupEnd ? streamEnd : groupEnd;
                    if (ReadGmByPassDCache(groupReadyBase + GMM2_READY_OFFSET) == static_cast<int32_t>(nTiles)) {
                        // Only rows not yet touched by a slice enter the
                        // original full-width fast path. Never replay a
                        // partial row: step0 overwrites its own GMM2 input.
                        TIMER_BLOCK(if (!streamTimerStarted) {
                            timer.Tik(AscendTimer::AIV_COMBINE, e);
                            streamTimerStarted = true;
                        });
                        ProcessCombineRowRange<ElementC>(
                            rankId, combineAccumGmGlobal, expandedRowIdx, distProbs, tokenPerExpertWin,
                            firstTokenPosPtr, stepIdxPtr, chainLenPtr, posDonePtr, globalExpertIdx, expertTokenStart,
                            streamRow, streamEnd, combineBatch, n, worldSize, m, topK, packedRouteMetadata, expertAlign,
                            remoteDOffset, useRouteMetadataCache, usePredecessorPipeline, combineMte2vEvent,
                            accumGuardEvent, combineAccumUb, combineBatchUb, combineRouteMetaUb, copyParamsBf16,
                            padParamsBf16);
                        streamRow = streamEnd;
                        continue;
                    }
                    // Preserve the old four-row publication boundary. In
                    // particular, rows 5..8 may depend on rows 1..4; they must
                    // not enter slice0 until the first four whole rows publish.
                    int32_t streamRows = streamEnd - streamRow;
                    int32_t columnBegin = 0;
                    while (columnBegin < n) {
                        int32_t columnEnd = WaitForExpertMajorColumnPrefix(groupReadyBase, columnBegin, n);
                        TIMER_BLOCK(if (!streamTimerStarted) {
                            timer.Tik(AscendTimer::AIV_COMBINE, e);
                            streamTimerStarted = true;
                        });
                        ProcessCombineRowRange<ElementC, true>(
                            rankId, combineAccumGmGlobal, expandedRowIdx, distProbs, tokenPerExpertWin,
                            firstTokenPosPtr, stepIdxPtr, chainLenPtr, posDonePtr, globalExpertIdx, expertTokenStart,
                            streamRow, streamEnd, streamRows, n, worldSize, m, topK, packedRouteMetadata, expertAlign,
                            remoteDOffset, useRouteMetadataCache, usePredecessorPipeline, combineMte2vEvent,
                            accumGuardEvent, combineAccumUb, combineBatchUb, combineRouteMetaUb, copyParamsBf16,
                            padParamsBf16, columnBegin, columnEnd - columnBegin, columnEnd == n);
                        columnBegin = columnEnd;
                    }
                    streamRow = streamEnd;
                }
                TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, e));
                continue;
            } else if (canUseColumnStreaming) {
                // The nonblocking probe above already observed all required
                // group votes; do not repeat those GM reads in the old wait.
                rangeAlreadyReady = true;
            }
        }

        if (useMGroupReady && !rangeAlreadyReady) {
            uint32_t firstMGroup = static_cast<uint32_t>(rowStart) / EXPERT_SCHEDULE_M_TILE_ROWS;
            uint32_t lastMGroup = static_cast<uint32_t>(rowEnd - 1) / EXPERT_SCHEDULE_M_TILE_ROWS;
            for (uint32_t mGroup = firstMGroup; mGroup <= lastMGroup; ++mGroup) {
                int32_t readyTarget = static_cast<int32_t>(nTiles);
                __gm__ int32_t *readyAddr =
                    dispatchTileReadyBasePtr + static_cast<uint64_t>(e) * dispatchTileReadyExpertStride +
                    static_cast<uint64_t>(mGroup) * DISPATCH_TILE_READY_STRIDE + GMM2_READY_OFFSET;
                WaitForGmm2MGroupReady(readyAddr, readyTarget);
            }
        } else if (!useMGroupReady) {
            __gm__ int32_t *flagAddr = gmm2ReadyFlagBasePtr + static_cast<int32_t>(e) * INT_CACHELINE;
            uint32_t pollMisses = 0U;
            int64_t pollDelayCycles = 100;
            while (AscendC::ReadGmByPassDCache(flagAddr) != static_cast<int32_t>(blockNum)) {
                GmSpinBackoffStep(pollMisses, pollDelayCycles);
            }
        }

        TIMER_BLOCK(timer.Tik(AscendTimer::AIV_COMBINE, e));
        ProcessCombineRowRange<ElementC>(rankId, combineAccumGmGlobal, expandedRowIdx, distProbs, tokenPerExpertWin,
                                         firstTokenPosPtr, stepIdxPtr, chainLenPtr, posDonePtr, globalExpertIdx,
                                         expertTokenStart, rowStart, rowEnd, combineBatch, n, worldSize, m, topK,
                                         packedRouteMetadata, expertAlign, remoteDOffset, useRouteMetadataCache,
                                         usePredecessorPipeline, combineMte2vEvent, accumGuardEvent, combineAccumUb,
                                         combineBatchUb, combineRouteMetaUb, copyParamsBf16, padParamsBf16);
        TIMER_BLOCK(
            if constexpr (EnableStartupLookahead && ForceEveryMGroupReady && !PublishAiv1UbOwnership) {
                if (startupProgressInitialized && startupProgressUb.GetValue(e) > 0) {
                    // Keep the real suffix endpoint; a duration sum would
                    // move the apparent completion ahead of unfinished work.
                    timer.template Tok<Span>(AscendTimer::AIV_COMBINE, e);
                } else {
                    timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, e);
                }
            } else { timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, e); });
        ReleaseAiv1UbAfterCombine<PublishAiv1UbOwnership>();
    }
    if (usePredecessorPipeline) {
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(accumGuardEvent[1]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(accumGuardEvent[0]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[2]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[1]);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(combineMte2vEvent[0]);
    }
}

constexpr uint32_t OWNER_READY_CACHE_ENTRIES = 256U;
constexpr uint32_t K16_OWNER_ACCUM_SLOTS = 8U;
constexpr uint32_t OWNER_CONTRIBUTION_SLOTS = 2U;
static_assert(OWNER_CONTRIBUTION_SLOTS == 2U, "owner contribution ping-pong uses one slot bit");
static_assert(K16_OWNER_ACCUM_SLOTS <= 8U && (K16_OWNER_ACCUM_SLOTS & (K16_OWNER_ACCUM_SLOTS - 1U)) == 0U,
              "K16 resident slots must fit the hardware event IDs and be a power of two");
static_assert((OWNER_READY_CACHE_ENTRIES & (OWNER_READY_CACHE_ENTRIES - 1U)) == 0U,
              "owner ready cache size must be a power of two");

__aicore__ inline bool IsOwnerChainStripeReady(LocalTensor<int32_t> readyIndexCache,
                                               LocalTensor<int32_t> readyCacheKeyUb,
                                               LocalTensor<int32_t> readyCacheStateUb,
                                               __gm__ int32_t *dispatchTileReadyBase, uint32_t descriptorTokenBase,
                                               int32_t chainLen, uint32_t readyNTileCount, uint32_t stripe,
                                               int32_t probeEpoch)
{
    for (int32_t contribution = 0; contribution < chainLen; ++contribution) {
        int32_t signedReadyIndex = readyIndexCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(contribution));
        if (signedReadyIndex < 0) {
            return false;
        }
        uint32_t readyIndex = static_cast<uint32_t>(signedReadyIndex);
        uint32_t stripeReadyIndex = readyIndex * readyNTileCount + stripe;
        int32_t stripeReadyKey = static_cast<int32_t>(stripeReadyIndex);
        uint32_t readyCacheSlot = (stripeReadyIndex ^ (stripeReadyIndex >> 8U) ^ (stripeReadyIndex >> 16U)) &
                                  (OWNER_READY_CACHE_ENTRIES - 1U);
        int32_t readyCacheKey = readyCacheKeyUb.GetValue(readyCacheSlot);
        int32_t readyState = readyCacheStateUb.GetValue(readyCacheSlot);
        if (readyCacheKey == stripeReadyKey && readyState == 1) {
            continue;
        }
        if (readyCacheKey == stripeReadyKey && readyState == probeEpoch) {
            return false;
        }
        __gm__ int32_t *readyAddr = dispatchTileReadyBase +
                                    static_cast<uint64_t>(readyIndex) * DISPATCH_TILE_READY_STRIDE +
                                    GMM2_STRIPE_READY_OFFSET + stripe;
        if (ReadGmByPassDCache(readyAddr) < 1) {
            readyCacheKeyUb.SetValue(readyCacheSlot, stripeReadyKey);
            readyCacheStateUb.SetValue(readyCacheSlot, probeEpoch);
            return false;
        }
        readyCacheKeyUb.SetValue(readyCacheSlot, stripeReadyKey);
        readyCacheStateUb.SetValue(readyCacheSlot, 1);
    }
    return true;
}

// Resolve all currently consumable columns in one pass over the local chain,
// instead of rescanning every contribution separately for every N stripe.
// The existing cache state packs a partial mask and its sweep epoch, avoiding
// repeat GM polling by tokens sharing an unfinished group. Complete masks
// remain valid for the launch; partial masks are refreshed every sweep.
constexpr uint32_t OWNER_READY_MASK_BITS = 16U;
constexpr uint32_t OWNER_READY_MASK_VALUE = (1U << OWNER_READY_MASK_BITS) - 1U;
constexpr int32_t OWNER_READY_MASK_MAX_EPOCH = INT32_MAX >> OWNER_READY_MASK_BITS;
static_assert(GMM2_STRIPE_READY_CAPACITY < OWNER_READY_MASK_BITS,
              "the ready mask and positive sweep epoch must fit one int32_t");
__aicore__ inline uint32_t GetOwnerChainReadyMask(LocalTensor<int32_t> readyIndexCache,
                                                  LocalTensor<int32_t> readyCacheKeyUb,
                                                  LocalTensor<int32_t> readyCacheStateUb,
                                                  __gm__ int32_t *dispatchTileReadyBase, uint32_t descriptorTokenBase,
                                                  int32_t chainLen, uint32_t completeMask, uint32_t pendingMask,
                                                  int32_t probeEpoch)
{
    uint32_t availableMask = pendingMask;
    for (int32_t contribution = 0; contribution < chainLen; ++contribution) {
        int32_t signedReadyIndex = readyIndexCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(contribution));
        if (signedReadyIndex < 0) {
            return 0U;
        }
        uint32_t readyIndex = static_cast<uint32_t>(signedReadyIndex);
        uint32_t cacheSlot = (readyIndex ^ (readyIndex >> 8U) ^ (readyIndex >> 16U)) & (OWNER_READY_CACHE_ENTRIES - 1U);
        uint32_t readyState = static_cast<uint32_t>(readyCacheStateUb.GetValue(cacheSlot));
        uint32_t readyMask = readyState & OWNER_READY_MASK_VALUE;
        if (readyCacheKeyUb.GetValue(cacheSlot) != signedReadyIndex ||
            (readyMask != completeMask && (readyState >> OWNER_READY_MASK_BITS) != static_cast<uint32_t>(probeEpoch))) {
            readyMask = static_cast<uint32_t>(ReadGmByPassDCache(
                            dispatchTileReadyBase + static_cast<uint64_t>(readyIndex) * DISPATCH_TILE_READY_STRIDE +
                            GMM2_STRIPE_MASK_OFFSET)) &
                        completeMask;
            readyState = readyMask == completeMask
                             ? readyMask
                             : (static_cast<uint32_t>(probeEpoch) << OWNER_READY_MASK_BITS) | readyMask;
            readyCacheKeyUb.SetValue(cacheSlot, signedReadyIndex);
            readyCacheStateUb.SetValue(cacheSlot, static_cast<int32_t>(readyState));
        }
        availableMask &= readyMask;
        if (availableMask == 0U) {
            break;
        }
    }
    return availableMask;
}

template <bool K16Specialized = false>
__aicore__ inline bool CanUseOwnerReadyCombine(const Params &params, uint32_t combineCoreCount)
{
    uint32_t n = params.tilingData->k;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t m = params.tilingData->m;
    uint32_t runtimeTopK = params.tilingData->topK;
    if constexpr (K16Specialized) {
        if (runtimeTopK != 16U) {
            return false;
        }
    }
    uint32_t topK = K16Specialized ? 16U : runtimeTopK;
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    uint64_t maxOutputSize = params.tilingData->maxOutputSize;
    if (n == 0U || worldSize == 0U || m == 0U || topK == 0U || expertPerRank < EXPERT_SCHEDULE_MIN_ACTIVE_EXPERTS ||
        combineCoreCount == 0U || n > UINT32_MAX / static_cast<uint32_t>(sizeof(bfloat16_t)) ||
        m > UINT32_MAX / worldSize || !CanPackTokenOwnerDescriptor(expertPerRank, topK, maxOutputSize)) {
        return false;
    }
    uint32_t ownerPayloadBits = GetTokenOwnerDescriptorPayloadBits(expertPerRank, topK);
    if (ownerPayloadBits >= 31U || maxOutputSize - 1U > (static_cast<uint64_t>(INT32_MAX) >> ownerPayloadBits)) {
        return false;
    }
    uint32_t sourceTokens = worldSize * m;
    if (sourceTokens > UINT32_MAX / topK) {
        return false;
    }
    uint64_t maxTiles64 = Ops::Base::CeilDiv(maxOutputSize, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS));
    if (maxTiles64 == 0U || maxTiles64 > UINT32_MAX || maxTiles64 > UINT32_MAX / expertPerRank ||
        maxTiles64 > (UINT32_MAX / DISPATCH_TILE_READY_STRIDE) / expertPerRank) {
        return false;
    }
    uint32_t maxOwnedTokens = Ops::Base::CeilDiv(sourceTokens, combineCoreCount);
    if (m >= combineCoreCount) {
        maxOwnedTokens = Ops::Base::CeilDiv(m, combineCoreCount);
    }

    // The generic owner path needs one accumulator and two contribution rows.
    // K16 keeps a bounded set of partial token accumulators resident in UB.
    // A ready prefix can therefore overlap the remaining GMM2 work without a
    // GM spill/reload.  Both owner modes use two contribution rows for MTE2 /
    // Vector ping-pong.  The remaining UB holds the endpoint/order tables,
    // per-token chain
    // state, one aligned descriptor slice, its decoded ready-index slice, a
    // small tagged ready cache, and one contiguous probability slice.  All
    // checks are subtractive so device code never lowers a wide
    // multiply to an unsupported __multi3.
    uint32_t rowBytes = n * static_cast<uint32_t>(sizeof(bfloat16_t));
    if (rowBytes > UINT32_MAX - (ALIGN_32 - 1U)) {
        return false;
    }
    uint32_t alignedRowBytes = Ops::Base::CeilAlign(rowBytes, static_cast<uint32_t>(ALIGN_32));
    if constexpr (K16Specialized) {
        // The K16 fast path below uses basic DataCopy for every full row.
        // Unsupported hidden sizes retain the generic padded-copy path.
        if ((rowBytes & (ALIGN_32 - 1U)) != 0U) {
            return false;
        }
    }
    constexpr uint32_t rowBufferCount = (K16Specialized ? K16_OWNER_ACCUM_SLOTS : 1U) + OWNER_CONTRIBUTION_SLOTS;
    if (alignedRowBytes > BUFFER_ALIGN / rowBufferCount) {
        return false;
    }
    uint32_t remainingBytes = BUFFER_ALIGN - rowBufferCount * alignedRowBytes;
    uint32_t alignWords = ALIGN_32 / sizeof(int32_t);
    if (expertPerRank > UINT32_MAX - alignWords || maxOwnedTokens > UINT32_MAX - (alignWords - 1U)) {
        return false;
    }
    uint32_t planWords = Ops::Base::CeilAlign(expertPerRank + 1U, alignWords);
    uint32_t orderWords = Ops::Base::CeilAlign(expertPerRank, alignWords);
    uint32_t tokenWords = Ops::Base::CeilAlign(maxOwnedTokens, alignWords);
    uint32_t chainStorageWords = tokenWords;
    if (chainStorageWords > UINT32_MAX - alignWords) {
        return false;
    }
    chainStorageWords += alignWords;
    if (planWords > UINT32_MAX - orderWords || planWords + orderWords > UINT32_MAX - chainStorageWords ||
        planWords + orderWords + chainStorageWords > UINT32_MAX - tokenWords) {
        return false;
    }
    uint32_t fixedWords = planWords + orderWords + chainStorageWords + tokenWords;
    if (fixedWords > remainingBytes / sizeof(int32_t)) {
        return false;
    }
    remainingBytes -= fixedWords * sizeof(int32_t);
    constexpr uint32_t descriptorAlignWords = ALIGN_32 / sizeof(int32_t);
    if (maxOwnedTokens > UINT32_MAX / topK) {
        return false;
    }
    uint32_t routeWords = maxOwnedTokens * topK;
    if (routeWords > UINT32_MAX - (descriptorAlignWords - 1U)) {
        return false;
    }
    uint32_t descriptorWords = Ops::Base::CeilAlign(routeWords + descriptorAlignWords - 1U, descriptorAlignWords);
    uint32_t readyIndexWords = Ops::Base::CeilAlign(routeWords, descriptorAlignWords);
    uint32_t readyIndexUniverse = expertPerRank * static_cast<uint32_t>(maxTiles64);
    if (readyIndexUniverse > static_cast<uint32_t>(INT32_MAX)) {
        return false;
    }
    uint32_t stripeCount = GetGmm2NTileCount(n);
    if constexpr (K16Specialized) {
        if (CanUseGmm2StripeReady(n) &&
            (stripeCount == 0U || readyIndexUniverse > static_cast<uint32_t>(INT32_MAX) / stripeCount)) {
            return false;
        }
    }
    constexpr uint32_t readyCacheWords = 2U * OWNER_READY_CACHE_ENTRIES;
    uint32_t remainingWords = remainingBytes / sizeof(int32_t);
    if (descriptorWords > remainingWords) {
        return false;
    }
    remainingWords -= descriptorWords;
    if (readyIndexWords > remainingWords) {
        return false;
    }
    remainingWords -= readyIndexWords;
    if (readyCacheWords > remainingWords) {
        return false;
    }
    remainingWords -= readyCacheWords;
    // Cache the complete probability slice for every owner-chain width.
    // Reserve one cacheline of prefix slack so K8 slices that start on an odd
    // token can still bulk-copy the aligned body instead of issuing one GM
    // scalar read for every local contribution.
    if (maxOwnedTokens > UINT32_MAX / topK) {
        return false;
    }
    uint32_t probabilityElements = maxOwnedTokens * topK;
    constexpr uint32_t probabilityAlignElements = ALIGN_32 / static_cast<uint32_t>(sizeof(bfloat16_t));
    if (probabilityElements > UINT32_MAX - (probabilityAlignElements - 1U)) {
        return false;
    }
    uint32_t probabilityWords =
        Ops::Base::CeilDiv(probabilityElements + probabilityAlignElements - 1U, static_cast<uint32_t>(2U));
    probabilityWords = Ops::Base::CeilAlign(probabilityWords, descriptorAlignWords);
    if (probabilityWords > remainingWords) {
        return false;
    }
    return true;
}

// Long-chain core-wave Combine.  Dispatch compacts every token's local
// contributions into exact BF16 accumulation order.  One AIV owns the whole
// token chain and consumes the largest currently ready GMM2 prefix, removing
// cross-core posDone hand-offs and all but chunk-boundary GM accumulator
// round-trips.  GMM2 never waits on this consumer, so the monotonic ready
// counters cannot form a synchronization cycle.
template <typename ElementC, bool K16Specialized = false, bool UseRaggedStripeMask = false>
__aicore__ inline void CombineTokensByOwnerReadyChunks(
    uint32_t rankId, GlobalTensor<ElementC> combineAccumGmGlobal, const Params &params,
    GlobalTensor<bfloat16_t> &distProbs, __gm__ int32_t *tokenChainDescriptors, __gm__ int32_t *tokenChainLen,
    __gm__ int32_t *dispatchTileReadyBase, __gm__ int32_t *expertOrderPtr, bool expertScheduleEnabled,
    uint32_t combineCoreCount, uint32_t combineCoreIdx, AscendTimerDevice &timer)
{
    static_assert(sizeof(ElementC) == sizeof(bfloat16_t), "owner ready UB gate assumes one BF16-sized output element");
    static_assert(!UseRaggedStripeMask || K16Specialized, "ragged stripe masks require the K16 owner-chain protocol");
    if (combineCoreIdx >= combineCoreCount || !CanUseOwnerReadyCombine<K16Specialized>(params, combineCoreCount)) {
        return;
    }
    uint32_t n = params.tilingData->k;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t m = params.tilingData->m;
    uint32_t topK = K16Specialized ? 16U : params.tilingData->topK;
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    uint64_t maxOutputSize = params.tilingData->maxOutputSize;
    uint32_t maxTilesPerExpert =
        static_cast<uint32_t>(Ops::Base::CeilDiv(maxOutputSize, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS)));
    const uint32_t readyNTileCount = GetGmm2NTileCount(n);
    int32_t readyTarget = static_cast<int32_t>(readyNTileCount);
    bool useStripeReady = false;
    if constexpr (K16Specialized) {
        __gm__ int32_t *planAddr = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrExpertOrderScratch);
        uint32_t headerBase = 1U + 2U * expertPerRank;
        uint32_t totalMGroups =
            static_cast<uint32_t>(ReadGmByPassDCache(planAddr + headerBase + EXPERT_PLAN_HEADER_TOTAL_MGROUPS));
        uint32_t gmm2ReductionColumns = params.tilingData->n / SWIGLU_N_HALF;
        useStripeReady = CanUseFullAGmm2StripeReady(n, gmm2ReductionColumns, totalMGroups, GetBlockNum());
    }

    uint32_t rowBytes = n * static_cast<uint32_t>(sizeof(ElementC));
    uint32_t alignedRowBytes = Ops::Base::CeilAlign(rowBytes, static_cast<uint32_t>(ALIGN_32));
    uint32_t alignedRowElements = alignedRowBytes / static_cast<uint32_t>(sizeof(ElementC));
    LocalTensor<ElementC> accumStorage{AscendC::TPosition::VECIN, 0U,
                                       (K16Specialized ? K16_OWNER_ACCUM_SLOTS : 1U) * alignedRowElements};
    LocalTensor<ElementC> contributionStorage{AscendC::TPosition::VECIN,
                                              (K16Specialized ? K16_OWNER_ACCUM_SLOTS : 1U) * alignedRowBytes,
                                              OWNER_CONTRIBUTION_SLOTS * alignedRowElements};
    DataCopyExtParams rowCopyParams{1U, rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<ElementC> rowPadParams{false, 0U, 0U, 0U};

    uint32_t totalTokens = worldSize * m;
    bool usePerSourcePartition = m >= combineCoreCount;
    uint32_t maxOwnedTokens = usePerSourcePartition ? Ops::Base::CeilDiv(m, combineCoreCount)
                                                    : Ops::Base::CeilDiv(totalTokens, combineCoreCount);
    uint32_t alignWords = ALIGN_32 / sizeof(int32_t);
    constexpr uint32_t rowBufferCount = (K16Specialized ? K16_OWNER_ACCUM_SLOTS : 1U) + OWNER_CONTRIBUTION_SLOTS;
    uint32_t metadataByteOffset = rowBufferCount * alignedRowBytes;
    uint32_t planWords = Ops::Base::CeilAlign(expertPerRank + 1U, alignWords);
    uint32_t orderWords = Ops::Base::CeilAlign(expertPerRank, alignWords);
    uint32_t tokenWords = Ops::Base::CeilAlign(maxOwnedTokens, alignWords);
    uint32_t chainStorageWords = tokenWords + alignWords;
    uint32_t maxRouteWords = maxOwnedTokens * topK;
    uint32_t descriptorWords = Ops::Base::CeilAlign(maxRouteWords + alignWords - 1U, alignWords);
    uint32_t readyIndexWords = Ops::Base::CeilAlign(maxRouteWords, alignWords);
    LocalTensor<int32_t> metadataUb{AscendC::TPosition::VECIN, metadataByteOffset,
                                    static_cast<uint32_t>((BUFFER_ALIGN - metadataByteOffset) / sizeof(int32_t))};
    LocalTensor<int32_t> expertPlanUb = metadataUb;
    LocalTensor<int32_t> expertOrderUb = expertPlanUb[planWords];
    LocalTensor<int32_t> chainLenStorage = expertOrderUb[orderWords];
    LocalTensor<int32_t> chainProgressUb = chainLenStorage[chainStorageWords];
    LocalTensor<int32_t> descriptorStorage = chainProgressUb[tokenWords];
    LocalTensor<int32_t> readyIndexStorage = descriptorStorage[descriptorWords];
    LocalTensor<int32_t> readyCacheKeyUb = readyIndexStorage[readyIndexWords];
    LocalTensor<int32_t> readyCacheStateUb = readyCacheKeyUb[OWNER_READY_CACHE_ENTRIES];
    LocalTensor<bfloat16_t> probabilityStorage =
        readyCacheStateUb[OWNER_READY_CACHE_ENTRIES].template ReinterpretCast<bfloat16_t>();

    GlobalTensor<int32_t> expertPlanGm;
    expertPlanGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrExpertOrderScratch));
    DataCopyPad(expertPlanUb, expertPlanGm,
                DataCopyExtParams{1U, (expertPerRank + 1U) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
    if (expertScheduleEnabled) {
        GlobalTensor<int32_t> expertOrderGm;
        expertOrderGm.SetGlobalBuffer(expertOrderPtr);
        DataCopyPad(expertOrderUb, expertOrderGm,
                    DataCopyExtParams{1U, expertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
    }
    // GMM2 ready counters are monotonic within one launch.  The small tagged
    // cache stores state 1 for a permanent hit and values >=2 for one-sweep
    // misses.  A hash collision only causes another exact GM probe; it cannot
    // manufacture readiness.  Initialization overlaps the independent
    // plan/order MTE2 above.
    Duplicate(readyCacheKeyUb, -1, OWNER_READY_CACHE_ENTRIES);
    Duplicate(readyCacheStateUb, 0, OWNER_READY_CACHE_ENTRIES);
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    GlobalTensor<int32_t> descriptorGm;
    descriptorGm.SetGlobalBuffer(tokenChainDescriptors);
    uint32_t slotBits = K16Specialized ? 4U : GetTokenOwnerFieldBits(topK);
    uint32_t scheduleBits = GetTokenOwnerFieldBits(expertPerRank);
    uint32_t payloadBits = slotBits + scheduleBits;
    uint32_t slotMask = K16Specialized ? 15U : (slotBits == 0U ? 0U : (UINT32_MAX >> (32U - slotBits)));
    uint32_t scheduleMask = scheduleBits == 0U ? 0U : (UINT32_MAX >> (32U - scheduleBits));
    uint64_t remoteDOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
    GlobalTensor<ElementC> gmRemoteD;

    // Both owner tiling keys use two contribution slots.  MTE2_V proves a row is
    // ready for Vector, while V_MTE2 prevents the next DMA from overwriting a
    // slot whose previous Axpy is still in flight.  Events are allocated
    // once per launch and fully consumed before release below.
    AscendC::TEventID contributionReadyEvent[2] = {EVENT_ID0, EVENT_ID0};
    AscendC::TEventID contributionGuardEvent[2] = {EVENT_ID0, EVENT_ID0};
    contributionReadyEvent[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
    contributionReadyEvent[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
    contributionGuardEvent[0] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
    contributionGuardEvent[1] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE2>();
    if constexpr (K16Specialized) {
        // Seed one MTE3->MTE2 credit per resident slot.  The allocator below
        // rotates over free slots, and a completed token republishes its
        // credit only after the final remote store has drained.
        for (uint32_t accumSlot = 0U; accumSlot < K16_OWNER_ACCUM_SLOTS; ++accumSlot) {
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
        }
    }

    uint32_t flatRowsPerCore = totalTokens / combineCoreCount;
    uint32_t flatExtraRows = totalTokens % combineCoreCount;
    uint32_t flatTokenBegin =
        combineCoreIdx * flatRowsPerCore + (combineCoreIdx < flatExtraRows ? combineCoreIdx : flatExtraRows);
    uint32_t flatTokenCount = flatRowsPerCore + (combineCoreIdx < flatExtraRows ? 1U : 0U);
    uint32_t flatTokenEnd = flatTokenBegin + flatTokenCount;
    uint32_t rowsPerSourceCore = m / combineCoreCount;
    uint32_t extraSourceRows = m % combineCoreCount;
    uint32_t sourceRowBegin =
        combineCoreIdx * rowsPerSourceCore + (combineCoreIdx < extraSourceRows ? combineCoreIdx : extraSourceRows);
    uint32_t sourceRowCount = rowsPerSourceCore + (combineCoreIdx < extraSourceRows ? 1U : 0U);
    uint32_t sourceRowEnd = sourceRowBegin + sourceRowCount;

    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_COMBINE, 0U));
    uint32_t firstSource = combineCoreIdx % worldSize;
    int32_t probeEpoch = 2;
    uint32_t nextAccumSlotHint = 0U;
    if constexpr (K16Specialized) {
        nextAccumSlotHint = combineCoreIdx & (K16_OWNER_ACCUM_SLOTS - 1U);
    }
    for (uint32_t sourceIteration = 0U; sourceIteration < worldSize; ++sourceIteration) {
        uint32_t sourceRank = firstSource + sourceIteration;
        if (sourceRank >= worldSize) {
            sourceRank -= worldSize;
        }
        uint32_t tokenRowBegin = sourceRowBegin;
        uint32_t tokenRowEnd = sourceRowEnd;
        if (!usePerSourcePartition) {
            uint32_t sourceTokenBegin = sourceRank * m;
            uint32_t sourceTokenEnd = sourceTokenBegin + m;
            if (sourceTokenEnd <= flatTokenBegin || sourceTokenBegin >= flatTokenEnd) {
                continue;
            }
            uint32_t overlapBegin = flatTokenBegin > sourceTokenBegin ? flatTokenBegin : sourceTokenBegin;
            uint32_t overlapEnd = flatTokenEnd < sourceTokenEnd ? flatTokenEnd : sourceTokenEnd;
            tokenRowBegin = overlapBegin - sourceTokenBegin;
            tokenRowEnd = overlapEnd - sourceTokenBegin;
        }
        if (tokenRowBegin >= tokenRowEnd) {
            continue;
        }
        uint32_t ownedTokens = tokenRowEnd - tokenRowBegin;
        uint32_t tokenIdxBegin = sourceRank * m + tokenRowBegin;
        uint32_t descriptorBase = K16Specialized ? tokenIdxBegin << 4U : tokenIdxBegin * topK;
        uint32_t descriptorCount = K16Specialized ? ownedTokens << 4U : ownedTokens * topK;
        constexpr uint32_t descriptorAlignWords = ALIGN_32 / sizeof(int32_t);
        uint32_t chainPrefix = tokenIdxBegin & (descriptorAlignWords - 1U);
        LocalTensor<int32_t> chainLenCache = chainLenStorage[chainPrefix];
        uint32_t descriptorPrefix = descriptorBase & (descriptorAlignWords - 1U);
        LocalTensor<int32_t> descriptorCache = descriptorStorage[descriptorPrefix];
        LocalTensor<int32_t> readyIndexCache = readyIndexStorage;
        uint32_t descriptorHeadCount = descriptorPrefix == 0U ? 0U : descriptorAlignWords - descriptorPrefix;
        descriptorHeadCount = descriptorHeadCount < descriptorCount ? descriptorHeadCount : descriptorCount;
        uint32_t descriptorBodyCount = (descriptorCount - descriptorHeadCount) & ~(descriptorAlignWords - 1U);
        uint32_t descriptorTailBegin = descriptorHeadCount + descriptorBodyCount;
        if (descriptorBodyCount > 0U) {
            DataCopyPad(descriptorCache[descriptorHeadCount], descriptorGm[descriptorBase + descriptorHeadCount],
                        DataCopyExtParams{1U, descriptorBodyCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        }
        // Fetch the contiguous probability slice once per source/core.  K16
        // is naturally cacheline aligned; K8 keeps only an at-most-15-element
        // prefix/suffix scalar and transfers the aligned body through MTE2.
        constexpr uint32_t probabilityAlignElements = ALIGN_32 / static_cast<uint32_t>(sizeof(bfloat16_t));
        uint32_t probabilityBase = tokenIdxBegin * topK;
        uint32_t probabilityCount = ownedTokens * topK;
        uint32_t probabilityPrefix = probabilityBase & (probabilityAlignElements - 1U);
        LocalTensor<bfloat16_t> probabilityCache = probabilityStorage[probabilityPrefix];
        uint32_t probabilityHeadCount = probabilityPrefix == 0U ? 0U : probabilityAlignElements - probabilityPrefix;
        probabilityHeadCount = probabilityHeadCount < probabilityCount ? probabilityHeadCount : probabilityCount;
        uint32_t probabilityBodyCount = (probabilityCount - probabilityHeadCount) & ~(probabilityAlignElements - 1U);
        uint32_t probabilityTailBegin = probabilityHeadCount + probabilityBodyCount;
        if constexpr (K16Specialized) {
            DataCopy(probabilityCache, distProbs[probabilityBase], probabilityCount);
            probabilityBodyCount = probabilityCount;
            probabilityHeadCount = 0U;
            probabilityTailBegin = probabilityCount;
        } else if (probabilityBodyCount > 0U) {
            DataCopyPad(
                probabilityCache[probabilityHeadCount], distProbs[probabilityBase + probabilityHeadCount],
                DataCopyExtParams{1U, probabilityBodyCount * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<bfloat16_t>{false, 0U, 0U, 0U});
        }
        for (uint32_t probabilityIdx = 0U; probabilityIdx < probabilityHeadCount; ++probabilityIdx) {
            probabilityCache.SetValue(probabilityIdx, distProbs.GetValue(probabilityBase + probabilityIdx));
        }
        for (uint32_t probabilityIdx = probabilityTailBegin; probabilityIdx < probabilityCount; ++probabilityIdx) {
            probabilityCache.SetValue(probabilityIdx, distProbs.GetValue(probabilityBase + probabilityIdx));
        }

        // Chain lengths are another contiguous per-source slice.  Share the
        // same aligned-body scheme so K8 does not serialize one bypass-GM
        // read per token before it can begin consuming ready chains.
        uint32_t chainHeadCount = chainPrefix == 0U ? 0U : descriptorAlignWords - chainPrefix;
        chainHeadCount = chainHeadCount < ownedTokens ? chainHeadCount : ownedTokens;
        uint32_t chainBodyCount = (ownedTokens - chainHeadCount) & ~(descriptorAlignWords - 1U);
        uint32_t chainTailBegin = chainHeadCount + chainBodyCount;
        if (chainBodyCount > 0U) {
            GlobalTensor<int32_t> chainLenGm;
            chainLenGm.SetGlobalBuffer(tokenChainLen);
            DataCopyPad(chainLenCache[chainHeadCount], chainLenGm[tokenIdxBegin + chainHeadCount],
                        DataCopyExtParams{1U, chainBodyCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                        DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
        }
        for (uint32_t chainIdx = 0U; chainIdx < chainHeadCount; ++chainIdx) {
            chainLenCache.SetValue(chainIdx, ReadGmByPassDCache(tokenChainLen + tokenIdxBegin + chainIdx));
        }
        for (uint32_t chainIdx = chainTailBegin; chainIdx < ownedTokens; ++chainIdx) {
            chainLenCache.SetValue(chainIdx, ReadGmByPassDCache(tokenChainLen + tokenIdxBegin + chainIdx));
        }
        for (uint32_t descriptorIdx = 0U; descriptorIdx < descriptorHeadCount; ++descriptorIdx) {
            descriptorCache.SetValue(descriptorIdx,
                                     ReadGmByPassDCache(tokenChainDescriptors + descriptorBase + descriptorIdx));
        }
        for (uint32_t descriptorIdx = descriptorTailBegin; descriptorIdx < descriptorCount; ++descriptorIdx) {
            descriptorCache.SetValue(descriptorIdx,
                                     ReadGmByPassDCache(tokenChainDescriptors + descriptorBase + descriptorIdx));
        }
        if (descriptorBodyCount > 0U || probabilityBodyCount > 0U || chainBodyCount > 0U) {
            SyncFunc<AscendC::HardEvent::MTE2_S>();
        }

        // Clamp the owned chain-length window and clear stripe progress with
        // three vector ops instead of a per-token scalar read-clamp-store plus
        // a constant zero store (one LD/ST pair per token in the line-annotated
        // SASS).  chainLenCache itself starts chainPrefix words into the
        // storage, so drive the clamp from the 32B-aligned chainLenStorage
        // base over the live span; the at-most-7 dead prefix words sit inside
        // the storage's own alignWords padding and are never read.  The S_V
        // and V_S fences mirror the ready-cache reset pattern at the bottom of
        // this loop; MTE2_V retires the chain-body DMA before Vector reads it.
        if (chainBodyCount > 0U) {
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        SyncFunc<AscendC::HardEvent::S_V>();
        Maxs(chainLenStorage, chainLenStorage, int32_t(0), static_cast<int32_t>(chainPrefix + ownedTokens));
        // Auto synchronization is disabled; Mins reads the Maxs result in
        // the same UB window and must wait for its vector write to retire.
        PipeBarrier<PIPE_V>();
        Mins(chainLenStorage, chainLenStorage, static_cast<int32_t>(topK),
             static_cast<int32_t>(chainPrefix + ownedTokens));
        Duplicate(chainProgressUb, int32_t(0), static_cast<int32_t>(ownedTokens));
        SyncFunc<AscendC::HardEvent::V_S>();

        int32_t remainingTokens = 0;
        for (uint32_t localToken = 0U; localToken < ownedTokens; ++localToken) {
            int32_t chainLen = chainLenCache.GetValue(localToken);
            remainingTokens += chainLen > 0 ? 1 : 0;

            // Decode every dependency once after the descriptor DMA.  The hot
            // readiness loop below then performs one UB lookup plus, at most,
            // one GM probe for each exact (physical expert, M-group) in a
            // sweep.  Canonical descriptors remain untouched for the later
            // BF16 Muls/Add sequence.
            uint32_t descriptorTokenBase = K16Specialized ? localToken << 4U : localToken * topK;
            for (int32_t step = 0; step < chainLen; ++step) {
                int32_t cachedReadyIndex = -1;
                int32_t signedDescriptor = descriptorCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(step));
                if (signedDescriptor >= 0) {
                    uint32_t descriptor = static_cast<uint32_t>(signedDescriptor);
                    uint32_t scheduleIdx = (descriptor >> slotBits) & scheduleMask;
                    uint32_t canonicalRow = descriptor >> payloadBits;
                    if (scheduleIdx < expertPerRank && static_cast<uint64_t>(canonicalRow) < maxOutputSize) {
                        uint32_t expertIdx = expertScheduleEnabled
                                                 ? static_cast<uint32_t>(expertOrderUb.GetValue(scheduleIdx))
                                                 : scheduleIdx;
                        if (expertIdx < expertPerRank) {
                            int32_t localStart = expertIdx == 0U ? 0 : expertPlanUb.GetValue(expertIdx);
                            int32_t localEnd = expertPlanUb.GetValue(expertIdx + 1U);
                            if (localStart >= 0 && localEnd > localStart &&
                                canonicalRow >= static_cast<uint32_t>(localStart) &&
                                canonicalRow < static_cast<uint32_t>(localEnd)) {
                                uint32_t mGroup =
                                    (canonicalRow - static_cast<uint32_t>(localStart)) / EXPERT_SCHEDULE_M_TILE_ROWS;
                                cachedReadyIndex = static_cast<int32_t>(expertIdx * maxTilesPerExpert + mGroup);
                            }
                        }
                    }
                }
                readyIndexCache.SetValue(descriptorTokenBase + static_cast<uint32_t>(step), cachedReadyIndex);
            }
        }

        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementC *>(GetRankWinAddrWithOffset(sourceRank, remoteDOffset)));
        uint32_t freeAccumMask = K16Specialized ? (1U << K16_OWNER_ACCUM_SLOTS) - 1U : 1U;
        if constexpr (K16Specialized) {
            if (useStripeReady) {
                // One AIV retains complete ownership of each token.  Columns
                // are independent, so consume one ready 256-column stripe of
                // the token's entire local expert chain at a time and publish
                // only that disjoint remote-D slice.  No partial row advances
                // shared chain state and no two AIVs write the same slice.
                uint32_t completeStripeMask = (1U << readyNTileCount) - 1U;
                // Full-A producers sweep N from low to high.  Probe tile zero
                // first so an ahead-of-consumer producer naturally coalesces
                // the whole ready prefix into one contiguous transfer.
                uint32_t stripeSweepStart = 0U;
                uint32_t tokenSweepStart = combineCoreIdx % ownedTokens;
                uint32_t idlePollMisses = 0U;
                int64_t idlePollDelayCycles = 256;
                while (remainingTokens > 0) {
                    bool advanced = false;
                    for (uint32_t scan = 0U; scan < ownedTokens; ++scan) {
                        uint32_t localToken = tokenSweepStart + scan;
                        if (localToken >= ownedTokens) {
                            localToken -= ownedTokens;
                        }
                        int32_t chainLen = chainLenCache.GetValue(localToken);
                        if (chainLen <= 0) {
                            continue;
                        }
                        uint32_t completedStripes = static_cast<uint32_t>(chainProgressUb.GetValue(localToken));
                        if (completedStripes == completeStripeMask) {
                            continue;
                        }
                        uint32_t descriptorTokenBase = localToken << 4U;
                        uint32_t availableStripeMask = 0U;
                        if constexpr (UseRaggedStripeMask) {
                            availableStripeMask = GetOwnerChainReadyMask(
                                readyIndexCache, readyCacheKeyUb, readyCacheStateUb, dispatchTileReadyBase,
                                descriptorTokenBase, chainLen, completeStripeMask,
                                completeStripeMask & ~completedStripes, probeEpoch);
                        }
                        uint32_t readyStripe = readyNTileCount;
                        for (uint32_t stripeScan = 0U; stripeScan < readyNTileCount; ++stripeScan) {
                            uint32_t stripe = stripeSweepStart + stripeScan;
                            if (stripe >= readyNTileCount) {
                                stripe -= readyNTileCount;
                            }
                            if ((completedStripes & (1U << stripe)) != 0U) {
                                continue;
                            }
                            if constexpr (UseRaggedStripeMask) {
                                if ((availableStripeMask & (1U << stripe)) != 0U) {
                                    readyStripe = stripe;
                                    break;
                                }
                            } else if (IsOwnerChainStripeReady(readyIndexCache, readyCacheKeyUb, readyCacheStateUb,
                                                               dispatchTileReadyBase, descriptorTokenBase, chainLen,
                                                               readyNTileCount, stripe, probeEpoch)) {
                                readyStripe = stripe;
                                break;
                            }
                        }
                        if (readyStripe == readyNTileCount) {
                            continue;
                        }

                        // Coalesce an already-ready adjacent run.  Early in
                        // GMM2 this is commonly one 256-column tile; once the
                        // producer gets ahead, one wider DMA/Vector operation
                        // amortizes the per-stripe setup and converges to the
                        // established full-row path when all stripes are ready.
                        uint32_t readyStripeEnd = readyStripe + 1U;
                        if constexpr (UseRaggedStripeMask) {
                            while (readyStripeEnd < readyNTileCount &&
                                   (availableStripeMask & (1U << readyStripeEnd)) != 0U) {
                                ++readyStripeEnd;
                            }
                        } else {
                            while (readyStripeEnd < readyNTileCount &&
                                   (completedStripes & (1U << readyStripeEnd)) == 0U &&
                                   IsOwnerChainStripeReady(readyIndexCache, readyCacheKeyUb, readyCacheStateUb,
                                                           dispatchTileReadyBase, descriptorTokenBase, chainLen,
                                                           readyNTileCount, readyStripeEnd, probeEpoch)) {
                                ++readyStripeEnd;
                            }
                        }
                        uint32_t columnBegin = readyStripe * GMM2_READY_N_TILE_COLUMNS;
                        uint32_t columnEnd = readyStripeEnd * GMM2_READY_N_TILE_COLUMNS;
                        columnEnd = columnEnd < n ? columnEnd : n;
                        uint32_t stripeColumns = columnEnd - columnBegin;
                        uint32_t accumSlot = nextAccumSlotHint;
                        ++nextAccumSlotHint;
                        if (nextAccumSlotHint == K16_OWNER_ACCUM_SLOTS) {
                            nextAccumSlotHint = 0U;
                        }
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
                        LocalTensor<ElementC> accumUb = accumStorage[accumSlot * alignedRowElements];
                        uint32_t firstDescriptor = static_cast<uint32_t>(descriptorCache.GetValue(descriptorTokenBase));
                        uint32_t firstCanonical = firstDescriptor >> payloadBits;
                        DataCopy(accumUb, combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n + columnBegin],
                                 stripeColumns);
                        SyncFunc<AscendC::HardEvent::MTE2_V>();
                        uint32_t probabilityLocalBase = localToken << 4U;
                        bfloat16_t firstProb =
                            probabilityCache.GetValue(probabilityLocalBase + (firstDescriptor & slotMask));
                        Muls(accumUb, accumUb, firstProb, stripeColumns);
                        PipeBarrier<PIPE_V>();

                        bool guardLive[2] = {false, false};
                        int32_t contribution = 1;
                        if (contribution < chainLen) {
                            uint32_t descriptor =
                                static_cast<uint32_t>(descriptorCache.GetValue(descriptorTokenBase + 1U));
                            uint32_t canonicalRow = descriptor >> payloadBits;
                            int32_t contributionSlot = contribution & 1;
                            DataCopy(contributionStorage[static_cast<uint32_t>(contributionSlot) * alignedRowElements],
                                     combineAccumGmGlobal[static_cast<uint64_t>(canonicalRow) * n + columnBegin],
                                     stripeColumns);
                            SetFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[contributionSlot]);
                            while (contribution < chainLen) {
                                contributionSlot = contribution & 1;
                                int32_t nextContribution = contribution + 1;
                                if (nextContribution < chainLen) {
                                    int32_t nextSlot = nextContribution & 1;
                                    if (guardLive[nextSlot]) {
                                        WaitFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[nextSlot]);
                                        guardLive[nextSlot] = false;
                                    }
                                    uint32_t nextDescriptor = static_cast<uint32_t>(descriptorCache.GetValue(
                                        descriptorTokenBase + static_cast<uint32_t>(nextContribution)));
                                    uint32_t nextCanonicalRow = nextDescriptor >> payloadBits;
                                    DataCopy(
                                        contributionStorage[static_cast<uint32_t>(nextSlot) * alignedRowElements],
                                        combineAccumGmGlobal[static_cast<uint64_t>(nextCanonicalRow) * n + columnBegin],
                                        stripeColumns);
                                    SetFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[nextSlot]);
                                }
                                WaitFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[contributionSlot]);
                                uint32_t currentDescriptor = static_cast<uint32_t>(descriptorCache.GetValue(
                                    descriptorTokenBase + static_cast<uint32_t>(contribution)));
                                bfloat16_t prob =
                                    probabilityCache.GetValue(probabilityLocalBase + (currentDescriptor & slotMask));
                                LocalTensor<ElementC> contributionUb =
                                    contributionStorage[static_cast<uint32_t>(contributionSlot) * alignedRowElements];
                                Axpy(accumUb, contributionUb, prob, stripeColumns);
                                PipeBarrier<PIPE_V>();
                                SetFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[contributionSlot]);
                                guardLive[contributionSlot] = true;
                                contribution = nextContribution;
                            }
                            for (int32_t guardSlot = 0; guardSlot < 2; ++guardSlot) {
                                if (guardLive[guardSlot]) {
                                    WaitFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[guardSlot]);
                                }
                            }
                        }

                        SyncFunc<AscendC::HardEvent::V_MTE3>();
                        uint32_t tokenRow = tokenRowBegin + localToken;
                        uint64_t outputOffset = (static_cast<uint64_t>(rankId) * m + tokenRow) * n + columnBegin;
                        DataCopy(gmRemoteD[outputOffset], accumUb, stripeColumns);
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
                        uint32_t stripeRunCount = readyStripeEnd - readyStripe;
                        uint32_t stripeRunMask = ((1U << stripeRunCount) - 1U) << readyStripe;
                        completedStripes |= stripeRunMask;
                        chainProgressUb.SetValue(localToken, static_cast<int32_t>(completedStripes));
                        if (completedStripes == completeStripeMask) {
                            --remainingTokens;
                        }
                        advanced = true;
                    }
                    ++stripeSweepStart;
                    if (stripeSweepStart == readyNTileCount) {
                        stripeSweepStart = 0U;
                    }
                    ++tokenSweepStart;
                    if (tokenSweepStart == ownedTokens) {
                        tokenSweepStart = 0U;
                    }
                    constexpr int32_t maxStripeProbeEpoch =
                        UseRaggedStripeMask ? OWNER_READY_MASK_MAX_EPOCH : INT32_MAX;
                    if (probeEpoch == maxStripeProbeEpoch) {
                        Duplicate(readyCacheKeyUb, -1, OWNER_READY_CACHE_ENTRIES);
                        Duplicate(readyCacheStateUb, 0, OWNER_READY_CACHE_ENTRIES);
                        SyncFunc<AscendC::HardEvent::V_S>();
                        probeEpoch = 2;
                    } else {
                        ++probeEpoch;
                    }
                    if (advanced) {
                        // Work reappeared; resume the fast probing cadence.
                        idlePollMisses = 0U;
                        idlePollDelayCycles = 256;
                    } else {
                        GmSpinBackoffStep(idlePollMisses, idlePollDelayCycles);
                    }
                }
                continue;
            }
        }
        uint32_t sweepStart = combineCoreIdx % ownedTokens;
        uint32_t idlePollMisses = 0U;
        int64_t idlePollDelayCycles = 256;
        while (remainingTokens > 0) {
            bool advanced = false;
            for (uint32_t scan = 0U; scan < ownedTokens; ++scan) {
                uint32_t localToken = sweepStart + scan;
                if (localToken >= ownedTokens) {
                    localToken -= ownedTokens;
                }
                int32_t chainLen = chainLenCache.GetValue(localToken);
                int32_t progressState = chainProgressUb.GetValue(localToken);
                int32_t readyBegin = progressState;
                int32_t residentSlot = -1;
                if constexpr (K16Specialized) {
                    readyBegin = progressState & 0xff;
                    residentSlot = (progressState >> 8) - 1;
                }
                if (readyBegin >= chainLen) {
                    continue;
                }
                uint32_t descriptorTokenBase = K16Specialized ? localToken << 4U : localToken * topK;
                int32_t readyEnd = readyBegin;
                while (readyEnd < chainLen) {
                    int32_t signedReadyIndex =
                        readyIndexCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(readyEnd));
                    if (signedReadyIndex < 0) {
                        break;
                    }
                    uint32_t readyIndex = static_cast<uint32_t>(signedReadyIndex);
                    uint32_t readyCacheSlot =
                        (readyIndex ^ (readyIndex >> 8U) ^ (readyIndex >> 16U)) & (OWNER_READY_CACHE_ENTRIES - 1U);
                    int32_t readyCacheKey = readyCacheKeyUb.GetValue(readyCacheSlot);
                    int32_t readyState = readyCacheStateUb.GetValue(readyCacheSlot);
                    if (readyCacheKey != signedReadyIndex || readyState != 1) {
                        // A miss is shared by every token whose current chain
                        // head depends on this exact counter, but expires at
                        // the end of the sweep so a later producer publication
                        // is always observed.  A hit is permanent because the
                        // producer counter never decreases within a launch.
                        if (readyCacheKey == signedReadyIndex && readyState == probeEpoch) {
                            break;
                        }
                        __gm__ int32_t *readyAddr =
                            dispatchTileReadyBase + readyIndex * DISPATCH_TILE_READY_STRIDE + GMM2_READY_OFFSET;
                        if (ReadGmByPassDCache(readyAddr) != readyTarget) {
                            readyCacheKeyUb.SetValue(readyCacheSlot, signedReadyIndex);
                            readyCacheStateUb.SetValue(readyCacheSlot, probeEpoch);
                            break;
                        }
                        readyCacheKeyUb.SetValue(readyCacheSlot, signedReadyIndex);
                        readyCacheStateUb.SetValue(readyCacheSlot, 1);
                    }
                    ++readyEnd;
                }
                if (readyEnd == readyBegin) {
                    continue;
                }
                bool hasResidentAccum = residentSlot >= 0;
                // The proven full chunk remains eight for K16 and four for
                // generic.  K16 may start at half a full chunk because the
                // prefix stays resident in UB; generic still needs the full
                // chunk to amortize its partial GM store/reload.  Resident
                // suffixes retain the same four-row granularity so a gradual
                // producer does not turn one 4+4 chain into 4+1+1+1+1.
                constexpr int32_t MIN_READY_CHUNK = K16Specialized ? 8 : 4;
                if (readyEnd < chainLen && readyEnd - readyBegin < MIN_READY_CHUNK) {
                    if constexpr (K16Specialized) {
                        constexpr int32_t MIN_RESIDENT_PREFIX = MIN_READY_CHUNK / 2;
                        constexpr int32_t MIN_RESIDENT_SUFFIX = MIN_READY_CHUNK / 2;
                        int32_t minResidentChunk = hasResidentAccum ? MIN_RESIDENT_SUFFIX : MIN_RESIDENT_PREFIX;
                        if (readyEnd - readyBegin < minResidentChunk) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                uint32_t accumSlot = 0U;
                if constexpr (K16Specialized) {
                    if (hasResidentAccum) {
                        accumSlot = static_cast<uint32_t>(residentSlot);
                    } else {
                        if (freeAccumMask == 0U) {
                            continue;
                        }
                        // Keep one slot available for a fully-ready token.
                        // Otherwise eight cold four-row prefixes can occupy
                        // every accumulator and block useful drain work.
                        if (readyEnd < chainLen && (freeAccumMask & (freeAccumMask - 1U)) == 0U) {
                            continue;
                        }
                        // Eight slots keep the search bounded.  Dynamic
                        // allocation avoids modulo-slot head-of-line blocking
                        // when a cold suffix delays only one resident token.
                        // Start after the previous allocation so fully-ready
                        // tokens spread final MTE3 stores over all credits
                        // instead of serially waiting on slot zero.
                        for (uint32_t slotOffset = 0U; slotOffset < K16_OWNER_ACCUM_SLOTS; ++slotOffset) {
                            uint32_t candidate = nextAccumSlotHint + slotOffset;
                            if (candidate >= K16_OWNER_ACCUM_SLOTS) {
                                candidate -= K16_OWNER_ACCUM_SLOTS;
                            }
                            if ((freeAccumMask & (1U << candidate)) != 0U) {
                                accumSlot = candidate;
                                break;
                            }
                        }
                        freeAccumMask &= ~(1U << accumSlot);
                        nextAccumSlotHint = accumSlot + 1U;
                        if (nextAccumSlotHint == K16_OWNER_ACCUM_SLOTS) {
                            nextAccumSlotHint = 0U;
                        }
                    }
                }
                uint32_t firstDescriptor = static_cast<uint32_t>(descriptorCache.GetValue(descriptorTokenBase));
                uint32_t firstCanonical = firstDescriptor >> payloadBits;
                LocalTensor<ElementC> accumUb = accumStorage[accumSlot * alignedRowElements];
                if constexpr (K16Specialized) {
                    if (!hasResidentAccum) {
                        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
                    }
                }
                int32_t contribution = readyBegin;
                if (readyBegin == 0 && !hasResidentAccum) {
                    uint32_t topKSlot = firstDescriptor & slotMask;
                    if constexpr (K16Specialized) {
                        DataCopy(accumUb, combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n], n);
                    } else {
                        DataCopyPad(accumUb, combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n],
                                    rowCopyParams, rowPadParams);
                    }
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    uint32_t probabilityLocalBase = K16Specialized ? localToken << 4U : localToken * topK;
                    bfloat16_t prob = probabilityCache.GetValue(probabilityLocalBase + topKSlot);
                    Muls(accumUb, accumUb, prob, n);
                    PipeBarrier<PIPE_V>();
                    contribution = 1;
                } else if (!hasResidentAccum) {
                    if constexpr (K16Specialized) {
                        DataCopy(accumUb, combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n], n);
                    } else {
                        DataCopyPad(accumUb, combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n],
                                    rowCopyParams, rowPadParams);
                    }
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                }
                bool guardLive[2] = {false, false};
                if (contribution < readyEnd) {
                    int32_t currentContribution = contribution;
                    uint32_t descriptor = static_cast<uint32_t>(
                        descriptorCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(currentContribution)));
                    uint32_t canonicalRow = descriptor >> payloadBits;
                    int32_t slot = currentContribution & 1;
                    if constexpr (K16Specialized) {
                        DataCopy(contributionStorage[static_cast<uint32_t>(slot) * alignedRowElements],
                                 combineAccumGmGlobal[static_cast<uint64_t>(canonicalRow) * n], n);
                    } else {
                        DataCopyPad(contributionStorage[static_cast<uint32_t>(slot) * alignedRowElements],
                                    combineAccumGmGlobal[static_cast<uint64_t>(canonicalRow) * n], rowCopyParams,
                                    rowPadParams);
                    }
                    SetFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[slot]);

                    while (currentContribution < readyEnd) {
                        slot = currentContribution & 1;
                        int32_t nextContribution = currentContribution + 1;
                        if (nextContribution < readyEnd) {
                            int32_t nextSlot = nextContribution & 1;
                            if (guardLive[nextSlot]) {
                                WaitFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[nextSlot]);
                                guardLive[nextSlot] = false;
                            }
                            uint32_t nextDescriptor = static_cast<uint32_t>(descriptorCache.GetValue(
                                descriptorTokenBase + static_cast<uint32_t>(nextContribution)));
                            uint32_t nextCanonicalRow = nextDescriptor >> payloadBits;
                            if constexpr (K16Specialized) {
                                DataCopy(contributionStorage[static_cast<uint32_t>(nextSlot) * alignedRowElements],
                                         combineAccumGmGlobal[static_cast<uint64_t>(nextCanonicalRow) * n], n);
                            } else {
                                DataCopyPad(contributionStorage[static_cast<uint32_t>(nextSlot) * alignedRowElements],
                                            combineAccumGmGlobal[static_cast<uint64_t>(nextCanonicalRow) * n],
                                            rowCopyParams, rowPadParams);
                            }
                            SetFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[nextSlot]);
                        }

                        // Queue the next independent row before joining the
                        // current MTE2.  This keeps the load engine occupied
                        // across the Vector hand-off for both K8 and K16.
                        WaitFlag<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[slot]);

                        descriptor = static_cast<uint32_t>(
                            descriptorCache.GetValue(descriptorTokenBase + static_cast<uint32_t>(currentContribution)));
                        uint32_t topKSlot = descriptor & slotMask;
                        uint32_t probabilityLocalBase = K16Specialized ? localToken << 4U : localToken * topK;
                        bfloat16_t prob = probabilityCache.GetValue(probabilityLocalBase + topKSlot);
                        LocalTensor<ElementC> contributionUb =
                            contributionStorage[static_cast<uint32_t>(slot) * alignedRowElements];
                        // Atlas 350/A5 supports BF16 Axpy directly.  Keep the
                        // ping-pong MTE2 guards, but fuse contribution*prob +
                        // accumulator into one Vector pass.
                        Axpy(accumUb, contributionUb, prob, n);
                        PipeBarrier<PIPE_V>();
                        SetFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[slot]);
                        guardLive[slot] = true;
                        currentContribution = nextContribution;
                    }
                    // Consume every Vector-to-MTE2 guard before the same
                    // event IDs or UB slots are reused by another chunk.
                    for (int32_t guardSlot = 0; guardSlot < 2; ++guardSlot) {
                        if (guardLive[guardSlot]) {
                            WaitFlag<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[guardSlot]);
                        }
                    }
                }

                if constexpr (K16Specialized) {
                    if (readyEnd == chainLen) {
                        SyncFunc<AscendC::HardEvent::V_MTE3>();
                        uint32_t tokenRow = tokenRowBegin + localToken;
                        uint64_t outputOffset = (static_cast<uint64_t>(rankId) * m + tokenRow) * n;
                        DataCopy(gmRemoteD[outputOffset], accumUb, n);
                        SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
                        freeAccumMask |= 1U << accumSlot;
                        --remainingTokens;
                    } else {
                        // Keep the prefix in UB.  The same token owns this
                        // slot until its suffix completes, eliminating the
                        // old partial accumulator GM round-trip and fence.
                    }
                } else {
                    SyncFunc<AscendC::HardEvent::V_MTE3>();
                    if (readyEnd == chainLen) {
                        uint32_t tokenRow = tokenRowBegin + localToken;
                        uint64_t outputOffset = (static_cast<uint64_t>(rankId) * m + tokenRow) * n;
                        DataCopyPad(gmRemoteD[outputOffset], accumUb, rowCopyParams);
                        --remainingTokens;
                    } else {
                        DataCopyPad(combineAccumGmGlobal[static_cast<uint64_t>(firstCanonical) * n], accumUb,
                                    rowCopyParams);
                    }
                    // No subsequent MTE2 may reuse the row or observe the
                    // partial until the corresponding store drains.
                    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                }
                if constexpr (K16Specialized) {
                    int32_t nextProgressState = readyEnd;
                    if (readyEnd < chainLen) {
                        nextProgressState |= static_cast<int32_t>((accumSlot + 1U) << 8U);
                    }
                    chainProgressUb.SetValue(localToken, nextProgressState);
                } else {
                    chainProgressUb.SetValue(localToken, readyEnd);
                }
                advanced = true;
            }
            ++sweepStart;
            if (sweepStart == ownedTokens) {
                sweepStart = 0U;
            }
            if (probeEpoch == INT32_MAX) {
                // This is unreachable for practical kernel durations, but
                // clearing preserves correctness if the epoch space is ever
                // exhausted.  Ready hits may be rediscovered safely.
                Duplicate(readyCacheKeyUb, -1, OWNER_READY_CACHE_ENTRIES);
                Duplicate(readyCacheStateUb, 0, OWNER_READY_CACHE_ENTRIES);
                SyncFunc<AscendC::HardEvent::V_S>();
                probeEpoch = 2;
            } else {
                ++probeEpoch;
            }
            if (advanced) {
                // Work reappeared; resume the fast probing cadence.
                idlePollMisses = 0U;
                idlePollDelayCycles = 256;
            } else {
                GmSpinBackoffStep(idlePollMisses, idlePollDelayCycles);
            }
        }
    }
    if constexpr (K16Specialized) {
        // Every resident chain has completed when remainingTokens reaches
        // zero.  Consume the final store (or untouched initial) credit for
        // every slot before UB/event teardown.
        for (uint32_t accumSlot = 0U; accumSlot < K16_OWNER_ACCUM_SLOTS; ++accumSlot) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<AscendC::TEventID>(accumSlot));
        }
    } else {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[1]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE2>(contributionGuardEvent[0]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[1]);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(contributionReadyEvent[0]);
    TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, 0U));
}

// C7 production bulk Combine.  Dispatch appends canonical rows in the exact
// expert traversal order, so one AIV can consume a token's complete local
// chain after the GMM2 phase barrier.  This removes posDone polling and all
// intermediate GM accumulator round-trips: every owner emits exactly one
// ZeRC partial for a source token.
template <typename ElementC>
__aicore__ inline void CombineTokensByOwnerBulk(uint32_t rankId, GlobalTensor<ElementC> combineAccumGmGlobal,
                                                const Params &params, GlobalTensor<bfloat16_t> &distProbs,
                                                __gm__ int32_t *tokenChainDescriptors, __gm__ int32_t *tokenChainLen,
                                                uint32_t combineCoreCount, uint32_t combineCoreIdx,
                                                AscendTimerDevice &timer)
{
    uint32_t n = params.tilingData->k;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t m = params.tilingData->m;
    uint32_t topK = params.tilingData->topK;
    uint64_t maxOutputSize = params.tilingData->maxOutputSize;
    if (worldSize == 0U || combineCoreCount == 0U || combineCoreIdx >= combineCoreCount ||
        !CanPackBulkTokenChainDescriptor(topK, maxOutputSize)) {
        return;
    }

    // Large token spaces are split independently for every source rank so
    // every AIV observes a comparable local/remote traffic mix.  Decode-sized
    // spaces retain a flat partition to keep min(world*m, AIV) workers live.
    bool usePerSourcePartition = m >= combineCoreCount;
    uint32_t rowsPerSourceCore = m / combineCoreCount;
    uint32_t extraRows = m % combineCoreCount;
    uint32_t tokenRowBegin =
        combineCoreIdx * rowsPerSourceCore + (combineCoreIdx < extraRows ? combineCoreIdx : extraRows);
    uint32_t tokenRowCount = rowsPerSourceCore + (combineCoreIdx < extraRows ? 1U : 0U);
    uint32_t tokenRowEnd = tokenRowBegin + tokenRowCount;
    uint32_t totalTokens = static_cast<uint32_t>(static_cast<uint64_t>(worldSize) * m);
    uint32_t flatRowsPerCore = totalTokens / combineCoreCount;
    uint32_t flatExtraRows = totalTokens % combineCoreCount;
    uint32_t flatTokenBegin =
        combineCoreIdx * flatRowsPerCore + (combineCoreIdx < flatExtraRows ? combineCoreIdx : flatExtraRows);
    uint32_t flatTokenCount = flatRowsPerCore + (combineCoreIdx < flatExtraRows ? 1U : 0U);
    uint32_t flatTokenEnd = flatTokenBegin + flatTokenCount;

    uint32_t rowBytes = n * sizeof(ElementC);
    constexpr uint32_t MAX_BULK_TOKENS = 32U;
    uint32_t ubRows = rowBytes == 0U ? 0U : BUFFER_ALIGN / rowBytes;
    uint32_t batchCapacity = ubRows > 1U ? ubRows - 1U : 0U;
    batchCapacity = batchCapacity < MAX_BULK_TOKENS ? batchCapacity : MAX_BULK_TOKENS;
    if (batchCapacity == 0U) {
        return;
    }
    // Keep one final accumulator row for every token in the batch and one
    // shared contribution scratch row.  H2048 uses B=32; H3200 uses B=29 in
    // the established 192-KiB safe UB budget.  Consecutive token outputs are
    // emitted by one remote DMA plus one fence instead of one fence/token.
    LocalTensor<ElementC> batchAccum{AscendC::TPosition::VECIN, 0U, batchCapacity * n};
    LocalTensor<ElementC> contributionScratch{AscendC::TPosition::VECIN, batchCapacity * rowBytes, n};
    DataCopyExtParams rowCopyParams{1U, rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<ElementC> rowPadParams{false, 0U, 0U, 0U};
    uint64_t remoteDOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
    uint32_t slotBits = GetBulkTokenChainPayloadBits(topK);
    uint32_t slotMask = slotBits == 0U ? 0U : (static_cast<uint32_t>(1U) << slotBits) - 1U;
    GlobalTensor<ElementC> gmRemoteD;

    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_COMBINE, 0U));
    for (uint32_t sourceRank = 0U; sourceRank < worldSize; ++sourceRank) {
        uint32_t sourceTokenRowBegin = tokenRowBegin;
        uint32_t sourceTokenRowEnd = tokenRowEnd;
        if (!usePerSourcePartition) {
            uint32_t sourceTokenBegin = static_cast<uint32_t>(static_cast<uint64_t>(sourceRank) * m);
            uint32_t sourceTokenEnd = sourceTokenBegin + m;
            if (sourceTokenEnd <= flatTokenBegin) {
                continue;
            }
            if (sourceTokenBegin >= flatTokenEnd) {
                break;
            }
            uint32_t overlapBegin = flatTokenBegin > sourceTokenBegin ? flatTokenBegin : sourceTokenBegin;
            uint32_t overlapEnd = flatTokenEnd < sourceTokenEnd ? flatTokenEnd : sourceTokenEnd;
            sourceTokenRowBegin = overlapBegin - sourceTokenBegin;
            sourceTokenRowEnd = overlapEnd - sourceTokenBegin;
        }
        if (sourceTokenRowBegin >= sourceTokenRowEnd) {
            continue;
        }
        gmRemoteD.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementC *>(GetRankWinAddrWithOffset(sourceRank, remoteDOffset)));

        uint32_t tokenRow = sourceTokenRowBegin;
        while (tokenRow < sourceTokenRowEnd) {
            // A rank that owns no expert for a token emits no ZeRC partial.
            // Skip such rows before opening a contiguous DMA batch.
            while (tokenRow < sourceTokenRowEnd && ReadGmByPassDCache(tokenChainLen + sourceRank * m + tokenRow) <= 0) {
                ++tokenRow;
            }
            if (tokenRow >= sourceTokenRowEnd) {
                break;
            }

            uint32_t batchTokenBegin = tokenRow;
            uint32_t batchCount = 0U;
            bool scratchUsed = false;
            while (tokenRow < sourceTokenRowEnd && batchCount < batchCapacity) {
                uint32_t tokenIdx = static_cast<uint32_t>(static_cast<uint64_t>(sourceRank) * m + tokenRow);
                int32_t chainLen = ReadGmByPassDCache(tokenChainLen + tokenIdx);
                // Preserve a contiguous destination interval.  A zero-chain
                // token terminates this batch and is skipped by the outer
                // loop before the next one starts.
                if (chainLen <= 0) {
                    break;
                }
                if (chainLen > static_cast<int32_t>(topK)) {
                    chainLen = static_cast<int32_t>(topK);
                }

                LocalTensor<ElementC> accumBuf = batchAccum[static_cast<uint64_t>(batchCount) * n];
                int32_t validChainIdx = 0;
                for (int32_t chainIdx = 0; chainIdx < chainLen; ++chainIdx) {
                    uint64_t descriptorOffset =
                        static_cast<uint64_t>(tokenIdx) * topK + static_cast<uint32_t>(chainIdx);
                    uint32_t descriptor =
                        static_cast<uint32_t>(ReadGmByPassDCache(tokenChainDescriptors + descriptorOffset));
                    uint32_t canonicalRow = descriptor >> slotBits;
                    uint32_t topkSlot = descriptor & slotMask;
                    if (static_cast<uint64_t>(canonicalRow) >= maxOutputSize || topkSlot >= topK) {
                        continue;
                    }

                    LocalTensor<ElementC> inputBuf = validChainIdx == 0 ? accumBuf : contributionScratch;
                    if (validChainIdx > 0 && scratchUsed) {
                        SyncFunc<AscendC::HardEvent::V_MTE2>();
                    }
                    DataCopyPad(inputBuf, combineAccumGmGlobal[static_cast<uint64_t>(canonicalRow) * n], rowCopyParams,
                                rowPadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    uint32_t probOffset =
                        static_cast<uint32_t>((static_cast<uint64_t>(sourceRank) * m + tokenRow) * topK + topkSlot);
                    bfloat16_t prob = distProbs.GetValue(probOffset);
                    Muls(inputBuf, inputBuf, prob, n);
                    PipeBarrier<PIPE_V>();
                    if (validChainIdx > 0) {
                        Add(accumBuf, accumBuf, inputBuf, n);
                        PipeBarrier<PIPE_V>();
                        scratchUsed = true;
                    }
                    ++validChainIdx;
                }
                // Dispatch generated every descriptor in this chain, so this
                // is only a defensive guard for malformed metadata.
                if (validChainIdx == 0) {
                    break;
                }
                ++batchCount;
                ++tokenRow;
            }

            if (batchCount == 0U) {
                ++tokenRow;
                continue;
            }
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            uint64_t outputOffset = (static_cast<uint64_t>(rankId) * m + batchTokenBegin) * n;
            DataCopyExtParams batchCopyParams{1U, batchCount * rowBytes, 0U, 0U, 0U};
            DataCopyPad(gmRemoteD[outputOffset], batchAccum, batchCopyParams);
            // The next batch may reuse every accumulator and scratch row.
            // One fence covers up to 32 token partials.
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, 0U));
}

// Legacy readiness-sweep token-owner experiment retained only as an
// uninstantiated helper for source-level regression coverage.  No production
// call site selects it; C7 uses the barrier-published bulk helper above, while
// descriptor-capacity fallback and generic use CombineTokensParallel.
// One AIV owns a token's complete local expert chain.
// Dispatch appends one packed (canonical row, physical expert, top-k slot)
// descriptor in schedule order.  This removes per-step atomics and
// intermediate accumulator round-trips while preserving BF16 accumulation
// order, without doubling C3's per-contribution metadata stores.
template <bool EnableMGroupReady>
__aicore__ inline bool IsTokenOwnerChainReady(uint32_t tokenIdx, int32_t chainLen,
                                              __gm__ int32_t *tokenChainDescriptors, uint32_t expertPerRank,
                                              uint32_t topK, uint32_t outputColumns, uint64_t maxOutputSize,
                                              uint32_t blockNum, uint32_t mGroupsPerWave,
                                              LocalTensor<int32_t> expertPlan, __gm__ int32_t *dispatchTileReadyBase,
                                              uint32_t maxTilesPerExpert, __gm__ int32_t *gmm2ReadyFlagBasePtr)
{
    uint32_t nTiles = GetGmm2NTileCount(outputColumns);
    uint32_t slotBits = GetTokenOwnerFieldBits(topK);
    uint32_t expertBits = GetTokenOwnerFieldBits(expertPerRank);
    uint32_t payloadBits = slotBits + expertBits;
    uint32_t expertMask = expertBits == 0U ? 0U : (static_cast<uint32_t>(1U) << expertBits) - 1U;
    for (int32_t chainIdx = 0; chainIdx < chainLen; ++chainIdx) {
        uint64_t descriptorOffset = static_cast<uint64_t>(tokenIdx) * expertPerRank + static_cast<uint32_t>(chainIdx);
        uint32_t descriptor = static_cast<uint32_t>(ReadGmByPassDCache(tokenChainDescriptors + descriptorOffset));
        int32_t canonicalRow = static_cast<int32_t>(descriptor >> payloadBits);
        uint32_t expertIdx = (descriptor >> slotBits) & expertMask;
        if (expertIdx >= expertPerRank || static_cast<uint64_t>(canonicalRow) >= maxOutputSize) {
            continue;
        }
        int32_t localStart = expertIdx == 0U ? 0 : expertPlan.GetValue(expertIdx);
        int32_t localEnd = expertPlan.GetValue(1U + expertIdx);
        int32_t rowsThisExpert = localEnd - localStart;
        if (rowsThisExpert <= 0 || static_cast<uint64_t>(localStart) >= maxOutputSize) {
            continue;
        }
        uint64_t remainingRows = maxOutputSize - static_cast<uint64_t>(localStart);
        if (static_cast<uint64_t>(rowsThisExpert) > remainingRows) {
            rowsThisExpert = static_cast<int32_t>(remainingRows);
        }
        if (canonicalRow < localStart || canonicalRow >= localStart + rowsThisExpert) {
            continue;
        }

        uint32_t expertMGroups = GetGmm2MGroupCount(static_cast<uint32_t>(rowsThisExpert));
        bool useMGroupReady = false;
        if constexpr (EnableMGroupReady) {
            useMGroupReady = expertMGroups <= mGroupsPerWave &&
                             UseGmm2MGroupReady(static_cast<uint32_t>(rowsThisExpert), outputColumns, blockNum);
        }
        if (useMGroupReady) {
            uint32_t rowInExpert = static_cast<uint32_t>(canonicalRow - localStart);
            uint32_t mGroup = rowInExpert / EXPERT_SCHEDULE_M_TILE_ROWS;
            uint32_t groupRowStart = mGroup * EXPERT_SCHEDULE_M_TILE_ROWS;
            uint32_t groupRows = static_cast<uint32_t>(rowsThisExpert) - groupRowStart;
            groupRows = groupRows < EXPERT_SCHEDULE_M_TILE_ROWS ? groupRows : EXPERT_SCHEDULE_M_TILE_ROWS;
            int32_t target = static_cast<int32_t>(nTiles);
            __gm__ int32_t *readyAddr =
                dispatchTileReadyBase +
                (static_cast<uint64_t>(expertIdx) * maxTilesPerExpert + mGroup) * DISPATCH_TILE_READY_STRIDE +
                GMM2_READY_OFFSET;
            if (ReadGmByPassDCache(readyAddr) != target) {
                return false;
            }
        } else {
            __gm__ int32_t *readyAddr = gmm2ReadyFlagBasePtr + static_cast<uint64_t>(expertIdx) * INT_CACHELINE;
            if (ReadGmByPassDCache(readyAddr) != static_cast<int32_t>(blockNum)) {
                return false;
            }
        }
    }
    return true;
}

template <typename ElementC, bool EnableMGroupReady = false>
__aicore__ inline void CombineTokensByOwner(uint32_t rankId, GlobalTensor<ElementC> combineAccumGmGlobal,
                                            const Params &params, GlobalTensor<bfloat16_t> &distProbs,
                                            __gm__ int32_t *tokenChainDescriptors, __gm__ int32_t *tokenChainLen,
                                            __gm__ int32_t *tokenProcessed, __gm__ int32_t *metadataReadyFlag,
                                            __gm__ int32_t *gmm2ReadyFlagBasePtr, uint32_t mGroupsPerWave,
                                            uint32_t combineCoreCount, uint32_t combineCoreIdx,
                                            AscendTimerDevice &timer)
{
    int32_t n = static_cast<int32_t>(params.tilingData->k);
    uint32_t expertPerRank = params.tilingData->expertPerRank;
    uint32_t worldSize = params.tilingData->epWorldSize;
    uint32_t m = params.tilingData->m;
    uint32_t topK = params.tilingData->topK;
    uint32_t blockNum = GetBlockNum();
    uint64_t maxOutputSize = params.tilingData->maxOutputSize;

    if (worldSize == 0U || combineCoreCount == 0U) {
        return;
    }

    uint32_t pollMisses = 0U;
    int64_t pollDelayCycles = 100;
    while (ReadGmByPassDCache(metadataReadyFlag) < static_cast<int32_t>(blockNum)) {
        GmSpinBackoffStep(pollMisses, pollDelayCycles);
    }

    if (combineCoreIdx >= combineCoreCount) {
        return;
    }
    // For prefill, split every source rank independently across all AIV1
    // cores.  The old flat source-major partition made roughly half the EP2
    // cores local-only writers and half remote-only writers; quotient/remainder
    // slices give every core a similar local/remote mix.  When m is smaller
    // than the core count, retain flat source-major ownership so up to
    // min(world*m, cores) AIVs remain active instead of collapsing to m AIVs.
    bool usePerSourcePartition = m >= combineCoreCount;
    uint32_t rowsPerSourceCore = m / combineCoreCount;
    uint32_t extraRows = m % combineCoreCount;
    uint32_t tokenRowBegin =
        combineCoreIdx * rowsPerSourceCore + (combineCoreIdx < extraRows ? combineCoreIdx : extraRows);
    uint32_t tokenRowCount = rowsPerSourceCore + (combineCoreIdx < extraRows ? 1U : 0U);
    uint32_t tokenRowEnd = tokenRowBegin + tokenRowCount;
    uint32_t totalTokens = worldSize * m;
    uint32_t flatRowsPerCore = totalTokens / combineCoreCount;
    uint32_t flatExtraRows = totalTokens % combineCoreCount;
    uint32_t flatTokenBegin =
        combineCoreIdx * flatRowsPerCore + (combineCoreIdx < flatExtraRows ? combineCoreIdx : flatExtraRows);
    uint32_t flatTokenCount = flatRowsPerCore + (combineCoreIdx < flatExtraRows ? 1U : 0U);
    uint32_t flatTokenEnd = flatTokenBegin + flatTokenCount;
    if ((usePerSourcePartition && tokenRowBegin >= tokenRowEnd) ||
        (!usePerSourcePartition && flatTokenBegin >= flatTokenEnd)) {
        return;
    }

    uint32_t rowBytes = static_cast<uint32_t>(n) * sizeof(ElementC);
    LocalTensor<ElementC> rowBuf0{AscendC::TPosition::VECIN, 0U, static_cast<uint32_t>(n)};
    LocalTensor<ElementC> rowBuf1{AscendC::TPosition::VECIN, rowBytes, static_cast<uint32_t>(n)};
    DataCopyExtParams rowCopyParams{1U, static_cast<uint32_t>(n * sizeof(ElementC)), 0U, 0U, 0U};
    DataCopyPadExtParams<ElementC> rowPadParams{false, 0U, 0U, 0U};
    uint64_t remoteDOffset = params.peermemInfo.ptrD - params.peermemInfo.ptrBase;
    GlobalTensor<ElementC> gmRemoteD;

    GlobalTensor<int32_t> expertPlanGm;
    expertPlanGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrExpertOrderScratch));
    uint32_t expertPlanWords =
        Ops::Base::CeilAlign(expertPerRank + 1U, static_cast<uint32_t>(ALIGN_32 / sizeof(int32_t)));
    LocalTensor<int32_t> expertPlan{AscendC::TPosition::VECIN, 2U * rowBytes, expertPlanWords};
    DataCopyPad(expertPlan, expertPlanGm,
                DataCopyExtParams{1U, static_cast<uint32_t>((expertPerRank + 1U) * sizeof(int32_t)), 0U, 0U, 0U},
                DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U});
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    __gm__ int32_t *dispatchTileReadyBase =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.ptrDispatchTileReady);
    uint32_t maxTilesPerExpert =
        static_cast<uint32_t>(Ops::Base::CeilDiv(maxOutputSize, static_cast<uint64_t>(EXPERT_SCHEDULE_M_TILE_ROWS)));
    uint32_t slotBits = GetTokenOwnerFieldBits(topK);
    uint32_t expertBits = GetTokenOwnerFieldBits(expertPerRank);
    uint32_t payloadBits = slotBits + expertBits;
    uint32_t slotMask = slotBits == 0U ? 0U : (static_cast<uint32_t>(1U) << slotBits) - 1U;
    uint32_t expertMask = expertBits == 0U ? 0U : (static_cast<uint32_t>(1U) << expertBits) - 1U;

    TIMER_BLOCK(timer.Tik(AscendTimer::AIV_COMBINE, 0U));
    bool allTokensComplete = false;
    uint32_t idlePollMisses = 0U;
    int64_t idlePollDelayCycles = 100;
    while (!allTokensComplete) {
        uint32_t pendingCount = 0U;
        uint32_t completedCount = 0U;
        for (uint32_t sourceRank = 0U; sourceRank < worldSize; ++sourceRank) {
            uint32_t sourceTokenRowBegin = tokenRowBegin;
            uint32_t sourceTokenRowEnd = tokenRowEnd;
            if (!usePerSourcePartition) {
                uint32_t sourceTokenBegin = sourceRank * m;
                uint32_t sourceTokenEnd = sourceTokenBegin + m;
                if (sourceTokenEnd <= flatTokenBegin) {
                    continue;
                }
                if (sourceTokenBegin >= flatTokenEnd) {
                    break;
                }
                uint32_t overlapBegin = flatTokenBegin > sourceTokenBegin ? flatTokenBegin : sourceTokenBegin;
                uint32_t overlapEnd = flatTokenEnd < sourceTokenEnd ? flatTokenEnd : sourceTokenEnd;
                sourceTokenRowBegin = overlapBegin - sourceTokenBegin;
                sourceTokenRowEnd = overlapEnd - sourceTokenBegin;
            }
            gmRemoteD.SetGlobalBuffer(
                reinterpret_cast<__gm__ ElementC *>(GetRankWinAddrWithOffset(sourceRank, remoteDOffset)));
            for (uint32_t tokenRow = sourceTokenRowBegin; tokenRow < sourceTokenRowEnd; ++tokenRow) {
                uint32_t tokenIdx = sourceRank * m + tokenRow;
                if (ReadGmByPassDCache(tokenProcessed + tokenIdx) != 0) {
                    continue;
                }
                pendingCount++;
                int32_t chainLen = ReadGmByPassDCache(tokenChainLen + tokenIdx);
                if (chainLen <= 0) {
                    WriteGmByPassDCache(tokenProcessed + tokenIdx, int32_t(1));
                    completedCount++;
                    continue;
                }
                if (!IsTokenOwnerChainReady<EnableMGroupReady>(tokenIdx, chainLen, tokenChainDescriptors, expertPerRank,
                                                               topK, static_cast<uint32_t>(n), maxOutputSize, blockNum,
                                                               mGroupsPerWave, expertPlan, dispatchTileReadyBase,
                                                               maxTilesPerExpert, gmm2ReadyFlagBasePtr)) {
                    continue;
                }

                // Readiness is monotonic.  Once the complete chain passes the
                // precheck, process it in the original schedule order with a
                // single final remote write and no per-contribution spin.
                LocalTensor<ElementC> accumBuf = rowBuf0;
                int32_t validChainIdx = 0;
                for (int32_t chainIdx = 0; chainIdx < chainLen; ++chainIdx) {
                    uint64_t descriptorOffset =
                        static_cast<uint64_t>(tokenIdx) * expertPerRank + static_cast<uint32_t>(chainIdx);
                    uint32_t descriptor =
                        static_cast<uint32_t>(ReadGmByPassDCache(tokenChainDescriptors + descriptorOffset));
                    int32_t canonicalRow = static_cast<int32_t>(descriptor >> payloadBits);
                    uint32_t expertIdx = (descriptor >> slotBits) & expertMask;
                    if (expertIdx >= expertPerRank || static_cast<uint64_t>(canonicalRow) >= maxOutputSize) {
                        continue;
                    }
                    int32_t localStart = expertIdx == 0U ? 0 : expertPlan.GetValue(expertIdx);
                    int32_t localEnd = expertPlan.GetValue(1U + expertIdx);
                    int32_t rowsThisExpert = localEnd - localStart;
                    if (rowsThisExpert <= 0 || localStart < 0 || static_cast<uint64_t>(localStart) >= maxOutputSize) {
                        continue;
                    }
                    uint64_t remainingRows = maxOutputSize - static_cast<uint64_t>(localStart);
                    if (static_cast<uint64_t>(rowsThisExpert) > remainingRows) {
                        rowsThisExpert = static_cast<int32_t>(remainingRows);
                    }
                    if (canonicalRow < localStart || canonicalRow >= localStart + rowsThisExpert) {
                        continue;
                    }
                    uint32_t topkSlot = descriptor & slotMask;
                    if (topkSlot >= topK) {
                        continue;
                    }
                    bool useRow0 = (validChainIdx & 1) == 0;
                    LocalTensor<ElementC> rowBuf = useRow0 ? rowBuf0 : rowBuf1;
                    if (validChainIdx >= 2) {
                        // This ping-pong row was the previous Add input.
                        // Explicitly join V before MTE2 overwrites it.
                        SyncFunc<AscendC::HardEvent::V_MTE2>();
                    }
                    DataCopyPad(rowBuf, combineAccumGmGlobal[static_cast<int64_t>(canonicalRow) * n], rowCopyParams,
                                rowPadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_V>();
                    bfloat16_t prob = distProbs.GetValue(sourceRank * m * topK + tokenRow * topK + topkSlot);
                    Muls(rowBuf, rowBuf, prob, n);
                    PipeBarrier<PIPE_V>();
                    if (validChainIdx > 0) {
                        Add(rowBuf, rowBuf, accumBuf, n);
                        PipeBarrier<PIPE_V>();
                    }
                    accumBuf = rowBuf;
                    validChainIdx++;
                }

                if (validChainIdx == 0) {
                    WriteGmByPassDCache(tokenProcessed + tokenIdx, int32_t(1));
                    completedCount++;
                    continue;
                }
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                int64_t outputOffset = (static_cast<int64_t>(rankId) * m + tokenRow) * n;
                DataCopyPad(gmRemoteD[outputOffset], accumBuf, rowCopyParams);
                // Do not let the next token reuse either ping-pong row until
                // the remote MTE3 store has consumed the final accumulator.
                // The former two-outstanding-event protocol was not covered
                // by device tests and produced finite but incorrect large-BS
                // outputs on A5.  Publish tokenProcessed only after the store
                // is globally complete, matching the proven C5 ownership
                // contract while retaining C6 descriptor/partition changes.
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                WriteGmByPassDCache(tokenProcessed + tokenIdx, int32_t(1));
                completedCount++;
            }
        }
        allTokensComplete = pendingCount == completedCount;
        if (completedCount > 0U) {
            // Work reappeared; resume the fast probing cadence.
            idlePollMisses = 0U;
            idlePollDelayCycles = 100;
        } else if (!allTokensComplete) {
            GmSpinBackoffStep(idlePollMisses, idlePollDelayCycles);
        }
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    TIMER_BLOCK(timer.template Tok<Overwrite>(AscendTimer::AIV_COMBINE, 0U));
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H
