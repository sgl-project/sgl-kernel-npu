/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "kernel_operator.h"
#define C_TEMPLATE 0
#define V_TEMPLATE 1

// The top-level AscendC build does not inject the cube event identifiers that
// the original OPP generator supplied.
#ifndef EVENT_ID0
#define EVENT_ID0 0
#endif
#ifndef EVENT_ID1
#define EVENT_ID1 1
#endif
#ifndef EVENT_ID2
#define EVENT_ID2 2
#endif
#ifndef EVENT_ID3
#define EVENT_ID3 3
#endif
#ifndef EVENT_ID4
#define EVENT_ID4 4
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 5
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 6
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 7
#endif

#include "fused_copy_sfa_tiling_data.h"

using FusedScatterCopySparseFlashAttentionTilingData = FusedCopySfaTilingData;
using FusedScatterCopySparseFlashAttentionTilingDataMla = FusedCopySfaTilingData;

#include "sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_kernel_mla.h"
#include "first_fill_scatter_copy_kernel.h"
#include "short_resident_copy.h"

using namespace AscendC;
namespace {
template <typename T, bool COMBINED_LAYOUT = false>
__aicore__ inline void RunFirstFillScatterCopy(
    __gm__ uint8_t *hbmKeyRope, __gm__ uint8_t *hbmKvCache, __gm__ uint8_t *dramKeyRope, __gm__ uint8_t *dramKvCache,
    __gm__ uint8_t *hbmBlockTable, __gm__ uint8_t *dramBlockTable, __gm__ uint8_t *missSourceIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCounts, __gm__ uint8_t *qEnds, bool refreshOnly,
    __gm__ uint8_t *current, __gm__ uint8_t *positions, __gm__ uint8_t *currentValid,
    const __gm__ FusedScatterCopySparseFlashAttentionTilingData *fusedTiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        FirstFillScatterCopyNs::FirstFillScatterCopyKernel<T, COMBINED_LAYOUT> copy(pipe, fusedTiling);
        copy.Init(hbmKeyRope, hbmKvCache, dramKeyRope, dramKvCache, hbmBlockTable, dramBlockTable, missSourceIds,
                  missDstSlots, missCounts, qEnds, refreshOnly, current, positions, currentValid,
                  fusedTiling->residentEnabled != 0U);
        copy.Process();
    }
}

__aicore__ inline uint32_t GetBatchCopyMode(__gm__ uint8_t *cacheTokens, __gm__ uint8_t *missCounts, uint32_t batchSize,
                                            uint32_t missCap, bool fullVisible, bool residentRefreshTail)
{
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> missCountsGm;
    cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens, batchSize);
    missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
    uint32_t mode = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        const int32_t cacheTokenCount = cacheTokensGm.GetValue(batchIdx);
        const int32_t missCount = missCountsGm.GetValue(batchIdx);
        ASSERT_MSG(cacheTokenCount >= 0, "num_cache_tokens must be non-negative.");
        ASSERT_MSG(missCount >= 0 && static_cast<uint32_t>(missCount) <= missCap,
                   "miss_count exceeds the request-level input capacity.");
        // Padding rows have both counts set to zero. They must not select the
        // batch-wide first-fill path for otherwise steady requests.
        if (cacheTokenCount > 0 && missCount >= cacheTokenCount) {
            return 1;  // Cold fill: all request-level copies precede attention.
        }
        if (cacheTokenCount > 0 && fullVisible && missCount > 0) {
            // Only native LIM producers guarantee the resident-refresh suffix.
            // Preserve the complete-copy contract for all legacy callers.
            mode = residentRefreshTail ? 2U : 1U;
        }
    }
    return mode;
}

template <typename T, bool SOURCE_AWARE, bool COMBINED_LAYOUT = false>
__aicore__ inline void
RunFusedMtp(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
            __gm__ uint8_t *cacheTokens, __gm__ uint8_t *hbmBlockTable, __gm__ uint8_t *actualSeqLengthsQuery,
            __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope, __gm__ uint8_t *hbmKeyRope,
            __gm__ uint8_t *dramKeyRope, __gm__ uint8_t *dramKvCache, __gm__ uint8_t *dramBlockTable,
            __gm__ uint8_t *topkSourceIds, __gm__ uint8_t *topkMissCounts, __gm__ uint8_t *missSourceIds,
            __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCounts, __gm__ uint8_t *attentionOut,
            __gm__ uint8_t *attentionWorkspace, __gm__ uint8_t *current, __gm__ uint8_t *positions,
            __gm__ uint8_t *currentValid, const __gm__ FusedScatterCopySparseFlashAttentionTilingData *fusedTiling,
            __gm__ uint8_t *tiling, TPipe *pipe)
{
    // Request-level payloads are consumed by the first-fill stage in this
    // kernel. Attention only reads missCounts for the batch path decision.
    (void)missSourceIds;
    (void)missDstSlots;
    using MtpType = SFAType<T, T, T, false, SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND, V_TEMPLATE, SFA_STAGE_NORMAL,
                            SOURCE_AWARE, true, COMBINED_LAYOUT>;
    FusedScatterCopySparseFlashAttentionMla<MtpType> attention;
    const auto *attentionTiling =
        reinterpret_cast<const __gm__ FusedScatterCopySparseFlashAttentionTilingDataMla *>(fusedTiling);
    attention.Init(query, key, value, sparseIndices, cacheTokens, nullptr, actualSeqLengthsQuery, actualSeqLengthsKv,
                   hbmBlockTable, queryRope, hbmKeyRope, attentionOut, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, attentionWorkspace, attentionTiling, tiling, pipe);
    if constexpr (SOURCE_AWARE) {
        attention.InitSourceAwareGather(dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                                        fusedTiling->copyCap, fusedTiling->dramMaxBlockNum, current, positions,
                                        currentValid, actualSeqLengthsQuery, fusedTiling->residentCurrentRows,
                                        fusedTiling->residentHostRows, fusedTiling->residentEnabled != 0U);
    }
    attention.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fused_scatter_copy_sparse_flash_attention_combined(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR cacheTokens, GM_ADDR hbmBlockTable,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv, GM_ADDR queryRope, GM_ADDR hbmKeyRope,
    GM_ADDR dramKeyRope, GM_ADDR dramKvCache, GM_ADDR dramBlockTable, GM_ADDR topkSourceIds, GM_ADDR topkMissCounts,
    GM_ADDR missSourceIds, GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR attentionOut, GM_ADDR current,
    GM_ADDR positions, GM_ADDR currentValid, GM_ADDR requestSlots, GM_ADDR reqToToken, GM_ADDR residentBegin,
    GM_ADDR residentEnd, GM_ADDR workspace, GM_ADDR tiling)
{
    // AscendC's HAVE_WORKSPACE + HAVE_TILING wrapper uses the last two
    // arguments for system workspace and tiling; keep them in that order.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto *fusedTiling = reinterpret_cast<const __gm__ FusedCopySfaTilingData *>(tiling);
    const bool isBf16 = (fusedTiling->baseParams.dispatchKey & (1U << 16)) != 0;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    const uint32_t copyMode =
        GetBatchCopyMode(cacheTokens, missCounts, fusedTiling->baseParams.batchSize, fusedTiling->missCap,
                         fusedTiling->fullVisible != 0U, fusedTiling->residentRefreshTail != 0U);
    const bool firstFill = copyMode == 1;
    bool hasShort = false;
    if (fusedTiling->residentEnabled) {
        GlobalTensor<int32_t> lengthsGm, cacheTokensGm;
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKv);
        cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        for (uint32_t b = 0; b < fusedTiling->baseParams.batchSize; ++b) {
            if (lengthsGm.GetValue(b) > 0 && cacheTokensGm.GetValue(b) == 0) hasShort = true;
        }
    }
    ShortResidentCopy::Copy<bfloat16_t> residentBf16;
    ShortResidentCopy::Copy<half> residentHalf;
    if (fusedTiling->residentEnabled) {
        if ASCEND_IS_AIV {
            const uint32_t cores = fusedTiling->singleCoreParams.usedCoreNum * 2U;
            if (isBf16) {
                residentBf16.Init(&pipe, current, dramKvCache, key, reqToToken, requestSlots, positions, currentValid,
                                  actualSeqLengthsQuery, hbmBlockTable, actualSeqLengthsKv, cacheTokens, residentBegin,
                                  residentEnd, fusedTiling->baseParams.batchSize, fusedTiling->residentRows,
                                  fusedTiling->residentContext, fusedTiling->hbmMaxBlockNum * 128U,
                                  fusedTiling->residentHostRows, fusedTiling->residentCurrentRows,
                                  fusedTiling->residentHbmPages, cores, hasShort);
                residentBf16.ValidateCurrent();
                if (hasShort) residentBf16.Process();
            } else {
                residentHalf.Init(&pipe, current, dramKvCache, key, reqToToken, requestSlots, positions, currentValid,
                                  actualSeqLengthsQuery, hbmBlockTable, actualSeqLengthsKv, cacheTokens, residentBegin,
                                  residentEnd, fusedTiling->baseParams.batchSize, fusedTiling->residentRows,
                                  fusedTiling->residentContext, fusedTiling->hbmMaxBlockNum * 128U,
                                  fusedTiling->residentHostRows, fusedTiling->residentCurrentRows,
                                  fusedTiling->residentHbmPages, cores, hasShort);
                residentHalf.ValidateCurrent();
                if (hasShort) residentHalf.Process();
            }
            if (hasShort && copyMode != 0) {
                PipeBarrier<PIPE_ALL>();
                pipe.Reset();
            }
        }
    }
    if (copyMode != 0) {
        if (isBf16) {
            RunFirstFillScatterCopy<bfloat16_t, true>(
                hbmKeyRope, key, dramKeyRope, dramKvCache, hbmBlockTable, dramBlockTable, missSourceIds, missDstSlots,
                missCounts, actualSeqLengthsQuery, copyMode == 2, current, positions, currentValid, fusedTiling, &pipe);
        } else {
            RunFirstFillScatterCopy<half, true>(
                hbmKeyRope, key, dramKeyRope, dramKvCache, hbmBlockTable, dramBlockTable, missSourceIds, missDstSlots,
                missCounts, actualSeqLengthsQuery, copyMode == 2, current, positions, currentValid, fusedTiling, &pipe);
        }
        // All MIX kernel tasks must reach the barrier. It establishes the
        // first-fill MTE3-to-Attention MTE2 dependency before either pipeline reads HBM.
    }
    if (copyMode != 0 || hasShort) {
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
    }
    if (fusedTiling->residentEnabled) {
        if ASCEND_IS_AIV {
            if (isBf16)
                residentBf16.Commit();
            else
                residentHalf.Commit();
        }
    }
    if (!isBf16) {
        if (firstFill) {
            RunFusedMtp<half, false, true>(query, key, value, sparseIndices, cacheTokens, hbmBlockTable,
                                           actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope,
                                           dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                                           missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace,
                                           current, positions, currentValid, fusedTiling, tiling, &pipe);
        } else {
            RunFusedMtp<half, true, true>(query, key, value, sparseIndices, cacheTokens, hbmBlockTable,
                                          actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope, dramKeyRope,
                                          dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts, missSourceIds,
                                          missDstSlots, missCounts, attentionOut, attentionWorkspace, current,
                                          positions, currentValid, fusedTiling, tiling, &pipe);
        }
    } else {
        if (firstFill) {
            RunFusedMtp<bfloat16_t, false, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, current, positions,
                currentValid, fusedTiling, tiling, &pipe);
        } else {
            RunFusedMtp<bfloat16_t, true, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, current, positions,
                currentValid, fusedTiling, tiling, &pipe);
        }
    }
}
