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
using FusedScatterCopySparseFlashAttentionTilingDataMla = SparseFlashAttentionTilingDataMla;

#include "sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_kernel_mla.h"
#include "first_fill_scatter_copy_kernel.h"

using namespace AscendC;
namespace {
template <typename T, bool COMBINED_LAYOUT = false>
__aicore__ inline void
RunFirstFillScatterCopy(__gm__ uint8_t *hbmKeyRope, __gm__ uint8_t *hbmKvCache, __gm__ uint8_t *dramKeyRope,
                        __gm__ uint8_t *dramKvCache, __gm__ uint8_t *hbmBlockTable, __gm__ uint8_t *dramBlockTable,
                        __gm__ uint8_t *missSourceIds, __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCounts,
                        const __gm__ FusedScatterCopySparseFlashAttentionTilingData *fusedTiling, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        FirstFillScatterCopyNs::FirstFillScatterCopyKernel<T, COMBINED_LAYOUT> copy(pipe, fusedTiling);
        copy.Init(hbmKeyRope, hbmKvCache, dramKeyRope, dramKvCache, hbmBlockTable, dramBlockTable, missSourceIds,
                  missDstSlots, missCounts);
        copy.Process();
    }
}

__aicore__ inline bool IsBatchFirstFill(__gm__ uint8_t *cacheTokens, __gm__ uint8_t *missCounts, uint32_t batchSize,
                                        uint32_t missCap)
{
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> missCountsGm;
    cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens, batchSize);
    missCountsGm.SetGlobalBuffer((__gm__ int32_t *)missCounts, batchSize);
    bool firstFill = false;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        const int32_t cacheTokenCount = cacheTokensGm.GetValue(batchIdx);
        const int32_t missCount = missCountsGm.GetValue(batchIdx);
        ASSERT_MSG(cacheTokenCount >= 0, "num_cache_tokens must be non-negative.");
        ASSERT_MSG(missCount >= 0 && static_cast<uint32_t>(missCount) <= missCap,
                   "miss_count exceeds the request-level input capacity.");
        // Padding rows have both counts set to zero. They must not select the
        // batch-wide first-fill path for otherwise steady requests.
        if (cacheTokenCount > 0 && missCount >= cacheTokenCount) {
            firstFill = true;
        }
    }
    return firstFill;
}

template <typename T, bool SOURCE_AWARE, bool COMBINED_LAYOUT = false>
__aicore__ inline void RunFusedMtp(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *cacheTokens,
                                   __gm__ uint8_t *hbmBlockTable, __gm__ uint8_t *actualSeqLengthsQuery,
                                   __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope,
                                   __gm__ uint8_t *hbmKeyRope, __gm__ uint8_t *dramKeyRope, __gm__ uint8_t *dramKvCache,
                                   __gm__ uint8_t *dramBlockTable, __gm__ uint8_t *topkSourceIds,
                                   __gm__ uint8_t *topkMissCounts, __gm__ uint8_t *missSourceIds,
                                   __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCounts,
                                   __gm__ uint8_t *attentionOut, __gm__ uint8_t *attentionWorkspace,
                                   const __gm__ FusedScatterCopySparseFlashAttentionTilingData *fusedTiling,
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
                                        fusedTiling->copyCap, fusedTiling->dramMaxBlockNum);
    }
    attention.Process();
}

}  // namespace

extern "C" __global__ __aicore__ void fused_scatter_copy_sparse_flash_attention_combined(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR cacheTokens, GM_ADDR hbmBlockTable,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv, GM_ADDR queryRope, GM_ADDR hbmKeyRope,
    GM_ADDR dramKeyRope, GM_ADDR dramKvCache, GM_ADDR dramBlockTable, GM_ADDR topkSourceIds, GM_ADDR topkMissCounts,
    GM_ADDR missSourceIds, GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR attentionOut, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto *fusedTiling = reinterpret_cast<const __gm__ FusedCopySfaTilingData *>(tiling);
    const bool isBf16 = (fusedTiling->baseParams.dispatchKey & (1U << 16)) != 0;

    TPipe pipe;
    __gm__ uint8_t *attentionWorkspace = GetUserWorkspace(workspace);
    const bool firstFill =
        IsBatchFirstFill(cacheTokens, missCounts, fusedTiling->baseParams.batchSize, fusedTiling->missCap);
    if (firstFill) {
        if (isBf16) {
            RunFirstFillScatterCopy<bfloat16_t, true>(hbmKeyRope, key, dramKeyRope, dramKvCache, hbmBlockTable,
                                                      dramBlockTable, missSourceIds, missDstSlots, missCounts,
                                                      fusedTiling, &pipe);
        } else {
            RunFirstFillScatterCopy<half, true>(hbmKeyRope, key, dramKeyRope, dramKvCache, hbmBlockTable,
                                                dramBlockTable, missSourceIds, missDstSlots, missCounts, fusedTiling,
                                                &pipe);
        }
        // All MIX kernel tasks must reach the barrier. It establishes the
        // first-fill MTE3-to-Attention MTE2 dependency before either pipeline reads HBM.
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
    }
    if (!isBf16) {
        if (firstFill) {
            RunFusedMtp<half, false, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, fusedTiling, tiling, &pipe);
        } else {
            RunFusedMtp<half, true, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, fusedTiling, tiling, &pipe);
        }
    } else {
        if (firstFill) {
            RunFusedMtp<bfloat16_t, false, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, fusedTiling, tiling, &pipe);
        } else {
            RunFusedMtp<bfloat16_t, true, true>(
                query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv,
                queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts,
                missSourceIds, missDstSlots, missCounts, attentionOut, attentionWorkspace, fusedTiling, tiling, &pipe);
        }
    }
}
