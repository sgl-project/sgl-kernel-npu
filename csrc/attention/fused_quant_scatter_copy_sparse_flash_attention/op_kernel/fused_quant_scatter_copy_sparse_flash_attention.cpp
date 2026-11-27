// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// CANN-2.0. Concrete normal-wheel launch entries adapted from PR #18490.
#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PREFETCH
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PREFETCH 1
#endif
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "fused_quant_copy_sfa_tiling_data.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_copy.h"
using FusedQuantScatterCopySparseFlashAttentionTilingDataMla = FusedQuantScatterCopySparseFlashAttentionTilingData;
#include "arch35/fused_quant_scatter_copy_sparse_flash_attention_kernel_mla.h"
template <typename QueryType, bool SPLIT_HEADS>
__aicore__ inline void RunQuantCopySfa(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes, GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
    GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling, GM_ADDR queryRope = nullptr,
    const fused_quant_scatter_copy_sfa::PackedResidentInputs &residentInputs = {}, bool nativeLimPlan = false)
{
    (void)hbmKvAlias;
    const auto *gm = reinterpret_cast<const __gm__ FusedQuantScatterCopySparseFlashAttentionTilingData *>(tiling);
    FusedQuantScatterCopySparseFlashAttentionTilingData tilingData{
        {gm->baseParams.batchSize, gm->baseParams.seqSize, gm->baseParams.qSeqSize, gm->baseParams.nNumOfQInOneGroup,
         gm->baseParams.sparseBlockCount, gm->baseParams.sparseMode, gm->baseParams.outputLayout,
         gm->baseParams.blockSize, gm->baseParams.maxBlockNumPerBatch, gm->baseParams.isActualLenDimsNull,
         gm->baseParams.isActualLenDimsKVNull, gm->baseParams.dSizeVInput, gm->baseParams.scaleValue},
        {gm->singleCoreParams.usedCoreNum},
        gm->dramMaxBlocks,
        gm->hbmPhysicalBlocks,
        gm->dramPhysicalBlocks,
        {gm->reserved[0]}};
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const bool fullVisible = queryRope != nullptr;
    bool hasShort = false;
    fused_quant_scatter_copy_sfa::PackedResidentStage resident;
    if (residentInputs.current != nullptr) {
        resident.Init(residentInputs, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, pipe);
    }
    const auto copyMode = fused_quant_scatter_copy_sfa::SelectCopyMode(
        missCounts, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, tilingData, fullVisible,
        resident.Enabled(), nativeLimPlan, hasShort);
    const bool stageShort = resident.Enabled() && hasShort;
    if (copyMode != fused_quant_scatter_copy_sfa::PackedCopyMode::NONE || stageShort) {
        fused_quant_scatter_copy_sfa::StageKv(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, missSrcIds,
                                              missDstSlots, missCounts, tilingData, resident, copyMode, stageShort);
    }
    if (resident.Enabled()) resident.PublishExtents(tilingData);
    using Cube = typename std::conditional<
        g_coreType == AscendC::AIC,
        BaseApi::QSFAMatmulService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true, QSFA_LAYOUT::TND,
                                   QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAMatmulServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true, QSFA_LAYOUT::TND,
                                        QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    using Vector = typename std::conditional<
        g_coreType == AscendC::AIC,
        BaseApi::QSFAVectorServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true, QSFA_LAYOUT::TND,
                                        QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAVectorService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true, QSFA_LAYOUT::TND,
                                   QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    BaseApi::FusedQuantScatterCopySparseFlashAttentionMla<Cube, Vector> op;
    op.Init(query, hbmKvBytes, hbmKvBytes, topkDstSlots, nullptr, nullptr, hbmBlockTable, actualSeqLengthsQuery,
            actualSeqLengthsKv, fullVisible ? nullptr : numCacheTokens, attentionOut, userWorkspace, &tilingData, &pipe,
            queryRope);
    op.InitSourceAwareGather(
        dramKvBytes, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, &tilingData,
        copyMode != fused_quant_scatter_copy_sfa::PackedCopyMode::COMPLETE,
        userWorkspace + fused_quant_scatter_copy_sfa::AttentionWorkspaceBytes(tilingData.singleCoreParams.usedCoreNum),
        residentInputs.current, residentInputs.positions, residentInputs.valid);
    op.Process();
}

template <typename QueryType, bool SPLIT_HEADS>
__aicore__ inline void RunQuantCopySfaV2(GM_ADDR queryNope, GM_ADDR queryRope, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
                                         GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
                                         GM_ADDR topkSrcIds, GM_ADDR topkMissCounts, GM_ADDR actualSeqLengthsQuery,
                                         GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds,
                                         GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR hbmKvAlias,
                                         GM_ADDR attentionOut, GM_ADDR current, GM_ADDR positions, GM_ADDR currentValid,
                                         GM_ADDR requestSlots, GM_ADDR reqToToken, GM_ADDR residentEnd,
                                         GM_ADDR workspace, GM_ADDR tiling)
{
    const auto *gm = reinterpret_cast<const __gm__ FusedQuantCopySfaResidentTilingData *>(tiling);
    const fused_quant_scatter_copy_sfa::PackedResidentInputs resident{
        current, positions, currentValid, requestSlots, reqToToken, residentEnd, gm->requestRows, gm->requestContext};
    RunQuantCopySfa<QueryType, SPLIT_HEADS>(
        queryNope, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts,
        actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias,
        attentionOut, workspace, tiling, queryRope, resident, gm->nativeLimPlan != 0);
}
#endif
// Standard CANN entry used by the refactored build. The original four direct
// launch symbols below remain unchanged for the legacy wheel and its 80-byte
// ABI. The refactored tiler records only dispatch bits in the final reserved
// word, leaving every historical field and offset intact.
__global__ __aicore__ void sgl_fused_quant_scatter_copy_sparse_flash_attention(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes, GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
    GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto *gm = reinterpret_cast<const __gm__ FusedQuantScatterCopySparseFlashAttentionTilingData *>(tiling);
    const bool bf16 = (gm->reserved[0] & 1U) != 0;
    const bool splitHeads = (gm->reserved[0] & 2U) != 0;
    if (bf16) {
        if (splitHeads) {
            RunQuantCopySfa<bfloat16_t, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable,
                                              topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery,
                                              actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts,
                                              hbmKvAlias, attentionOut, workspace, tiling);
        } else {
            RunQuantCopySfa<bfloat16_t, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable,
                                               topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery,
                                               actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts,
                                               hbmKvAlias, attentionOut, workspace, tiling);
        }
    } else if (splitHeads) {
        RunQuantCopySfa<half, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots,
                                    topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv,
                                    numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut,
                                    workspace, tiling);
    } else {
        RunQuantCopySfa<half, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots,
                                     topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv,
                                     numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut,
                                     workspace, tiling);
    }
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_bf16(GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
                                                     GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
                                                     GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
                                                     GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
                                                     GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
                                                     GM_ADDR missCounts, GM_ADDR hbmKvAlias, GM_ADDR attentionOut,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<bfloat16_t, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots,
                                       topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv,
                                       numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut,
                                       workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_bf16_split(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes, GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
    GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<bfloat16_t, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots,
                                      topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv,
                                      numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut,
                                      workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_fp16(GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
                                                     GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
                                                     GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
                                                     GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
                                                     GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
                                                     GM_ADDR missCounts, GM_ADDR hbmKvAlias, GM_ADDR attentionOut,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<half, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots,
                                 topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens,
                                 missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_fp16_split(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes, GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable,
    GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts, GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<half, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds,
                                topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds,
                                missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}

#if (__CCE_AICORE__ == 310)
#define DEFINE_COPY_SFA_V2_KERNEL(name, dtype, splitHeads)                                                            \
    __global__ __aicore__ void name(                                                                                  \
        GM_ADDR queryNope, GM_ADDR queryRope, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes, GM_ADDR hbmBlockTable,         \
        GM_ADDR dramBlockTable, GM_ADDR topkDstSlots, GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,                     \
        GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv, GM_ADDR numCacheTokens, GM_ADDR missSrcIds,        \
        GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR current,          \
        GM_ADDR positions, GM_ADDR currentValid, GM_ADDR requestSlots, GM_ADDR reqToToken, GM_ADDR residentEnd,       \
        GM_ADDR workspace, GM_ADDR tiling)                                                                            \
    {                                                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                            \
        RunQuantCopySfaV2<dtype, splitHeads>(queryNope, queryRope, hbmKvBytes, dramKvBytes, hbmBlockTable,            \
                                             dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts,                \
                                             actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds,   \
                                             missDstSlots, missCounts, hbmKvAlias, attentionOut, current, positions,  \
                                             currentValid, requestSlots, reqToToken, residentEnd, workspace, tiling); \
    }

DEFINE_COPY_SFA_V2_KERNEL(fused_quant_copy_sfa_v2_bf16, bfloat16_t, false)
DEFINE_COPY_SFA_V2_KERNEL(fused_quant_copy_sfa_v2_bf16_split, bfloat16_t, true)
DEFINE_COPY_SFA_V2_KERNEL(fused_quant_copy_sfa_v2_fp16, half, false)
DEFINE_COPY_SFA_V2_KERNEL(fused_quant_copy_sfa_v2_fp16_split, half, true)

#undef DEFINE_COPY_SFA_V2_KERNEL
#endif
