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
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)hbmKvAlias;
    const auto *gm = reinterpret_cast<const __gm__ FusedQuantScatterCopySparseFlashAttentionTilingData *>(tiling);
    FusedQuantScatterCopySparseFlashAttentionTilingData tilingData{{gm->baseParams.batchSize, gm->baseParams.seqSize, gm->baseParams.qSeqSize, gm->baseParams.nNumOfQInOneGroup, gm->baseParams.sparseBlockCount, gm->baseParams.sparseMode, gm->baseParams.outputLayout, gm->baseParams.blockSize, gm->baseParams.maxBlockNumPerBatch, gm->baseParams.isActualLenDimsNull, gm->baseParams.isActualLenDimsKVNull, gm->baseParams.dSizeVInput, gm->baseParams.scaleValue}, {gm->singleCoreParams.usedCoreNum}, gm->dramMaxBlocks, gm->hbmPhysicalBlocks, gm->dramPhysicalBlocks, {0}};
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const bool fallback = fused_quant_scatter_copy_sfa::RequiresUnionCopy(missCounts, topkMissCounts,
        actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, tilingData);
    if (fallback) {
        fused_quant_scatter_copy_sfa::UnionCopy(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable,
            dramBlockTable, missSrcIds, missDstSlots, missCounts, tilingData);
    }
    using Cube = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAMatmulService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAMatmulServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    using Vector = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAVectorServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAVectorService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    BaseApi::FusedQuantScatterCopySparseFlashAttentionMla<Cube, Vector> op;
    op.Init(query, hbmKvBytes, hbmKvBytes, topkDstSlots, nullptr, nullptr, hbmBlockTable,
        actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, attentionOut,
        userWorkspace, &tilingData, &pipe);
    op.InitSourceAwareGather(dramKvBytes, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, &tilingData,
        !fallback, userWorkspace + fused_quant_scatter_copy_sfa::AttentionWorkspaceBytes(
            tilingData.singleCoreParams.usedCoreNum));
    op.Process();
}
#endif
__global__ __aicore__ void fused_quant_copy_sfa_bf16(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<bfloat16_t, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_bf16_split(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<bfloat16_t, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_fp16(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<half, false>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}
__global__ __aicore__ void fused_quant_copy_sfa_fp16_split(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunQuantCopySfa<half, true>(query, hbmKvBytes, dramKvBytes, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots, missCounts, hbmKvAlias, attentionOut, workspace, tiling);
#endif
}
