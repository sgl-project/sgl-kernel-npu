/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "fused_copy_sfa_impl.h"

extern "C" __global__ __aicore__ void fused_scatter_copy_sparse_flash_attention_combined(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR sparseIndices, GM_ADDR cacheTokens, GM_ADDR hbmBlockTable,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv, GM_ADDR queryRope, GM_ADDR hbmKeyRope,
    GM_ADDR dramKeyRope, GM_ADDR dramKvCache, GM_ADDR dramBlockTable, GM_ADDR topkSourceIds, GM_ADDR topkMissCounts,
    GM_ADDR missSourceIds, GM_ADDR missDstSlots, GM_ADDR missCounts, GM_ADDR attentionOut, GM_ADDR current,
    GM_ADDR positions, GM_ADDR currentValid, GM_ADDR requestSlots, GM_ADDR reqToToken, GM_ADDR residentEnd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunCopySfa<true>(query, key, value, sparseIndices, cacheTokens, hbmBlockTable, actualSeqLengthsQuery,
                     actualSeqLengthsKv, queryRope, hbmKeyRope, dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds,
                     topkMissCounts, missSourceIds, missDstSlots, missCounts, attentionOut, current, positions,
                     currentValid, requestSlots, reqToToken, residentEnd, workspace, tiling);
}
