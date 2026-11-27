// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/fused_quant_lightning_indexer_manage_impl.h"
#endif

extern "C" __global__ __aicore__ void fused_quant_lightning_indexer_manage_full_visible(
    GM_ADDR weights, GM_ADDR queryScale, GM_ADDR query, GM_ADDR keyScale, GM_ADDR key, GM_ADDR blockTable,
    GM_ADDR queryEnds, GM_ADDR lengths, GM_ADDR cacheTokens, GM_ADDR requestState, GM_ADDR poolEntries,
    GM_ADDR slotPool, GM_ADDR topkSources, GM_ADDR topkSlots, GM_ADDR topkMissCounts, GM_ADDR missSources,
    GM_ADDR missSlots, GM_ADDR missCounts, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunFusedQuantLightningIndexerManageArch35(weights, queryScale, query, keyScale, key, blockTable, queryEnds, lengths,
                                              cacheTokens, requestState, poolEntries, slotPool, topkSources, topkSlots,
                                              topkMissCounts, missSources, missSlots, missCounts, workspace, tiling);
#endif
}
