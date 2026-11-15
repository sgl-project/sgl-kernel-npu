/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch32/sparse_attn_sharedkv_tree_scfa_kernel.h"
#include "arch32/sparse_attn_sharedkv_tree_swa_kernel.h"
#include "sparse_attn_sharedkv_tree_metadata.h"

using namespace AscendC;
using namespace SASTreeKernel;
using SparseAttnSharedkvTreeTilingData = optiling::SparseAttnSharedkvTreeTilingData;

template <typename DTYPE, int TEMPLATE_MODE>
__aicore__ inline void LaunchTreeAttention(GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV,
                                           GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
                                           GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable,
                                           GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv, GM_ADDR cuSeqlensCmpKv,
                                           GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,
                                           GM_ADDR cmpTreeKv, GM_ADDR treeAncestors, GM_ADDR treePrefix,
                                           GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR userWorkspace,
                                           GM_ADDR tiling, TPipe *pipe)
{
    using Type = SASTreeType<DTYPE, DTYPE, DTYPE, false, SASTree_LAYOUT::TND,
                             SASTree_LAYOUT::PA_ND, TEMPLATE_MODE>;
    const __gm__ SparseAttnSharedkvTreeTilingData *tilingData =
        reinterpret_cast<const __gm__ SparseAttnSharedkvTreeTilingData *>(tiling);
    if constexpr (TEMPLATE_MODE == SCFA_TEMPLATE) {
        SparseAttnSharedkvTreeScfa<Type> op;
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata,
                cmpTreeKv, treeAncestors, treePrefix, attentionOut, softmaxLse, userWorkspace,
                tilingData, tiling, pipe);
        op.Process();
    } else {
        SparseAttnSharedkvTreeSwa<Type> op;
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata,
                cmpTreeKv, treeAncestors, treePrefix, attentionOut, softmaxLse, userWorkspace,
                tilingData, tiling, pipe);
        op.Process();
    }
}

#define SASTREE_ENTRY(NAME, DTYPE, MODE)                                                                               \
    extern "C" __global__ __aicore__ void NAME(                                                                        \
        GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,               \
        GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv,                      \
        GM_ADDR cuSeqlensCmpKv, GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,                  \
        GM_ADDR cmpTreeKv, GM_ADDR treeAncestors, GM_ADDR treePrefix, GM_ADDR attentionOut,                           \
        GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)                                                        \
    {                                                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                            \
        TPipe pipe;                                                                                                    \
        __gm__ uint8_t *userWorkspace = workspace;                                                                     \
        if (workspace != nullptr) {                                                                                    \
            userWorkspace = GetUserWorkspace(workspace);                                                               \
        }                                                                                                              \
        LaunchTreeAttention<DTYPE, MODE>(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable,        \
                                         cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ,          \
                                         seqUsedKV, sinks, metadata, cmpTreeKv, treeAncestors, treePrefix,            \
                                         attentionOut, softmaxLse, userWorkspace, tiling, &pipe);                      \
    }

SASTREE_ENTRY(sparse_attn_sharedkv_tree_scfa_fp16, half, SCFA_TEMPLATE)
SASTREE_ENTRY(sparse_attn_sharedkv_tree_scfa_bf16, bfloat16_t, SCFA_TEMPLATE)
SASTREE_ENTRY(sparse_attn_sharedkv_tree_cfa_fp16, half, CFA_TEMPLATE)
SASTREE_ENTRY(sparse_attn_sharedkv_tree_cfa_bf16, bfloat16_t, CFA_TEMPLATE)
SASTREE_ENTRY(sparse_attn_sharedkv_tree_swa_fp16, half, SWA_TEMPLATE)
SASTREE_ENTRY(sparse_attn_sharedkv_tree_swa_bf16, bfloat16_t, SWA_TEMPLATE)
#undef SASTREE_ENTRY
