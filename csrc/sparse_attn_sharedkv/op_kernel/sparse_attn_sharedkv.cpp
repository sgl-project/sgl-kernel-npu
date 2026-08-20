/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch32/sparse_attn_sharedkv_scfa_kernel.h"
#include "arch32/sparse_attn_sharedkv_swa_kernel.h"
#include "sparse_attn_sharedkv_metadata.h"

using namespace AscendC;
using namespace SASKernel;

template <typename DTYPE, SAS_LAYOUT Q_LAYOUT, int MODE>
__aicore__ inline void LaunchSparseAttnSharedkv(GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices,
                                                GM_ADDR cmpSparseIndices, GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable,
                                                GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv, GM_ADDR cuSeqlensCmpKv,
                                                GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,
                                                GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR userWorkspace,
                                                GM_ADDR tiling, TPipe *pipe)
{
    using Type = SASType<DTYPE, DTYPE, DTYPE, false, Q_LAYOUT, SAS_LAYOUT::PA_ND, MODE>;
    auto tilingData = reinterpret_cast<const __gm__ optiling::SparseAttnSharedkvTilingData *>(tiling);
    if constexpr (MODE == SCFA_TEMPLATE) {
        SparseAttnSharedkvScfa<Type> op;
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, softmaxLse,
                userWorkspace, tilingData, tiling, pipe);
        op.Process();
    } else {
        SparseAttnSharedkvSwa<Type> op;
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,
                cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, softmaxLse,
                userWorkspace, tilingData, tiling, pipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void sparse_attn_sharedkv(GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV,
                                                           GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
                                                           GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable,
                                                           GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv,
                                                           GM_ADDR cuSeqlensCmpKv, GM_ADDR seqUsedQ, GM_ADDR seqUsedKV,
                                                           GM_ADDR sinks, GM_ADDR metadata, GM_ADDR attentionOut,
                                                           GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    auto tilingData = reinterpret_cast<const __gm__ optiling::SparseAttnSharedkvTilingData *>(tiling);
    __gm__ uint8_t *userWorkspace = workspace;
    if (workspace != nullptr) {
        userWorkspace = GetUserWorkspace(workspace);
    }

#define SAS_LAUNCH(dtype, qlayout, mode)                                                                               \
    LaunchSparseAttnSharedkv<dtype, qlayout, mode>(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices,            \
                                                   oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv,           \
                                                   cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, \
                                                   softmaxLse, userWorkspace, tiling, &pipe)

    switch (tilingData->dispatchKey) {
        case 0x00000200:
            SAS_LAUNCH(half, SAS_LAYOUT::BSND, SWA_TEMPLATE);
            break;
        case 0x00000201:
            SAS_LAUNCH(half, SAS_LAYOUT::BSND, CFA_TEMPLATE);
            break;
        case 0x00000202:
            SAS_LAUNCH(half, SAS_LAYOUT::BSND, SCFA_TEMPLATE);
            break;
        case 0x00010200:
            SAS_LAUNCH(half, SAS_LAYOUT::TND, SWA_TEMPLATE);
            break;
        case 0x00010201:
            SAS_LAUNCH(half, SAS_LAYOUT::TND, CFA_TEMPLATE);
            break;
        case 0x00010202:
            SAS_LAUNCH(half, SAS_LAYOUT::TND, SCFA_TEMPLATE);
            break;
        case 0x01000200:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::BSND, SWA_TEMPLATE);
            break;
        case 0x01000201:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::BSND, CFA_TEMPLATE);
            break;
        case 0x01000202:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::BSND, SCFA_TEMPLATE);
            break;
        case 0x01010200:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::TND, SWA_TEMPLATE);
            break;
        case 0x01010201:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::TND, CFA_TEMPLATE);
            break;
        case 0x01010202:
            SAS_LAUNCH(bfloat16_t, SAS_LAYOUT::TND, SCFA_TEMPLATE);
            break;
    }
#undef SAS_LAUNCH
}
