/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv.cpp
 * \brief
 */

#include "kernel_operator.h"

// BiSheng also parses this file for the host launch stub. MicroAPI is only
// available in the AI Core pass, so keep the implementation headers there.
#ifdef __CCE_AICORE__
#include "lib/matmul_intf.h"
#include "kv_quant_sparse_attn_sharedkv_tiling_data.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h"
#include "kv_quant_sparse_attn_sharedkv_common.h"

using namespace AscendC;

#if !defined(__DAV_C310_CUBE__)
// The vector-side implementation keeps a pointer to tiling data in local
// memory. Copy scalar fields from GM, as in the other A5 kernels, rather than
// passing a __gm__ pointer to a function expecting a local pointer.
__aicore__ inline void LoadKvQuantTilingData(optiling::KvQuantSparseAttnSharedkvTilingData &local,
                                             __gm__ uint8_t *tiling)
{
    const auto *gm = reinterpret_cast<const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *>(tiling);
    local.dispatchKey = gm->dispatchKey;
    local.baseParams.batchSize = gm->baseParams.batchSize;
    local.baseParams.qSeqSize = gm->baseParams.qSeqSize;
    local.baseParams.kvSeqSize = gm->baseParams.kvSeqSize;
    local.baseParams.paOriBlockSize = gm->baseParams.paOriBlockSize;
    local.baseParams.paCmpBlockSize = gm->baseParams.paCmpBlockSize;
    local.baseParams.oriMaxBlockNumPerBatch = gm->baseParams.oriMaxBlockNumPerBatch;
    local.baseParams.cmpMaxBlockNumPerBatch = gm->baseParams.cmpMaxBlockNumPerBatch;
    local.baseParams.nNumOfQInOneGroup = gm->baseParams.nNumOfQInOneGroup;
    local.baseParams.sparseBlockCount = gm->baseParams.sparseBlockCount;
    local.baseParams.softmaxScale = gm->baseParams.softmaxScale;
    local.baseParams.oriKvStride = gm->baseParams.oriKvStride;
    local.baseParams.cmpKvStride = gm->baseParams.cmpKvStride;
    local.baseParams.tileSize = gm->baseParams.tileSize;
    local.baseParams.ropeHeadDim = gm->baseParams.ropeHeadDim;
    local.baseParams.cmpRatio = gm->baseParams.cmpRatio;
    local.baseParams.oriMaskMode = gm->baseParams.oriMaskMode;
    local.baseParams.cmpMaskMode = gm->baseParams.cmpMaskMode;
    local.baseParams.oriWinLeft = gm->baseParams.oriWinLeft;
    local.baseParams.oriWinRight = gm->baseParams.oriWinRight;
    local.baseParams.sparseBlockSize = gm->baseParams.sparseBlockSize;
    local.baseParams.hasOriSparseIndices = gm->baseParams.hasOriSparseIndices;
    local.baseParams.oriSparseIndexWidth = gm->baseParams.oriSparseIndexWidth;
    local.baseParams.dSize = gm->baseParams.dSize;
    local.baseParams.dSizeVInput = gm->baseParams.dSizeVInput;
}
#endif

#if defined(__DAV_C310_CUBE__)
#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                                           \
    do {                                                                                                           \
        using CubeBlockType =                                                                                      \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SCFABlockCube<__VA_ARGS__>,             \
                                      BaseApi::SCFABlockCubeDummy<__VA_ARGS__>>::type;                             \
        using VecBlockType =                                                                                       \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SCFABlockVecDummy<__VA_ARGS__>,         \
                                      BaseApi::SCFABlockVec<__VA_ARGS__>>::type;                                   \
        templateClass<CubeBlockType, VecBlockType> op;                                                             \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \
                seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, nullptr, &tPipe);                        \
        op.Process();                                                                                              \
    } while (0)
#else
#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                                           \
    do {                                                                                                           \
        using CubeBlockType =                                                                                      \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SCFABlockCube<__VA_ARGS__>,             \
                                      BaseApi::SCFABlockCubeDummy<__VA_ARGS__>>::type;                             \
        using VecBlockType =                                                                                       \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::SCFABlockVecDummy<__VA_ARGS__>,         \
                                      BaseApi::SCFABlockVec<__VA_ARGS__>>::type;                                   \
        templateClass<CubeBlockType, VecBlockType> op;                                                             \
        tilingdataClass tilingDataIn;                                                                              \
        LoadKvQuantTilingData(tilingDataIn, tiling);                                                               \
        const tilingdataClass *tilingData = &tilingDataIn;                                                         \
        op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, \
                seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, tilingData, &tPipe);                     \
        op.Process();                                                                                              \
    } while (0)
#endif
#endif  // __CCE_AICORE__

extern "C" __global__ __aicore__ void kv_quant_sparse_attn_sharedkv(
    GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
    GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv, GM_ADDR cuSeqlensCmpKv,
    GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata, GM_ADDR attentionOut, GM_ADDR softmax_lse,
    GM_ADDR workspace, GM_ADDR tiling)
{
#ifdef __CCE_AICORE__
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // The source op uses compile-time tiling-key specialization. The sgl-kernel-npu
    // build exports one launch symbol, so select the same specializations here.
    // The key is the first field of the host/device shared tiling buffer.
    const auto *tilingData = reinterpret_cast<const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *>(tiling);
#define KV_QUANT_CASE(layout, mode, split)                                                                             \
    case ((layout << 8) | (mode << 4) | split):                                                                        \
        SAS_OP_IMPL(BaseApi::KvQuantSparseAttnSharedkvScfa, optiling::KvQuantSparseAttnSharedkvTilingData, bfloat16_t, \
                    fp8_e4m3fn_t, float, bfloat16_t, 0, true, static_cast<SAS_LAYOUT>(layout), SAS_LAYOUT::PA_ND,      \
                    static_cast<SASTemplateMode>(mode), split);                                                        \
        break
    switch (tilingData->dispatchKey) {
        KV_QUANT_CASE(0, 0, 0);
        KV_QUANT_CASE(0, 0, 1);
        KV_QUANT_CASE(0, 1, 0);
        KV_QUANT_CASE(0, 1, 1);
        KV_QUANT_CASE(0, 2, 0);
        KV_QUANT_CASE(0, 2, 1);
        KV_QUANT_CASE(1, 0, 0);
        KV_QUANT_CASE(1, 0, 1);
        KV_QUANT_CASE(1, 1, 0);
        KV_QUANT_CASE(1, 1, 1);
        KV_QUANT_CASE(1, 2, 0);
        KV_QUANT_CASE(1, 2, 1);
        default:
            break;
    }
#undef KV_QUANT_CASE
#endif  // __CCE_AICORE__
}
