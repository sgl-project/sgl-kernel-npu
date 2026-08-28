/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_kernel.cpp
 * \brief
 *
 * 上游 (vllm-ascend csrc/attention/lightning_indexer) 使用 CANN 的
 * ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL 由框架为每个模板组合生成一个二进制，
 * 内核入口只需接收模板形参。sgl-kernel-npu 走 kernel-direct-launch，只有一个
 * __global__ 符号，因此这里把上游 ASCENDC_TPL_SEL 中列举的合法组合手工展开成
 * 一个 tilingKey 分发表。tilingKey 的编码与 host 侧 DoTiling 保持一致。
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "lightning_indexer_v2_common.h"
#include "lightning_indexer_v2_tiling_key.h"

#if (__CCE_AICORE__ == 310)
#include "arch35/lightning_indexer_kernel.h"
#else
#include "arch22/lightning_indexer_kernel.h"
#endif

namespace sglang::npu_kernel::liv2 {

// tilingData 由 host 以 GM buffer 传入。上游用 GET_TILING_DATA_WITH_STRUCT 把
// GM 拷到本地再传给 kernel，这里做同样的事，从而 arch22/arch35 的 Init 签名
// (const LITilingData *) 可以原样复用。
__aicore__ inline void LoadTilingData(LITilingData &dst, __gm__ uint8_t *src)
{
    auto *__restrict s = reinterpret_cast<__gm__ LITilingData *>(src);
    dst.preTokens = s->preTokens;
    dst.nextTokens = s->nextTokens;
    dst.bSize = s->bSize;
    dst.n2Size = s->n2Size;
    dst.gSize = s->gSize;
    dst.s1Size = s->s1Size;
    dst.s2Size = s->s2Size;
    dst.sparseCount = s->sparseCount;
    dst.usedCoreNum = s->usedCoreNum;
    dst.blockSize = s->blockSize;
    dst.maxBlockNumPerBatch = s->maxBlockNumPerBatch;
    dst.sparseMode = s->sparseMode;
    dst.returnValue = s->returnValue;
    dst.tilingKey = s->tilingKey;
}

}  // namespace sglang::npu_kernel::liv2

#define INVOKE_LI_V2_OP_IMPL(...)                                                                                  \
    do {                                                                                                           \
        LIKernel::LightningIndexerKernel<LICommon::LIType<__VA_ARGS__>> op;                                        \
        op.Init(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, sparseIndices, sparseValues, \
                userWorkspace, &tilingData, &tPipe);                                                               \
        op.Process();                                                                                              \
    } while (0)

// 一条 dispatch 分支。Q_T/K_T 由 dtype 决定，其余为 LIType 的非类型模板实参。
#define LI_V2_CASE(KEY_Q, KEY_K, Q_T, K_T, PA, Q_LAYOUT, K_LAYOUT, W_FP32)                                        \
    case SGL_LI_V2_TILING_KEY(KEY_Q, KEY_K, SGL_LI_V2_DT_INT32, W_FP32, PA, SGL_LI_V2_LAYOUT_##Q_LAYOUT,          \
                              SGL_LI_V2_LAYOUT_##K_LAYOUT):                                                       \
        INVOKE_LI_V2_OP_IMPL(Q_T, K_T, int32_t, PA, LICommon::LI_LAYOUT::Q_LAYOUT, LICommon::LI_LAYOUT::K_LAYOUT,  \
                             W_FP32);                                                                             \
        break;

// 上游 ASCENDC_TPL_SEL 允许的 (PAGE_ATTENTION, LAYOUT_T, K_LAYOUT_T) 组合：
// 非 PA 场景要求 layout_query == layout_key，故上游共 4 种布局组合（PA 2 种 +
// 非 PA 2 种）x 2 dtype x 2 weights 档 = 16 份实例，全部编进同一个 kernel 二进制。
#define LI_V2_NON_PA_COMBOS(KEY_D, T, W_FP32)             \
    LI_V2_CASE(KEY_D, KEY_D, T, T, 0, BSND, BSND, W_FP32) \
    LI_V2_CASE(KEY_D, KEY_D, T, T, 0, TND, TND, W_FP32)

#define LI_V2_LAYOUT_COMBOS(KEY_D, T, W_FP32)                \
    LI_V2_CASE(KEY_D, KEY_D, T, T, 1, BSND, PA_BSND, W_FP32) \
    LI_V2_CASE(KEY_D, KEY_D, T, T, 1, TND, PA_BSND, W_FP32)  \
    LI_V2_NON_PA_COMBOS(KEY_D, T, W_FP32)

#define LI_V2_DTYPE_COMBOS(KEY_D, T)  \
    LI_V2_LAYOUT_COMBOS(KEY_D, T, 0)  \
    LI_V2_LAYOUT_COMBOS(KEY_D, T, 1)

extern "C" __global__ __aicore__ void lightning_indexer_v2(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
    GM_ADDR blocktable, GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace sglang::npu_kernel::liv2;

    AscendC::TPipe tPipe;
    __gm__ uint8_t *userWorkspace = workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    LITilingData tilingData;
    LoadTilingData(tilingData, tiling);

    switch (tilingData.tilingKey) {
        LI_V2_DTYPE_COMBOS(SGL_LI_V2_DT_FP16, half)
        LI_V2_DTYPE_COMBOS(SGL_LI_V2_DT_BF16, bfloat16_t)
        default:
            break;
    }
}
