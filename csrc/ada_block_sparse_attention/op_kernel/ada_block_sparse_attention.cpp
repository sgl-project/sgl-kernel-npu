/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "kernel_operator.h"
#include "ada_block_sparse_attention_tiling_data.h"
#include "ada_block_sparse_attention_s1s2_bns1_x910.h"

using namespace AscendC;

template <typename Type>
__aicore__ inline void RunAdaBlockSparseAttention(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR actualSeqLengths,
                                                  GM_ADDR actualSeqLengthsKV, GM_ADDR sparseMask,
                                                  GM_ADDR sparseCntTable, GM_ADDR attentionOut, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    AdaBlockSparseAttentionTilingData tilingData;
    auto dst = reinterpret_cast<uint32_t *>(&tilingData);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0; i < sizeof(tilingData) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    TPipe pipe;
    AdaBlockSparseAttentionS1s2Bns1X910<Type> op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.bmm1TilingDataRect, op.bmm2,
                      &tilingData.bmm2TilingDataRect);
    op.Init(query, key, value, sparseMask, sparseCntTable, nullptr, nullptr, actualSeqLengths, actualSeqLengthsKV,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, attentionOut, nullptr, workspace,
            &tilingData, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void ada_block_sparse_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                 GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKV,
                                                                 GM_ADDR sparseMask, GM_ADDR sparseCntTable,
                                                                 GM_ADDR attentionOut, uint64_t tilingKey,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // HAVE_WORKSPACE + HAVE_TILING requires workspace and tiling to be the
    // final two arguments. The generated wrapper initializes system workspace
    // and passes the user workspace to this entry point.
#define RUN_ADA(...)                                                                                             \
    RunAdaBlockSparseAttention<__VA_ARGS__>(query, key, value, actualSeqLengths, actualSeqLengthsKV, sparseMask, \
                                            sparseCntTable, attentionOut, workspace, tiling)
    switch (tilingKey) {
        case 1000000000000101012ULL:
        case 1000000000002101012ULL:
            RUN_ADA(BSAType<BSALayout::BSH, half, bool>);
            break;
        case 1000000000000001012ULL:
        case 1000000000002001012ULL:
            RUN_ADA(BSAType<BSALayout::BNSD, half, uint8_t>);
            break;
        case 1000000000000111112ULL:
        case 1000000000002111112ULL:
            RUN_ADA(BSAType<BSALayout::BSH, bfloat16_t, bool, bfloat16_t>);
            break;
        case 1000000000000011112ULL:
        case 1000000000002011112ULL:
            RUN_ADA(BSAType<BSALayout::BNSD, bfloat16_t, bool, bfloat16_t>);
            break;
    }
#undef RUN_ADA
}
