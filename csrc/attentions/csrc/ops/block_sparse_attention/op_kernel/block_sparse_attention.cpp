/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "kernel_operator.h"
#include "block_sparse_attention_s1s2_bns1_x910.h"
#include "block_sparse_attention_empty_tensor.h"

#define INVOKE_BSA_GENERAL_OP_IMPL(templateClass, ...)                                                                \
    TPipe tPipe;                                                                                                      \
    do {                                                                                                              \
        if (query == nullptr) {return;}                                                                               \
        INVOKE_BSA_TILING_DATA(tiling);                                                                               \
        templateClass<__VA_ARGS__> op;                                                                                \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, bmm1tiling, op.bmm2, bmm2tiling);                      \
        op.Init(query, key, value, sparseMask, sparseCntTable, pseShift, attenMask,                                   \
                actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,                                   \
                kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,                             \
                attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                                         \
        op.Process();                                                                                                 \
    } while (0)

#define INVOKE_BSA_TILING_DATA(tiling)                                                                                \
    GET_TILING_DATA_WITH_STRUCT(BlockSparseAttentionTilingData, tiling_data_in, tiling);                              \
    const BlockSparseAttentionTilingData* __restrict tiling_data = &tiling_data_in;                                   \
    const TCubeTiling* __restrict bmm1tiling = &(tiling_data->bmm1TilingDataRect);                                    \
    const TCubeTiling* __restrict bmm2tiling = &(tiling_data->bmm2TilingDataRect)

extern "C" __global__ __aicore__ void block_sparse_attention_FIAS(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                                            __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                                            __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
                                                            __gm__ uint8_t* actualSeqLengthsKV,
                                                            __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                            __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                            __gm__ uint8_t* quant_offset2,
                                                            __gm__ uint8_t* antiquant_scale,
                                                            __gm__ uint8_t* antiquant_offset,
                                                            __gm__ uint8_t* blocktable,
                                                            __gm__ uint8_t* queryPaddingSize,
                                                            __gm__ uint8_t* kvPaddingSize,
                                                            __gm__ uint8_t* key_antiquant_scale,
                                                            __gm__ uint8_t* key_antiquant_offset,
                                                            __gm__ uint8_t* value_antiquant_scale,
                                                            __gm__ uint8_t* value_antiquant_offset,
                                                            __gm__ uint8_t* keySharedPrefix,
                                                            __gm__ uint8_t* valueSharedPrefix,
                                                            __gm__ uint8_t* actualSharedPrefixLen,
                                                            __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope,
                                                            __gm__ uint8_t * sparseMask,
                                                            __gm__ uint8_t* sparseCntTable,
                                                            __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                            __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    GET_TILING_DATA_MEMBER(BlockSparseAttentionTilingData, promptAttentionBaseParams, baseParams, tiling);

    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // template <BSALayout L, typename T, typename U, typename O = T, typename KV_T = T, Mode M = Mode::HighPerformance,
    //  const MatMulType MM_TYPE_TMP = MatMulType::MM_SP, const bool F = false,
    // const MsdMode MSD_MODE = MsdMode::MSD_OFF, typename...Args>

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_KEY != DT_INT4)
    TILING_KEY_IS(1000000000000101012);
    TILING_KEY_IS(1000000000000001012);
    TILING_KEY_IS(1000000000002101012);
    TILING_KEY_IS(1000000000002001012);
    
    #if TILING_KEY_VAR == 1000000000000101012 || TILING_KEY_VAR == 1000000000002101012
        INVOKE_BSA_GENERAL_OP_IMPL(BlockSparseAttentionS1s2Bns1X910, BSAType<BSALayout::BSH, half, bool>);
    #elif TILING_KEY_VAR == 1000000000000001012 || TILING_KEY_VAR == 1000000000002001012
        INVOKE_BSA_GENERAL_OP_IMPL(BlockSparseAttentionS1s2Bns1X910, BSAType<BSALayout::BNSD, half, uint8_t>);
    #endif
#endif

#if (ORIG_DTYPE_QUERY == DT_BF16) && (ORIG_DTYPE_KEY != DT_INT4) && (ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    TILING_KEY_IS(1000000000000111112);
    TILING_KEY_IS(1000000000000011112);
    TILING_KEY_IS(1000000000002111112);
    TILING_KEY_IS(1000000000002011112);
    #if TILING_KEY_VAR == 1000000000000111112 || TILING_KEY_VAR == 1000000000002111112
        // BSH layout bf16 cvdiff
        INVOKE_BSA_GENERAL_OP_IMPL(BlockSparseAttentionS1s2Bns1X910, BSAType<BSALayout::BSH, bfloat16_t,
            bool, bfloat16_t>);
    #elif TILING_KEY_VAR == 1000000000000011112 || TILING_KEY_VAR == 1000000000002011112
        // BNSD layout bf16 cvdiff
        INVOKE_BSA_GENERAL_OP_IMPL(BlockSparseAttentionS1s2Bns1X910, BSAType<BSALayout::BNSD, bfloat16_t,
            bool, bfloat16_t>);
    #endif

#endif
}

extern "C" __global__ __aicore__ void block_sparse_attention(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                                             __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                                             __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths,
                                                             __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2,
                                                             __gm__ uint8_t* sparseMask, __gm__ uint8_t* sparseCntTable,
                                                             __gm__ uint8_t* attentionOut,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    block_sparse_attention_FIAS(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV,
        deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        sparseMask, sparseCntTable, attentionOut, nullptr, workspace, tiling);
}