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
#include "sparse_block_estimate_tiling_data.h"
#include "sparse_block_estimate.h"

using namespace AscendC;

template <typename Type>
__aicore__ inline void RunSparseBlockEstimate(GM_ADDR q, GM_ADDR k, GM_ADDR actualSeqLengths,
                                              GM_ADDR actualSeqLengthsKV, GM_ADDR sparseMask, GM_ADDR sparseCountTable,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    SparseBlockEstimateTilingData tilingData;
    auto dst = reinterpret_cast<uint32_t *>(&tilingData);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0; i < sizeof(tilingData) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    SparseBlockEstimate<Type> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.cubeTilingData);
    op.Init(q, k, actualSeqLengths, actualSeqLengthsKV, sparseMask, sparseCountTable, GetUserWorkspace(workspace),
            tilingData);
    op.InitBuffers();
    op.Process();
}

extern "C" __global__ __aicore__ void sparse_block_estimate(GM_ADDR q, GM_ADDR k, GM_ADDR actualSeqLengths,
                                                            GM_ADDR actualSeqLengthsKV, GM_ADDR sparseMask,
                                                            GM_ADDR sparseCountTable, GM_ADDR workspace, GM_ADDR tiling,
                                                            uint64_t tilingKey)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSysWorkspace(workspace);
#define RUN_ESTIMATE(...)                                                                                         \
    RunSparseBlockEstimate<__VA_ARGS__>(q, k, actualSeqLengths, actualSeqLengthsKV, sparseMask, sparseCountTable, \
                                        workspace, tiling)
    switch (tilingKey) {
        case 1000000000000000000ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BNSD, half, false>);
            break;
        case 1000000000000000001ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BNSD, half, true>);
            break;
        case 1000000000000000010ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BSH, half, false>);
            break;
        case 1000000000000000011ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BSH, half, true>);
            break;
        case 1000000000000000020ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::TND, half, false>);
            break;
        case 1000000000000000021ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::TND, half, true>);
            break;
        case 1000000000000000100ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BNSD, bfloat16_t, false>);
            break;
        case 1000000000000000101ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BNSD, bfloat16_t, true>);
            break;
        case 1000000000000000110ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BSH, bfloat16_t, false>);
            break;
        case 1000000000000000111ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::BSH, bfloat16_t, true>);
            break;
        case 1000000000000000120ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::TND, bfloat16_t, false>);
            break;
        case 1000000000000000121ULL:
            RUN_ESTIMATE(INVOKE_TYPE<INPUT_LAYOUT::TND, bfloat16_t, true>);
            break;
    }
#undef RUN_ESTIMATE
}
