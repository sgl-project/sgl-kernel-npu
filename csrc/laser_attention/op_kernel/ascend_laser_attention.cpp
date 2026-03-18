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
#include "lib/matmul_intf.h"
#include "CubeForward.h"
#include "VectorForward.h"
#include "la_preprocess.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void ascend_laser_attention(
    GM_ADDR q_gm,
    GM_ADDR k_gm,
    GM_ADDR v_gm,
    GM_ADDR atten_mask_gm,
    GM_ADDR alibi_mask_gm, // ???
    GM_ADDR drop_mask_gm, // ？？？
    GM_ADDR softmax_log_max_sum_gm,
    GM_ADDR attention_out_gm,
    GM_ADDR workspace_in,
    GM_ADDR tiling_in)
{
    auto tiling_data = reinterpret_cast<__gm__ AscendLaserAttentionTilingData *>(tiling_in);
    SetSysWorkspace(workspace_in);
    GM_ADDR user = GetUserWorkspace(workspace_in);

    int32_t y = tiling_data->coreNumPerGroup;
    int32_t f = tiling_data->coreGroupNum;
    int32_t b = tiling_data->batchSize;
    int32_t n = tiling_data->headNum;
    int32_t s1 = tiling_data->qSeqLength;
    int32_t s2 = tiling_data->kSeqLength;
    int32_t d = tiling_data->headDim;
    int32_t g = tiling_data->headGroupSize;
    int32_t qkTriangle = tiling_data->isTriangle; // 需要换成bool值
    int32_t sparseMode = tiling_data->sparseMode; // sparseMode: 0:dense, 1:sparse
    int32_t windowLen = tiling_data->windowLen; // sparse场景下，滑动窗口的长度
    bool isHighPrecision = true;
    int32_t maskSeqLength = tiling_data->maskSeqLength; // attention mask 的 length
    float scale = tiling_data->scale;
    auto aicNum = y * f;

    // 尽量限制在192kb以下
    __gm__ float * __restrict__ gm_attention_out = (__gm__ float *__restrict__)attention_out_gm; // cube最后的输出
    __gm__ float* __restrict__ softmax_log_max_sum = (__gm__ float *__restrict__)softmax_log_max_sum_gm; // cube最后的输出

    // 第i个core的rowsum_diag地址 = 起始地址 + core_index * 128 *128* 2 * 2
    __gm__ float * __restrict__ gm_rowmax_diag = (__gm__ float *__restrict__)user;  // 128*128*2 * 2 * aicNum;
    __gm__ float * __restrict__ gm_rowsum_diag =
        (__gm__ float *__restrict__)(gm_rowmax_diag + 256 * 128 * 2 * MAX_SWITCH_TIME * aicNum);
    __gm__ uint8_t * __restrict__ score_gm =
        (__gm__ uint8_t *__restrict__)(gm_rowsum_diag + 256 * 128 * 2 * MAX_SWITCH_TIME* aicNum);

#ifdef __DAV_C220_CUBE__
    CUBE_FORWARD_ONLINE::CubeForward<half, false, half> op;
    op.Init(q_gm, k_gm, v_gm, score_gm, gm_attention_out, gm_rowsum_diag, gm_rowmax_diag, softmax_log_max_sum,
        y, f, b, n, s1, s2, d, g, qkTriangle, sparseMode, windowLen);
    op.Run();
#elif __DAV_C220_VEC__
    VectorForward<half, false, half> op;
    op.Init(q_gm, k_gm, v_gm, atten_mask_gm, score_gm, gm_attention_out, softmax_log_max_sum, gm_rowsum_diag,
        gm_rowmax_diag, s1, s2, n, b, y, qkTriangle, windowLen / BASE_BLOCK_SIDE_LEN, maskSeqLength, scale, windowLen);
    op.SetHighPrecision(true);
    op.Run();
#endif
}
