/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 1.0.
 */
#define SGLANG_M3_SWIGLU_OAI

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_tiling.h"
#include "dispatch_ffn_combine_m3_kernel.h"

using namespace AscendC;
using namespace DispatchFFNCombineM3Impl;

extern "C" __global__ __aicore__ void dispatch_ffn_combine_m3(
    GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expert_idx, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR probs,
    GM_ADDR out, GM_ADDR expert_token_nums, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineTilingData, tiling_data, tiling);
        DispatchFFNCombineM3<DTYPE_W1, DTYPE_OUT> op;
        op.Init(x, w1, w2, expert_idx, scale1, scale2, probs, out, expert_token_nums, workspace, tiling);
        op.Process();
    }
}
