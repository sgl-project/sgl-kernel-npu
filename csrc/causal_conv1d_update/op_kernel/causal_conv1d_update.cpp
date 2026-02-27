/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update.cpp
 * \brief causal_conv1d_update kernel entry point
 */

#include "causal_conv1d_update.h"
#include "causal_conv1d_update_struct.h"

extern "C" __global__ __aicore__ void causal_conv1d_update_bfloat16_t(GM_ADDR x, GM_ADDR weight, GM_ADDR conv_state, GM_ADDR conv_state_indices,
                                                                   GM_ADDR bias, GM_ADDR num_accepted_tokens, GM_ADDR query_start_loc,
                                                                   GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dUpdateOp::CausalConv1dUpdate<bfloat16_t> op;
    op.Init(x, weight, conv_state, conv_state_indices, bias, num_accepted_tokens, query_start_loc, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void causal_conv1d_update_half(GM_ADDR x, GM_ADDR weight, GM_ADDR conv_state, GM_ADDR conv_state_indices,
                                                               GM_ADDR bias, GM_ADDR num_accepted_tokens, GM_ADDR query_start_loc,
                                                               GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dUpdateOp::CausalConv1dUpdate<half> op;
    op.Init(x, weight, conv_state, conv_state_indices, bias, num_accepted_tokens, query_start_loc, y, tiling);
    op.Process();
}
