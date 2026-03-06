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
#include "la_preprocess.h"


extern "C" __global__ __aicore__ void la_preprocess(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR out_query, GM_ADDR out_key, GM_ADDR out_value,
    GM_ADDR workspace_in, GM_ADDR tiling_in)
{
    auto *tiling_data = reinterpret_cast<__gm__ LaPreprocessTiling *>(tiling_in);
    if (TILING_KEY_IS(0)) {
        mmdit_ops::kernels::LaPreprocess<bfloat16_t, half, 128, 1> op;
        AscendC::TPipe pipe;
        op.Init(query, key, value, out_query, out_key, out_value, tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        mmdit_ops::kernels::LaPreprocess<half, half, 128, 1> op;
        AscendC::TPipe pipe;
        op.Init(query, key, value, out_query, out_key, out_value, tiling_data, &pipe);
        op.Process();
    }
}
