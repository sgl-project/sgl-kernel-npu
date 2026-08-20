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
 * \file compressor.cpp
 * \brief single named kernel entry for the Compressor op (ge_helper / direct-launch).
 */

// EVENT_ID constants used by the cube kernel. In the top-level ge_helper build
// these are not provided by the msopgen toolchain, so define them here.
#ifndef EVENT_ID0
#define EVENT_ID0 0
#endif
#ifndef EVENT_ID1
#define EVENT_ID1 1
#endif
#ifndef EVENT_ID2
#define EVENT_ID2 2
#endif
#ifndef EVENT_ID3
#define EVENT_ID3 3
#endif
#ifndef EVENT_ID4
#define EVENT_ID4 4
#endif
#ifndef EVENT_ID5
#define EVENT_ID5 5
#endif
#ifndef EVENT_ID6
#define EVENT_ID6 6
#endif
#ifndef EVENT_ID7
#define EVENT_ID7 7
#endif

#include "compressor_kernel.h"
#include "compressor_kernel_perf.h"
#include "compressor_template_tiling_key.h"

using namespace Compressor;

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...)                                                      \
    do {                                                                                                           \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData);                                                \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed, \
                startPos, cmpKvOut, workspace);                                                                    \
        op.Process();                                                                                              \
    } while (0)

#define LAUNCH_COMPRESSOR_KEY(LAYOUT_BIT, DTYPE_BIT, COFF_VAL, ROT_VAL, CACHE_VAL)                                \
    case GET_TPL_TILING_KEY(LAYOUT_BIT, DTYPE_BIT, COFF_VAL, ROT_VAL, CACHE_VAL, 2):                              \
        INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, static_cast<X_LAYOUT>(LAYOUT_BIT),                \
                                          static_cast<X_DTYPE>(DTYPE_BIT), static_cast<COFF>(COFF_VAL),           \
                                          static_cast<ROTARY_MODE>(ROT_VAL), static_cast<CACHE_MODE>(CACHE_VAL)); \
        break;

extern "C" __global__ __aicore__ void compressor(GM_ADDR x, GM_ADDR wKv, GM_ADDR wGate, GM_ADDR stateCache, GM_ADDR ape,
                                                 GM_ADDR normWeight, GM_ADDR ropeSin, GM_ADDR ropeCos,
                                                 GM_ADDR stateBlockTable, GM_ADDR cuSeqlens, GM_ADDR seqUsed,
                                                 GM_ADDR startPos, GM_ADDR cmpKvOut, GM_ADDR stateCacheOut,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    auto tilingData = reinterpret_cast<__gm__ optiling::CompressorTilingData *>(tiling);
    uint64_t key = tilingData->tilingKey;
    // TEMPLATE_ID bits 11-12: 1 = EMPTY_X (nothing to do)
    if (((key >> 11) & 0x3) == 1) {
        return;
    }
    switch (key) {
        // TH layout (layout bit = 1)
        LAUNCH_COMPRESSOR_KEY(1, 0, 1, 2, 2)  // TH bf16 coff1 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(1, 0, 2, 2, 2)  // TH bf16 coff2 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(1, 0, 1, 2, 1)  // TH bf16 coff1 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(1, 0, 2, 2, 1)  // TH bf16 coff2 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(1, 1, 1, 2, 1)  // TH fp16 coff1 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(1, 1, 2, 2, 1)  // TH fp16 coff2 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(1, 1, 1, 2, 2)  // TH fp16 coff1 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(1, 1, 2, 2, 2)  // TH fp16 coff2 rot2 cache2
        // BSH layout (layout bit = 0)
        LAUNCH_COMPRESSOR_KEY(0, 0, 1, 2, 1)  // BSH bf16 coff1 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(0, 0, 2, 2, 1)  // BSH bf16 coff2 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(0, 0, 1, 2, 2)  // BSH bf16 coff1 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(0, 0, 2, 2, 2)  // BSH bf16 coff2 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(0, 1, 1, 2, 1)  // BSH fp16 coff1 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(0, 1, 2, 2, 1)  // BSH fp16 coff2 rot2 cache1
        LAUNCH_COMPRESSOR_KEY(0, 1, 1, 2, 2)  // BSH fp16 coff1 rot2 cache2
        LAUNCH_COMPRESSOR_KEY(0, 1, 2, 2, 2)  // BSH fp16 coff2 rot2 cache2
        default:
            break;
    }
}

#undef LAUNCH_COMPRESSOR_KEY
#undef INVOKE_COMPRESSOR_GENERAL_OP_IMPL
