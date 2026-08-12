/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*!
 * \file compressor.cpp
 * \brief
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/compressor_kernel.h"
#include "arch22/compressor_kernel_perf.h"
#else
#include "arch35/compressor_kernel.h"
#include "arch35/compressor_kernel_full_load.h"
#endif
 
using namespace Compressor;

#define INVOKE_COMPRESSOR_GENERAL_OP_IMPL(templateClass, ...)                                                          \
    do {                                                                                                               \
        templateClass<COMPType<__VA_ARGS__>> op(&pipe, tilingData);                                                    \
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,  \
                cuSeqlens, seqUsed, startPos, cmpKvOut, workspace);                                                    \
        op.Process();                                                                                                  \
    } while (0)

template<uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId>
__global__ __aicore__ void compressor(
    __gm__ uint8_t *x,
    __gm__ uint8_t *wKv,
    __gm__ uint8_t *wGate,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressorTilingData, tilingDataIn, tiling);
    if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::EMPTY_X) {
        return;
    }
    const optiling::CompressorTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    constexpr auto xLayout = static_cast<X_LAYOUT>(XLayout);
    constexpr auto xDtype = static_cast<X_DTYPE>(XDType);
    constexpr auto coff = static_cast<COFF>(Coff);
    constexpr auto rotaryMode = static_cast<ROTARY_MODE>(RotaryMode);
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
    #if (__CCE_AICORE__ == 220)
        if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::PERF) {
            INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, xLayout, xDtype, coff, rotaryMode, cacheMode);
        } else {
            INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode, cacheMode);
        }
    #else
        if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::FULL_LOAD) {
            INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelFullLoad, xLayout, xDtype, coff, rotaryMode, cacheMode);
        } else {
            INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, xLayout, xDtype, coff, rotaryMode, cacheMode);
        }
    #endif
}
