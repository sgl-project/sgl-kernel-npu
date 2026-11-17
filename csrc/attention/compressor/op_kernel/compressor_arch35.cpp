/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define SGL_COMPRESSOR_ARCH35 1
#include "compressor_cann_entry.h"
#include "arch35/sgl_compressor_tiling_key.h"

#include <type_traits>

using namespace Compressor;

// Each generated key instantiates only the selected calculation path.
template <uint8_t XLayout, uint8_t XDType, uint8_t Coff, uint8_t RotaryMode, uint8_t CacheMode, uint8_t TemplateId>
__global__ __aicore__ void sgl_compressor(GM_ADDR x, GM_ADDR wKv, GM_ADDR wGate, GM_ADDR stateCache,
                                        GM_ADDR ape, GM_ADDR normWeight, GM_ADDR ropeSin, GM_ADDR ropeCos,
                                        GM_ADDR stateBlockTable, GM_ADDR cuSeqlens, GM_ADDR seqUsed,
                                        GM_ADDR startPos, GM_ADDR cmpKvOut, GM_ADDR stateCacheOut,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::CompressorTilingData);
    workspace = AscendC::GetUserWorkspace(workspace);
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    constexpr auto templateId = static_cast<TEMPLATE_ID>(TemplateId);
    if constexpr (templateId != TEMPLATE_ID::EMPTY_X) {
        auto tilingData = reinterpret_cast<__gm__ optiling::CompressorTilingData *>(tiling);
        using Config = COMPType<static_cast<X_LAYOUT>(XLayout), static_cast<X_DTYPE>(XDType),
                                static_cast<COFF>(Coff), static_cast<ROTARY_MODE>(RotaryMode),
                                static_cast<CACHE_MODE>(CacheMode)>;
        using Kernel = std::conditional_t<templateId == TEMPLATE_ID::FULL_LOAD,
                                          CompressorKernelFullLoad<Config>, CompressorKernel<Config>>;
        Kernel op(&pipe, tilingData);
        op.Init(x, wKv, wGate, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable, cuSeqlens, seqUsed,
                startPos, cmpKvOut, workspace);
        op.Process();
    }
}
