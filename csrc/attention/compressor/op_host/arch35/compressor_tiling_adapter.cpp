/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compressor_tiling.h"
#include "../../op_kernel/arch35/sgl_compressor_tiling_key.h"

namespace optiling {
static uint64_t GetCompressorCANNKey(uint64_t payloadKey)
{
    // Decode the existing payload fields and let CANN encode its own dispatch key.
    const auto layout = payloadKey & 0x1;
    const auto dtype = (payloadKey >> 1) & 0xf;
    const auto coff = (payloadKey >> 5) & 0x3;
    const auto rotaryMode = (payloadKey >> 7) & 0x3;
    const auto cacheMode = (payloadKey >> 9) & 0x3;
    const auto templateId = (payloadKey >> 11) & 0x3;
    return GET_TPL_TILING_KEY(layout, dtype, coff, rotaryMode, cacheMode, templateId);
}
}  // namespace optiling

#define SGL_COMPRESSOR_HAS_CANN_KEY
#include "../compressor_tiling_adapter.h"
#undef SGL_COMPRESSOR_HAS_CANN_KEY
