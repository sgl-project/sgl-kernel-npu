/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "register/op_impl_registry.h"
#include <cstring>

namespace optiling {
static ge::graphStatus TilingCompressor(gert::TilingContext *context)
{
    if (!context) return ge::GRAPH_FAILED;
    auto *raw = context->GetRawTilingData();
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(CompressorTilingData)) return ge::GRAPH_FAILED;
    // Keep the copied algorithm's original platform-information source.
    if (!platform_ascendc::PlatformAscendCManager::GetInstance()) return ge::GRAPH_FAILED;
    sglang::ge_helper::TilingContext legacyContext(*context);
    const auto *attrs = legacyContext.GetAttrs();
    if (!attrs || *attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX) != 2) return ge::GRAPH_FAILED;
    CompressorContext compressorContext{};
    if (CompressorTiling::ConvertContext(legacyContext, compressorContext) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    CompressorTilingData data{};
    CompressorTiling tiling(&compressorContext);
    if (tiling.RunBigKernelTiling(&data) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    data.tilingKey = compressorContext.tilingKey;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(compressorContext.blockDim);
    // The unchanged legacy dispatch key is in the payload; CANN has one entry.
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

struct SglCompressorCompileInfo {};
static ge::graphStatus TilingPrepare(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SglCompressor).Tiling(TilingCompressor).TilingParse<SglCompressorCompileInfo>(TilingPrepare);
}  // namespace optiling
