/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Derived from the legacy Host UB calculation at upstream dc112d9ef68c.
#include "../op_kernel/recurrent_gated_delta_rule_tiling_data.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstring>
#include <limits>

namespace optiling {
static ge::graphStatus TilingRecurrentGatedDeltaRule(gert::TilingContext *context)
{
    if (!context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 5; ++i) {
        const auto *desc = context->GetInputDesc(i);
        if (!desc || !context->GetInputShape(i) || desc->GetDataType() != (i < 3 ? ge::DT_BF16 : ge::DT_INT32))
            return ge::GRAPH_FAILED;
    }
    const auto &mix = context->GetInputShape(0)->GetStorageShape();
    const auto &state = context->GetInputShape(1)->GetStorageShape();
    const auto &beta = context->GetInputShape(2)->GetStorageShape();
    const auto &lengths = context->GetInputShape(3)->GetStorageShape();
    const auto &indices = context->GetInputShape(4)->GetStorageShape();
    const auto *attrs = context->GetAttrs();
    const auto *scale = attrs ? attrs->GetAttrPointer<float>(0) : nullptr;
    const auto *nkAttr = attrs ? attrs->GetAttrPointer<int64_t>(1) : nullptr;
    const auto *nvAttr = attrs ? attrs->GetAttrPointer<int64_t>(2) : nullptr;
    if (!scale || !nkAttr || !nvAttr || mix.GetDimNum() != 3 || state.GetDimNum() != 4 || beta.GetDimNum() != 3 ||
        lengths.GetDimNum() != 1 || indices.GetDimNum() != 2)
        return ge::GRAPH_FAILED;
    const int64_t b = mix.GetDim(0), s = mix.GetDim(1), nk = *nkAttr, nv = *nvAttr;
    const int64_t dv = state.GetDim(2), dk = state.GetDim(3);
    constexpr int64_t LIMIT = std::numeric_limits<uint32_t>::max();
    if (b <= 0 || s <= 0 || s > 8 || nk <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || nk > LIMIT || nv > LIMIT ||
        dk > LIMIT / 16 || dv > LIMIT / 16 || b > LIMIT / s || b > LIMIT / nv || nv % nk != 0 || state.GetDim(0) <= 0 ||
        state.GetDim(1) != nv || nk > (LIMIT / 2) / (2 * dk) || nv > (LIMIT / 2 - 2 * nk * dk) / dv ||
        mix.GetDim(2) != 2 * nk * dk + nv * dv || beta.GetDim(0) != b || beta.GetDim(1) != s || beta.GetDim(2) != nv ||
        lengths.GetDim(0) != b || indices.GetDim(0) != b || indices.GetDim(1) != s)
        return ge::GRAPH_FAILED;

    const bool intermediate = context->GetOptionalInputDesc(5) != nullptr;
    if (intermediate) {
        const auto *seed = context->GetOptionalInputShape(5);
        const auto *cache = context->GetOptionalInputShape(6);
        if (!seed || !cache || context->GetOptionalInputDesc(5)->GetDataType() != ge::DT_BF16 ||
            !context->GetOptionalInputDesc(6) || context->GetOptionalInputDesc(6)->GetDataType() != ge::DT_INT32)
            return ge::GRAPH_FAILED;
        const auto &shape = seed->GetStorageShape();
        const auto &cacheShape = cache->GetStorageShape();
        if (shape.GetDimNum() != 4 || shape.GetDim(0) <= 0 || shape.GetDim(1) != nv || shape.GetDim(2) != dv ||
            shape.GetDim(3) != dk || cacheShape.GetShapeSize() < b)
            return ge::GRAPH_FAILED;
    }
    const bool accepted = context->GetOptionalInputDesc(7) != nullptr;
    if (accepted) {
        const auto *shape = context->GetOptionalInputShape(7);
        if (!shape || context->GetOptionalInputDesc(7)->GetDataType() != ge::DT_INT32 ||
            shape->GetStorageShape().GetShapeSize() < b)
            return ge::GRAPH_FAILED;
    }
    const bool gamma = context->GetOptionalInputDesc(8) != nullptr;
    // Gate copies use uint16 byte counts. Bound the maximum sequence length
    // without reading actual_seq_lengths back from the device.
    constexpr int64_t COPY_LIMIT = std::numeric_limits<uint16_t>::max();
    if (nv > COPY_LIMIT / sizeof(uint16_t) / s || (gamma && nv > COPY_LIMIT / sizeof(float) / s))
        return ge::GRAPH_FAILED;
    if (gamma) {
        const auto *shape = context->GetOptionalInputShape(8);
        if (!shape || context->GetOptionalInputDesc(8)->GetDataType() != ge::DT_FLOAT ||
            shape->GetStorageShape().GetShapeSize() < b * s * nv)
            return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputDesc(9) && context->GetOptionalInputDesc(9)->GetDataType() != ge::DT_FLOAT)
        return ge::GRAPH_FAILED;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const bool arch35 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    const auto coreNum = arch35 ? platform.GetCoreNumAiv() : platform.GetCoreNum();
    if (coreNum <= 0 || static_cast<uint64_t>(coreNum) > LIMIT || ubSize > static_cast<uint64_t>(LIMIT))
        return ge::GRAPH_FAILED;

    // Same two-stage vStep/UB budget as the direct-launch Host. Guard only invalid
    // budgets and narrowing; valid configurations retain the original arithmetic.
    auto align = [](int64_t n, int64_t a) { return (n + a - 1) & ~(a - 1); };
    auto ceilDiv = [](int64_t n, int64_t d) { return (n + d - 1) / d; };
    const int64_t aNv = align(nv, 16), aDv = align(dv, 16), aDk = align(dk, 16);
    // arch22 BinaryRepeatParams stores FP32 row strides in uint8 32-byte blocks.
    if (!arch35 && aDk / (32 / sizeof(float)) > std::numeric_limits<uint8_t>::max()) return ge::GRAPH_FAILED;
    int64_t used = 8 * (4 * aDk + 2 * aDv) + 128 + 8 * (4 * aNv + 2 * aNv);
    int64_t rest = static_cast<int64_t>(ubSize) - used;
    used += 8 * (8 * aDk + 4 * aDv + 4 * aNv);
    const int64_t coeff = 4 * aDk + 4 + 8 * aDk + 8;
    if (static_cast<int64_t>(ubSize) <= used) return ge::GRAPH_FAILED;
    int64_t vStep = (static_cast<int64_t>(ubSize) - used) / coeff / 8 * 8;
    if (vStep < 8) return ge::GRAPH_FAILED;
    vStep = align(ceilDiv(dv, ceilDiv(dv, vStep)), 8);
    rest -= (4 * aDk + 4) * vStep;
    if (rest <= 0 || vStep > LIMIT || rest > LIMIT) return ge::GRAPH_FAILED;

    SglRecurrentGatedDeltaRuleTilingData data{static_cast<uint32_t>(b),
                                              static_cast<uint32_t>(s),
                                              static_cast<uint32_t>(nk),
                                              static_cast<uint32_t>(dk),
                                              static_cast<uint32_t>(nv),
                                              static_cast<uint32_t>(dv),
                                              static_cast<uint32_t>(intermediate),
                                              static_cast<uint32_t>(accepted),
                                              static_cast<uint32_t>(gamma),
                                              static_cast<uint32_t>(vStep),
                                              static_cast<uint32_t>(rest),
                                              *scale};
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(data) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &data, sizeof(data));
    raw->SetDataSize(sizeof(data));
    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    context->SetTilingKey(0);
    workspace[0] = 0;  // AIV-only computation has no algorithm workspace.
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(SglRecurrentGatedDeltaRule).Tiling(TilingRecurrentGatedDeltaRule);
}  // namespace optiling
