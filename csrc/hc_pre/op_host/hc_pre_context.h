/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <array>
#include <cstdio>
#include "ge_helper.h"

// Keep the upstream tiling API local to this op; the shared helper intentionally
// does not provide OPP launch metadata or nullable optional shapes.
namespace sglang::hc_pre {
class TilingContext : public ge_helper::TilingContext
{
public:
    TilingContext() : ge_helper::TilingContext("hc_pre") {}
    const gert::StorageShape *GetInputShape(uint32_t index) const
    {
        return GetInputTensor(index) ? ge_helper::TilingContext::GetInputShape(index) : nullptr;
    }
    const gert::StorageShape *GetOutputShape(uint32_t index) const
    {
        return GetOutputTensor(index) ? ge_helper::TilingContext::GetOutputShape(index) : nullptr;
    }
    void SetBlockDim(uint32_t value)
    {
        blockDim = value;
    }
    void SetTilingKey(uint64_t value)
    {
        tilingKey = value;
    }
    template <typename T>
    void SaveTilingData(const T &data)
    {
        TORCH_CHECK(data.GetDataSize() == serialized.size(), "hc_pre: unexpected tiling data size");
        data.SaveToBuffer(serialized.data(), serialized.size());
    }
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    std::array<uint8_t, 416> serialized{};
};
}  // namespace sglang::hc_pre

#define OPS_LOG_E(context, ...)            \
    do {                                   \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n");        \
    } while (0)
#define OPS_LOG_I(context, ...) ((void)0)
#define OPS_REPORT_VECTOR_INNER_ERR(context, ...) OPS_LOG_E(context, __VA_ARGS__)
#define OPS_ERR_IF(cond, log, expr) \
    do {                            \
        if (cond) {                 \
            log;                    \
            expr;                   \
        }                           \
    } while (0)
#define OPS_LOG_E_IF_NULL(context, ptr, expr) \
    OPS_ERR_IF((ptr) == nullptr, OPS_LOG_E(context, "hc_pre: null " #ptr), expr)
