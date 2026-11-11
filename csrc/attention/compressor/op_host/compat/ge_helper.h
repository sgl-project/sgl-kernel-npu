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

// Operator-local CANN boundary for the unchanged legacy tiling sources.
// No Torch dependency, SDK symbol replacement, or scheduling formula lives here.
#include "exe_graph/runtime/tiling_context.h"
#include <array>
#include <cstdint>
#include <type_traits>

namespace sglang {
namespace ge_helper {

class RuntimeAttrs
{
public:
    explicit RuntimeAttrs(gert::TilingContext &context)
    {
        const auto *attrs = context.GetAttrs();
        if (!attrs) return;
        valid_ = true;
        for (size_t index = 0; index < integers_.size(); ++index) {
            if (index == 3) continue;  // norm_eps is the sole float attribute.
            const auto *value = attrs->GetAttrPointer<int64_t>(index);
            if (!value) {
                valid_ = false;
                continue;
            }
            int64_t effective = *value;
            if (index == 6 && effective == 0) {
                const auto *cache = context.GetInputShape(3);
                if (!cache || cache->GetStorageShape().GetDimNum() != 3) {
                    valid_ = false;
                    continue;
                }
                effective = cache->GetStorageShape().GetDim(1) * cache->GetStorageShape().GetDim(2);
            }
            // Match the original Torch Host's static_cast<int32_t> exactly.
            integers_[index] = static_cast<int32_t>(effective);
        }
        const auto *epsilon = attrs->GetAttrPointer<float>(3);
        if (epsilon)
            epsilon_ = *epsilon;
        else
            valid_ = false;
    }

    bool Valid() const
    {
        return valid_;
    }
    template <typename T>
    const T *GetAttrPointer(size_t index) const
    {
        if constexpr (std::is_same_v<T, int>) {
            static_assert(sizeof(int) == sizeof(int32_t));
            return index < integers_.size() && index != 3 ? &integers_[index] : nullptr;
        } else {
            static_assert(std::is_same_v<T, float>);
            return index == 3 ? &epsilon_ : nullptr;
        }
    }

private:
    std::array<int32_t, 7> integers_{};
    float epsilon_ = 1e-6f;
    bool valid_ = false;
};

class TilingContext
{
public:
    explicit TilingContext(gert::TilingContext &context) : context_(context), attrs_(context) {}
    const char *GetNodeName() const
    {
        return context_.GetNodeName();
    }
    const gert::CompileTimeTensorDesc *GetInputDesc(uint32_t index) const
    {
        return context_.GetInputDesc(index);
    }
    const gert::StorageShape *GetInputShape(uint32_t index) const
    {
        return context_.GetInputShape(index);
    }
    const gert::CompileTimeTensorDesc *GetOptionalInputDesc(uint32_t index) const
    {
        return context_.GetOptionalInputDesc(index);
    }
    const gert::StorageShape *GetOptionalInputShape(uint32_t index) const
    {
        return context_.GetOptionalInputShape(index);
    }
    const gert::CompileTimeTensorDesc *GetOutputDesc(uint32_t index) const
    {
        return context_.GetOutputDesc(index);
    }
    const gert::StorageShape *GetOutputShape(uint32_t index) const
    {
        return context_.GetOutputShape(index);
    }
    const RuntimeAttrs *GetAttrs() const
    {
        return attrs_.Valid() ? &attrs_ : nullptr;
    }
    size_t *GetWorkspaceSizes(size_t count)
    {
        return context_.GetWorkspaceSizes(count);
    }

private:
    gert::TilingContext &context_;
    RuntimeAttrs attrs_;  // Attribute pointers remain alive through RunBigKernelTiling.
};

}  // namespace ge_helper
}  // namespace sglang
