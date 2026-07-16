/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_DST_HPP
#define CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_DST_HPP

namespace Catlass::Conv::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToDstQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToDst fp32 to fp32
template <>
struct CopyL0CToDstQuantMode<
    Arch::Ascend950,
    float, float,
    ScaleGranularity::NO_QUANT    
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToDst fp32 to fp16
template <>
struct CopyL0CToDstQuantMode<
    Arch::Ascend950,
    float, half,
    ScaleGranularity::NO_QUANT    
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

// CopyL0CToDst fp32 to bf16
template <>
struct CopyL0CToDstQuantMode<
    Arch::Ascend950,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT    
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

} // namespace Catlass::Conv::Tile
#endif // CATLASS_CONV_TILE_ASCEND950_COPY_L0C_TO_DST_HPP