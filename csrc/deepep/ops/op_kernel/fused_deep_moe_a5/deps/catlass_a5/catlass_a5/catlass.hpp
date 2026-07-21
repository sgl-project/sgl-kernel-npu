/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CATLASS_HPP
#define CATLASS_CATLASS_HPP

#include <cstdint>

#if defined(__CCE__)
#include <kernel_operator.h>
#endif

#include "catlass_a5/detail/alignment.hpp"
#include "catlass_a5/detail/dependent_false.hpp"
#include "catlass_a5/detail/macros.hpp"

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

#if !defined(CATLASS_ARCH) || CATLASS_ARCH == 2201
constexpr uint32_t BYTE_PER_BLK_FP = 128;  /// datablock size of A1->C2PiPE2GM
#elif defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
constexpr uint32_t BYTE_PER_BLK_FP = 64;
#endif

class EmptyClass
{};

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
constexpr uint32_t MX_SCALE_COPY_GROUP_NUM = 2;  // Mx-scale matrix 2-byte aligned
constexpr uint32_t MX_SCALE_GROUP_NUM = 32;      // Data count for one MX-scale factor per group
constexpr uint32_t MX_BASEK_FACTOR = 64;         // Data matrix alignment at K-dimension
#endif
}  // namespace Catlass

#endif  // CATLASS_CATLASS_HPP
