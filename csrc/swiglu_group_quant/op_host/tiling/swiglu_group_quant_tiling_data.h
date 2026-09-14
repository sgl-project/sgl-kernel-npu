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
 * \file swiglu_group_quant_tiling_data.h
 * \brief Plain tiling-data struct for swiglu_group_quant (A5-only).
 *
 * A plain packed POD, memcpy'd to the device by the host and read field-by-field from a __gm__
 * pointer by the kernel.
 *
 * Field names and types must match what the kernel headers dereference by name: all int64_t except
 * `clampValue`, which is float.
 *
 * Field order puts every int64_t first so each one is 8-byte aligned, keeping the kernel's scalar
 * GM loads naturally aligned under #pragma pack(1) — a packed struct does not get that for free.
 */

#ifndef SWIGLU_GROUP_QUANT_TILING_DATA_H
#define SWIGLU_GROUP_QUANT_TILING_DATA_H

#include <cstdint>

namespace sglang {

#pragma pack(push, 1)
struct SwigluGroupQuantTilingData {
    // ---- shape / row tiling (25 x int64_t, offsets 0..192, all 8-byte aligned) ----
    int64_t bs;
    int64_t d;
    int64_t splitD;
    int64_t scaleCol;
    int64_t rowOfFormerBlock;
    int64_t rowOfTailBlock;
    int64_t rowLoopOfFormerBlock;
    int64_t rowLoopOfTailBlock;
    int64_t rowFactor;
    int64_t tailRowFactorOfFormerBlock;
    int64_t tailRowFactorOfTailBlock;
    int64_t dLoop;
    int64_t dFactor;
    int64_t tailDFactor;
    int64_t roundScale;
    int64_t ue8m0Scale;
    int64_t outputOrigin;
    int64_t hasClampValue;
    int64_t g;
    int64_t ubSize;
    int64_t gLoop;
    int64_t gFactor;
    int64_t tailGFactor;
    int64_t groupListType;
    int64_t coreNum;

    // ---- dtype / launch metadata ----
    float clampValue;    // offset 200
    int32_t dtype;       // offset 204: bit0 x is bf16, bit1 y is e5m2, bit2 scale is e8m0
    uint32_t tilingKey;  // offset 208: 1 group, 2 mx, 31 fp8, 32 fp8 + y_origin
};
#pragma pack(pop)

// Tiling-key values; must match the kernel entry's dispatch and the host's GetTilingKey().
constexpr uint32_t SWIGLU_GROUP_QUANT_TILING_KEY_GROUP_QUANT = 1;
constexpr uint32_t SWIGLU_GROUP_QUANT_TILING_KEY_MX_QUANT = 2;
constexpr uint32_t SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT = 31;
constexpr uint32_t SWIGLU_GROUP_QUANT_TILING_KEY_FP8_QUANT_YORIGIN = 32;

/** dtype code bits, mirroring the host-side packing of the x/y/scale dtypes. */
constexpr int32_t SWIGLU_GROUP_QUANT_DTYPE_X_BF16 = 1 << 0;
constexpr int32_t SWIGLU_GROUP_QUANT_DTYPE_Y_E5M2 = 1 << 1;
constexpr int32_t SWIGLU_GROUP_QUANT_DTYPE_SCALE_E8M0 = 1 << 2;

}  // namespace sglang

#endif  // SWIGLU_GROUP_QUANT_TILING_DATA_H
