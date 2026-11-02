// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#ifndef SGL_KERNEL_NPU_SGEMMV_EXPAND_TILING_H
#define SGL_KERNEL_NPU_SGEMMV_EXPAND_TILING_H

#include <cstdint>

namespace optiling {

// Field order follows the legacy Kernel Init argument order so the Host and
// device sides can be compared one-to-one. sliceCount stays derived from
// sliceOffsetsSize - 1 in the Kernel, exactly as in the legacy implementation.
struct SglSgemmvExpandTilingData {
    uint32_t loraIndicesSize;
    uint32_t seqLenSize;
    uint32_t loraRanksSize;
    uint32_t sliceOffsetsSize;
    uint32_t batchSize;
    uint32_t numTokensPerCore;
    uint32_t maxLoRARank;
    uint32_t outputFullDim;
};
static_assert(sizeof(SglSgemmvExpandTilingData) == 8 * sizeof(uint32_t),
              "Host and Kernel must share the eight-field tiling layout");

}  // namespace optiling

#endif  // SGL_KERNEL_NPU_SGEMMV_EXPAND_TILING_H
