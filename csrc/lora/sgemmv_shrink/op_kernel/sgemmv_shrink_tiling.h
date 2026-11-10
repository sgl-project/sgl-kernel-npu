// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#ifndef SGL_KERNEL_NPU_SGEMMV_SHRINK_TILING_H
#define SGL_KERNEL_NPU_SGEMMV_SHRINK_TILING_H

#include <cstdint>

namespace optiling {

// Field order follows the legacy Kernel Init argument order so the Host and
// device sides can be compared one-to-one.
struct SglSgemmvShrinkTilingData {
    uint32_t loraIndicesSize;
    uint32_t seqLenSize;
    uint32_t loraRanksSize;
    uint32_t loraScalesSize;
    uint32_t batchSize;
    uint32_t numTokensPerCore;
    uint32_t inputHiddenDim;
    uint32_t maxLoRARank;
    uint32_t slicesCount;
};
static_assert(sizeof(SglSgemmvShrinkTilingData) == 9 * sizeof(uint32_t),
              "Host and Kernel must share the nine-field tiling layout");

}  // namespace optiling

#endif  // SGL_KERNEL_NPU_SGEMMV_SHRINK_TILING_H
