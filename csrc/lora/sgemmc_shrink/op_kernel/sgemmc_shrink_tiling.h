// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

#ifndef SGL_KERNEL_NPU_SGEMMC_SHRINK_TILING_H
#define SGL_KERNEL_NPU_SGEMMC_SHRINK_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace optiling {

// Field order follows the legacy Kernel Init argument order; cubeTiling keeps
// the matmul tiling the legacy Host generated with MatmulApiTiling. The Host and
// Kernel share this definition, so both sides resolve every field identically.
struct SglSgemmcShrinkTilingData {
    uint32_t loraIndicesSize;
    uint32_t seqLenSize;
    uint32_t loraRanksSize;
    uint32_t loraScalesSize;
    uint32_t batchSize;
    uint32_t inputHiddenDim;
    uint32_t maxLoRARank;
    uint32_t slicesCount;
    AscendC::tiling::TCubeTiling cubeTiling;
};
static_assert(sizeof(SglSgemmcShrinkTilingData) % sizeof(uint32_t) == 0,
              "Tiling data must stay 32-bit word aligned for GET_TILING_DATA");

}  // namespace optiling

#endif  // SGL_KERNEL_NPU_SGEMMC_SHRINK_TILING_H
