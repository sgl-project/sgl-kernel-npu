#pragma once
#include <cstdint>

namespace optiling {
struct SglApplyTokenBitMaskTilingData {
    uint32_t numRows;
    uint32_t vocabSize;
    uint32_t logitsStride;
    uint32_t bitmaskStride;
    uint32_t baseRows;
    uint32_t extraCores;
    uint32_t tileLength;
    uint32_t blockDim;
    uint32_t dtypeSize;
};
static_assert(sizeof(SglApplyTokenBitMaskTilingData) == 9 * sizeof(uint32_t),
              "Host and Kernel must share the nine-field tiling layout");
}  // namespace optiling
