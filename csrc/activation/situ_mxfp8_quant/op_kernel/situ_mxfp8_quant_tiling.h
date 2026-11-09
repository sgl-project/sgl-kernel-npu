#ifndef SGL_CANN_SITU_MXFP8_QUANT_TILING_H
#define SGL_CANN_SITU_MXFP8_QUANT_TILING_H

#include <cstdint>

namespace optiling {

// Fields mirror the legacy Kernel Init arguments one-to-one, in the same order.
struct SglSituMxfp8QuantTilingData {
    uint32_t capacityRows{0};
    uint32_t numExperts{0};
    uint32_t groupListType{1};
    uint32_t groupDtype{0};  // 0 = int32 group_list, 1 = int64 group_list
    float beta{4.0f};
    float linearBeta{25.0f};
};
static_assert(sizeof(SglSituMxfp8QuantTilingData) == 4 * sizeof(uint32_t) + 2 * sizeof(float),
              "Host and Kernel must share the same tiling layout");

// Fixed problem size of the A5 fused operator: 2 * hidden -> hidden.
constexpr uint32_t SITU_INPUT_COLS = 6144;
constexpr uint32_t SITU_OUTPUT_COLS = 3072;
constexpr uint32_t SITU_SCALE_COLS = SITU_OUTPUT_COLS / 32;

}  // namespace optiling

#endif  // SGL_CANN_SITU_MXFP8_QUANT_TILING_H
