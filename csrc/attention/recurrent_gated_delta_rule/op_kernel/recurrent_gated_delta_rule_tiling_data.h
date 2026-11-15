// Shared, fixed-width CANN transport for the original scalar launch parameters.
#ifndef SGL_CANN_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H
#define SGL_CANN_RECURRENT_GATED_DELTA_RULE_TILING_DATA_H

#include <cstdint>

namespace optiling {
struct SglRecurrentGatedDeltaRuleTilingData {
    uint32_t b;
    uint32_t s;
    uint32_t nk;
    uint32_t dk;
    uint32_t nv;
    uint32_t dv;
    uint32_t hasIntermediateState;
    uint32_t hasAcceptedTokens;
    uint32_t hasGama;
    uint32_t vStep;
    uint32_t ubRestBytes;
    float scale;
};
static_assert(sizeof(SglRecurrentGatedDeltaRuleTilingData) == 48,
              "Host and Kernel must share the scalar launch layout");
}  // namespace optiling
#endif
