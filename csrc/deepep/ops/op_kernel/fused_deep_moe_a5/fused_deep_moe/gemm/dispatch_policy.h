#pragma once
#include "catlass_a5/gemm/dispatch_policy.hpp"

namespace Catlass::Gemm {

template <class ArchTag_, bool ENABLE_UNIT_FLAG_ = false, uint32_t L1_SCALE_FACTOR_K_ = 16, uint32_t L0C_STAGES_ = 1,
          bool ENABLE_L1_RESIDENT_ = false, uint32_t L1A_STAGES_ = 2, uint32_t L1B_STAGES_ = 2,
          uint32_t L0A_STAGES_ = 2, uint32_t L0B_STAGES_ = 2>
struct MmadMxWithCallback : public MmadMx<ArchTag_, ENABLE_UNIT_FLAG_, L1_SCALE_FACTOR_K_, L0C_STAGES_,
                                          ENABLE_L1_RESIDENT_, L1A_STAGES_, L1B_STAGES_, L0A_STAGES_, L0B_STAGES_> {};

}  // namespace Catlass::Gemm
