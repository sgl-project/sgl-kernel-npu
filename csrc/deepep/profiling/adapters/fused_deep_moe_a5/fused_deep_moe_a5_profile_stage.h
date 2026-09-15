#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::fused_deep_moe_a5 {

enum class ProfileStage : uint32_t {
    DispatchSend = 0,
    DispatchRecv = 1,
    Gmm1 = 2,
    Swiglu = 3,
    Quant = 4,
    StageBarrier = 5,
    Gmm2 = 6,
    Combine = 7,
    WeightSumAllToAllSend = 8,
    WeightSumReducePermute = 9,
    WeightSumClean = 10,
    Count = 11,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::fused_deep_moe_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_STAGE_H
