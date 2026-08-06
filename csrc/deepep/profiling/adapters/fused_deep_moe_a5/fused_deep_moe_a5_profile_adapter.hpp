/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP

#include <cstdint>
#include <string>

#include "profiling/core/profile_runtime.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

using LaunchContext = runtime::ProfileLaunchContext;

bool IsActive();

LaunchContext PrepareLaunch(int64_t numExperts, int64_t numRanks, bool profileEnable);
void CompleteLaunch(const LaunchContext &ctx, int64_t rank);

}  // namespace deep_ep::profiling::fused_deep_moe_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_ADAPTER_HPP
