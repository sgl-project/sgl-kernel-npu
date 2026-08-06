/*
 * SPDX-License-Identifier: MIT
 */

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_adapter.hpp"

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_traits.hpp"
#include "profiling/core/profile_runtime.hpp"
#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

bool IsActive()
{
    return runtime::IsSessionActive();
}

LaunchContext PrepareLaunch(int64_t numExperts, int64_t numRanks, bool profileEnable)
{
    uint32_t groupCountCapacity = GetGroupCountCapacity(numExperts, numRanks);
    ProfileLaunchConfig launchConfig{};
    launchConfig.groupCountCapacity = groupCountCapacity;
    launchConfig.stageLayout = BuildStageLayout(groupCountCapacity);
    return runtime::PrepareLaunch(GetProfileRegistration(), launchConfig, profileEnable);
}

void CompleteLaunch(const LaunchContext &ctx, int64_t rank)
{
    runtime::CompleteLaunch(ctx, rank);
}

}  // namespace deep_ep::profiling::fused_deep_moe_a5
