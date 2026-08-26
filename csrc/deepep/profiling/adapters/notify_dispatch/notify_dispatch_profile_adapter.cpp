#include "profiling/adapters/notify_dispatch/notify_dispatch_profile_adapter.hpp"

#include "profiling/adapters/notify_dispatch/notify_dispatch_profile_traits.hpp"
#include "profiling/core/profile_runtime.hpp"
#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::notify_dispatch {

bool IsActive()
{
    return runtime::IsSessionActive();
}

LaunchContext PrepareLaunch(bool profileEnable)
{
    ProfileLaunchConfig launchConfig{};
    launchConfig.groupCountCapacity = 1U;
    launchConfig.stageLayout = BuildStageLayout();
    return runtime::PrepareLaunch(GetProfileRegistration(), launchConfig, profileEnable);
}

void CompleteLaunch(const LaunchContext &ctx, int64_t rank)
{
    runtime::CompleteLaunch(ctx, rank);
}

}  // namespace deep_ep::profiling::notify_dispatch