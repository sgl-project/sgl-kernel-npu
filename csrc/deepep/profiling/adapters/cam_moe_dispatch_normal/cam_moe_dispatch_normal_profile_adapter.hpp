#ifndef DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_ADAPTER_HPP
#define DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_ADAPTER_HPP

#include <cstdint>

#include "profiling/core/profile_runtime.hpp"

namespace deep_ep::profiling::cam_moe_dispatch_normal {

using LaunchContext = runtime::ProfileLaunchContext;

bool IsActive();

LaunchContext PrepareLaunch(bool profileEnable);
void CompleteLaunch(const LaunchContext &ctx, int64_t rank);

}  // namespace deep_ep::profiling::cam_moe_dispatch_normal

#endif  // DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_ADAPTER_HPP
