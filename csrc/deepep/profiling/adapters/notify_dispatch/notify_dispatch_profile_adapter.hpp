#ifndef DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_ADAPTER_HPP
#define DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_ADAPTER_HPP

#include <cstdint>

#include "profiling/core/profile_runtime.hpp"

namespace deep_ep::profiling::notify_dispatch {

using LaunchContext = runtime::ProfileLaunchContext;

bool IsActive();

LaunchContext PrepareLaunch(bool profileEnable);
void CompleteLaunch(const LaunchContext &ctx, int64_t rank);

}  // namespace deep_ep::profiling::notify_dispatch

#endif  // DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_ADAPTER_HPP