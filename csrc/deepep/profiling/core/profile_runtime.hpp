#ifndef DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP

#include <cstdint>
#include <string>

#include <torch/types.h>

#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::runtime {

struct ProfileLaunchContext {
    bool enabled{false};
    bool sessionActive{false};
    int64_t launchId{0};
    int64_t profileBufferBytes{0};
    const ProfileOpRegistration *registration{nullptr};
    const at::Tensor *profileBuffer{nullptr};
    at::Tensor ownedProfileBuffer;
};

bool IsSessionActive();
std::string GetProfileTraceDir();
int64_t GetNumProfileSkipLaunches();
int64_t GetNumProfileActiveLaunches();
int64_t GetExpectedLaunches();

void BeginSession(int64_t numProfileSkipLaunches, int64_t numProfileActiveLaunches, const std::string &profileTraceDir,
                  int64_t numRanks);
void EndSession(int64_t rank);
void CaptureSessionBeginAnchor(int64_t rank);
void CaptureSessionEndAnchor(int64_t rank);

ProfileLaunchContext PrepareLaunch(const ProfileOpRegistration &registration, const ProfileLaunchConfig &launchConfig,
                                   bool profileEnable);
void CompleteLaunch(const ProfileLaunchContext &ctx, int64_t rank);

}  // namespace deep_ep::profiling::runtime

#endif  // DEEPEP_PROFILING_CORE_PROFILE_RUNTIME_HPP
