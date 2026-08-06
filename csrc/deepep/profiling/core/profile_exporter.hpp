/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP

#include <string>
#include <vector>

#include <torch/types.h>

#include "profiling/core/profile_schema.hpp"
#include "profiling/core/profile_session.hpp"

namespace deep_ep::profiling::exporter {

struct ProfileTraceSource {
    const at::Tensor *profileBuffer{nullptr};
    const ProfileSchema *schema{nullptr};
    const char *launchEventName{nullptr};
    int64_t numProfileSkipLaunches{0};
    int64_t launchCountCaptured{0};
};

void ExportBufferToTrace(const at::Tensor &profileBuffer, int64_t rank, const std::string &profileTraceDir,
                         int64_t numProfileSkipLaunches, int64_t launchCountCaptured, const ProfileSchema &schema,
                         const session::ProfileTimeCalibration *calibration = nullptr, int64_t numRanks = 1);
void ExportAggregatedTrace(const std::vector<ProfileTraceSource> &sources, int64_t rank,
                           const std::string &profileTraceDir,
                           const session::ProfileTimeCalibration *calibration = nullptr, int64_t numRanks = 1);

}  // namespace deep_ep::profiling::exporter

#endif  // DEEPEP_PROFILING_CORE_PROFILE_EXPORTER_HPP
