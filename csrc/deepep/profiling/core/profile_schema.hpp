/*
 * SPDX-License-Identifier: MIT
 */
#ifndef DEEPEP_PROFILING_CORE_PROFILE_SCHEMA_HPP
#define DEEPEP_PROFILING_CORE_PROFILE_SCHEMA_HPP

#include <cstdint>
#include <string>

#include "profiling/common/profile_protocol_common.h"

namespace deep_ep::profiling {

struct ProfileCoreTopology {
    uint32_t aicCount{0};
    uint32_t aivCount{0};
    uint32_t logicalCoreCount{0};
};

struct ProfileSchema {
    const char *opName{nullptr};
    uint32_t stageCount{0};
    uint32_t activeStageCapacity{0};
    ProfileCoreTopology topology{};
    const char *(*stageName)(uint64_t stageId){nullptr};
    std::string (*stageDisplayName)(uint64_t stageId, uint64_t occurrenceId,
                                    const Cam::ProfileStageLayout &layout){nullptr};
    // Returns flattened JSON fields that can be appended directly into args,
    // Return empty string when no private payload should be shown.
    std::string (*privateDataJson)(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                                   const Cam::ProfileStageLayout &layout){nullptr};
};

struct ProfileLaunchConfig {
    uint32_t groupCountCapacity{0};
    Cam::ProfileStageLayout stageLayout{};
};

struct ProfileOpRegistration {
    const char *opKey{nullptr};
    const ProfileSchema &(*schemaProvider)(){nullptr};
    const char *(*launchEventName)(){nullptr};
};

}  // namespace deep_ep::profiling

#endif  // DEEPEP_PROFILING_CORE_PROFILE_SCHEMA_HPP
