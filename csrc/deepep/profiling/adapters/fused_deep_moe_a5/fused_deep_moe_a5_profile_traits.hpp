#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_TRAITS_HPP
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_TRAITS_HPP

#include <cstdint>
#include <string>

#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_payload.h"
#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_stage.h"
#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

static_assert(kStageCount <= Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
              "fused_deep_moe_a5 stage count must fit in active profiling stage capacity");

const ProfileSchema &GetProfileSchema();
const ProfileOpRegistration &GetProfileRegistration();
const char *GetLaunchEventName();
const char *GetStageName(uint64_t stageId);
std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout);
std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout);
Cam::ProfileStageLayout BuildStageLayout(uint32_t groupCountCapacity);
uint32_t GetGroupCountCapacity(int64_t numExperts, int64_t numRanks);

}  // namespace deep_ep::profiling::fused_deep_moe_a5

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_TRAITS_HPP
