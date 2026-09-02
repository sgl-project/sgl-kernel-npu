#ifndef DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_TRAITS_HPP
#define DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_TRAITS_HPP

#include <cstdint>
#include <string>

#include "profiling/adapters/cam_moe_combine_normal/cam_moe_combine_normal_profile_payload.h"
#include "profiling/adapters/cam_moe_combine_normal/cam_moe_combine_normal_profile_stage.h"
#include "profiling/common/profile_protocol_common.h"
#include "profiling/core/profile_schema.hpp"

namespace deep_ep::profiling::cam_moe_combine_normal {

static_assert(kStageCount <= Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
              "cam_moe_combine_normal stage count must fit in active profiling stage capacity");

const ProfileSchema &GetProfileSchema();
const ProfileOpRegistration &GetProfileRegistration();
const char *GetLaunchEventName();
const char *GetStageName(uint64_t stageId);
std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout);
std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout);
Cam::ProfileStageLayout BuildStageLayout();

}  // namespace deep_ep::profiling::cam_moe_combine_normal

#endif  // DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_TRAITS_HPP
