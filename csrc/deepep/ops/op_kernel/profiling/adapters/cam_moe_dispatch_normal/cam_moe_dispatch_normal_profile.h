#ifndef CAM_MOE_DISPATCH_NORMAL_PROFILE_H
#define CAM_MOE_DISPATCH_NORMAL_PROFILE_H

#include "../../kernel/profile_writer_kernel.h"
#include "cam_moe_dispatch_normal_profile_payload.h"
#include "cam_moe_dispatch_normal_profile_stage.h"

namespace Cam {

using CamMoeDispatchNormalProfileStage = deep_ep::profiling::cam_moe_dispatch_normal::ProfileStage;

// dispatch 算子专属的 ProfileWriter 封装（仅 AIV 参与打点，AIC 槽位留空即可）。
struct CamMoeDispatchNormalProfileWriter : public ProfileWriter {
    __aicore__ inline void Init(GM_ADDR profileGM, bool enable, uint32_t launchId_, uint32_t coreType_,
                                uint64_t profileBufferBytes_)
    {
        ProfileWriter::Init(profileGM, enable, launchId_, coreType_,
                            static_cast<uint32_t>(deep_ep::profiling::cam_moe_dispatch_normal::ProfileStage::Count),
                            profileBufferBytes_);
    }

    __aicore__ inline void Record(CamMoeDispatchNormalProfileStage stage, uint64_t startCycle, uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle);
    }

    __aicore__ inline void Record(CamMoeDispatchNormalProfileStage stage, uint32_t occurrenceId, uint64_t startCycle,
                                  uint64_t endCycle) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), occurrenceId, startCycle, endCycle);
    }

    __aicore__ inline void Record(CamMoeDispatchNormalProfileStage stage, uint64_t startCycle, uint64_t endCycle,
                                  const ProfilePrivatePayloadRaw &payload) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), startCycle, endCycle, payload);
    }

    __aicore__ inline void Record(CamMoeDispatchNormalProfileStage stage, uint32_t occurrenceId, uint64_t startCycle,
                                  uint64_t endCycle, const ProfilePrivatePayloadRaw &payload) const
    {
        ProfileWriter::Record(static_cast<uint32_t>(stage), occurrenceId, startCycle, endCycle, payload);
    }
};

}  // namespace Cam

#endif  // CAM_MOE_DISPATCH_NORMAL_PROFILE_H
