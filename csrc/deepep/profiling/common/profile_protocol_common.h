#ifndef DEEPEP_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H
#define DEEPEP_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define DEEPEP_PROFILE_INLINE __aicore__ inline
#else
#define DEEPEP_PROFILE_INLINE inline
#endif

#include "ops/profiling/common/profile_protocol_common_core.h"

namespace Cam {

DEEPEP_PROFILE_INLINE void SetProfileRecordPayload(ProfileRecord &record, const ProfilePrivatePayloadRaw &payload)
{
    record.payload = payload;
}

}  // namespace Cam

#undef DEEPEP_PROFILE_INLINE

#endif  // DEEPEP_PROFILING_COMMON_PROFILE_PROTOCOL_COMMON_H
