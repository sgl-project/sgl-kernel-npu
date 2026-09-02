#ifndef DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H
#define DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H

#include "profiling/common/profile_protocol_common.h"

namespace Cam {

struct CombineTokenCountPrivatePayloadV1 {
    uint64_t header;
    uint64_t tokenCount;
    uint64_t rank;
};

static_assert(sizeof(CombineTokenCountPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "Combine payload must fit into raw payload slots");

inline constexpr CombineTokenCountPrivatePayloadV1
AsCombineTokenCountPrivatePayloadV1(const ProfilePrivatePayloadRaw &payload)
{
    return CombineTokenCountPrivatePayloadV1{payload.private0, payload.private1, payload.private2};
}

inline constexpr CombineTokenCountPrivatePayloadV1 AsCombineTokenCountPrivatePayloadV1(const ProfileRecord &record)
{
    return AsCombineTokenCountPrivatePayloadV1(record.payload);
}

}  // namespace Cam

#endif  // DEEPEP_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H
