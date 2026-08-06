#ifndef DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
#define DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H

#include "profiling/common/profile_protocol_common.h"

namespace Cam {

struct DispatchSendPrivatePayloadV1 {
    uint64_t header;
    uint64_t validTokenCount;
    uint64_t perTokenCommBytes;
};

struct DispatchRecvPrivatePayloadV1 {
    uint64_t header;
    uint64_t aivTokenCount;
};

static_assert(sizeof(DispatchSendPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "DispatchSend payload must fit into raw payload slots");
static_assert(sizeof(DispatchRecvPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "DispatchRecv payload must fit into raw payload slots");

inline constexpr DispatchSendPrivatePayloadV1 AsDispatchSendPrivatePayloadV1(const ProfilePrivatePayloadRaw &payload)
{
    return DispatchSendPrivatePayloadV1{payload.private0, payload.private1, payload.private2};
}

inline constexpr DispatchSendPrivatePayloadV1 AsDispatchSendPrivatePayloadV1(const ProfileRecord &record)
{
    return AsDispatchSendPrivatePayloadV1(record.payload);
}

inline constexpr DispatchRecvPrivatePayloadV1 AsDispatchRecvPrivatePayloadV1(const ProfileRecord &record)
{
    return DispatchRecvPrivatePayloadV1{record.payload.private0, record.payload.private1};
}

}  // namespace Cam

#endif  // DEEPEP_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
