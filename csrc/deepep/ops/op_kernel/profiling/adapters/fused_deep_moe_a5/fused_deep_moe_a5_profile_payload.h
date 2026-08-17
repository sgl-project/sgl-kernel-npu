#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H

#include "../../common/profile_protocol_common.h"

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

__aicore__ inline constexpr DispatchSendPrivatePayloadV1 MakeDispatchSendPrivatePayloadV1(uint8_t validTag,
                                                                                          uint8_t formatId,
                                                                                          uint64_t validTokenCount,
                                                                                          uint64_t perTokenCommBytes)
{
    return DispatchSendPrivatePayloadV1{PackProfilePrivate0(validTag, formatId), validTokenCount, perTokenCommBytes};
}

__aicore__ inline constexpr ProfilePrivatePayloadRaw
ToProfilePrivatePayloadRaw(const DispatchSendPrivatePayloadV1 &payload)
{
    return MakeProfilePrivatePayloadRaw(payload.header, payload.validTokenCount, payload.perTokenCommBytes);
}

__aicore__ inline constexpr DispatchRecvPrivatePayloadV1 MakeDispatchRecvPrivatePayloadV1(uint8_t validTag,
                                                                                          uint8_t formatId,
                                                                                          uint64_t aivTokenCount)
{
    return DispatchRecvPrivatePayloadV1{PackProfilePrivate0(validTag, formatId), aivTokenCount};
}

__aicore__ inline constexpr ProfilePrivatePayloadRaw
ToProfilePrivatePayloadRaw(const DispatchRecvPrivatePayloadV1 &payload)
{
    return MakeProfilePrivatePayloadRaw(payload.header, payload.aivTokenCount, 0ULL);
}

}  // namespace Cam

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_FUSED_DEEP_MOE_A5_PROFILE_PAYLOAD_H
