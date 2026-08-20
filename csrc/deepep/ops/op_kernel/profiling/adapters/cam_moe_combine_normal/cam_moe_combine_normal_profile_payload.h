#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H

#include "../../common/profile_protocol_common.h"

namespace Cam {

constexpr uint8_t COMBINE_TOKEN_COUNT_PRIVATE_FORMAT_V1 = 1U;

// 统一 payload：发送阶段携带“本核（本轮）实际发送 token 数 + 目标 rank”，
// 接收阶段携带“本核（本轮）实际接收（处理）的 token 数”，rank 默认 0。
struct CombineTokenCountPrivatePayloadV1 {
    uint64_t header;
    uint64_t tokenCount;
    uint64_t rank;
};

static_assert(sizeof(CombineTokenCountPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "Combine payload must fit into raw payload slots");

__aicore__ inline constexpr CombineTokenCountPrivatePayloadV1 MakeCombineTokenCountPrivatePayloadV1(
    uint8_t validTag, uint8_t formatId, uint64_t tokenCount, uint64_t rank = 0U)
{
    return CombineTokenCountPrivatePayloadV1{PackProfilePrivate0(validTag, formatId), tokenCount, rank};
}

__aicore__ inline constexpr ProfilePrivatePayloadRaw
ToProfilePrivatePayloadRaw(const CombineTokenCountPrivatePayloadV1 &payload)
{
    return MakeProfilePrivatePayloadRaw(payload.header, payload.tokenCount, payload.rank);
}

}  // namespace Cam

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_PAYLOAD_H
