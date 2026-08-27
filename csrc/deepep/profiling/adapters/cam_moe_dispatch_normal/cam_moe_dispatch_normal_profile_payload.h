#ifndef DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H
#define DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H

#include "profiling/common/profile_protocol_common.h"

namespace Cam {

constexpr uint8_t SHARE_TO_OUTPUT_EXPERT_PRIVATE_FORMAT_V1 = 1U;

// 与 kernel 侧（ops/op_kernel/profiling/adapters/cam_moe_dispatch_normal/...）payload 布局保持一致。
// ShareToOutputExpert：private1 = (fromRank << 32) | localE，private2 = count。
struct ShareToOutputExpertPrivatePayloadV1 {
    uint64_t header;
    uint64_t fromRankLocalE;
    uint64_t count;
};

static_assert(sizeof(ShareToOutputExpertPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "ShareToOutputExpert payload must fit into raw payload slots");

inline constexpr uint64_t GetShareToOutputExpertFromRank(const ShareToOutputExpertPrivatePayloadV1 &payload)
{
    return payload.fromRankLocalE >> 32;
}

inline constexpr uint64_t GetShareToOutputExpertLocalE(const ShareToOutputExpertPrivatePayloadV1 &payload)
{
    return payload.fromRankLocalE & 0xFFFFFFFFULL;
}

inline constexpr ShareToOutputExpertPrivatePayloadV1
AsShareToOutputExpertPrivatePayloadV1(const ProfilePrivatePayloadRaw &payload)
{
    return ShareToOutputExpertPrivatePayloadV1{payload.private0, payload.private1, payload.private2};
}

inline constexpr ShareToOutputExpertPrivatePayloadV1 AsShareToOutputExpertPrivatePayloadV1(const ProfileRecord &record)
{
    return AsShareToOutputExpertPrivatePayloadV1(record.payload);
}

}  // namespace Cam

#endif  // DEEPEP_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H
