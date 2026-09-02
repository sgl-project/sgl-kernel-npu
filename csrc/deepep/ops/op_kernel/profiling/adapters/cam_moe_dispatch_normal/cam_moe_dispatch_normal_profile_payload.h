#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H

#include "../../common/profile_protocol_common.h"

namespace Cam {

constexpr uint8_t SHARE_TO_OUTPUT_EXPERT_PRIVATE_FORMAT_V1 = 1U;

// ShareToOutputExpert（per-expert）阶段专用 payload。
// 原始槽位 24B（private0/private1/private2）限制下，4 个语义字段需合并：
//   private0 = PackProfilePrivate0(validTag, formatId)
//   private1 = (fromRank << 32) | localE    // fromRank：源端 rank；localE：本 rank 内专家索引
//   private2 = count                        // 该 expert 从源端接收的 token 数
struct ShareToOutputExpertPrivatePayloadV1 {
    uint64_t header;
    uint64_t fromRankLocalE;
    uint64_t count;
};

static_assert(sizeof(ShareToOutputExpertPrivatePayloadV1) <= sizeof(ProfilePrivatePayloadRaw),
              "ShareToOutputExpert payload must fit into raw payload slots");

__aicore__ inline constexpr ShareToOutputExpertPrivatePayloadV1 MakeShareToOutputExpertPrivatePayloadV1(
    uint8_t validTag, uint8_t formatId, uint64_t fromRank, uint64_t localE, uint64_t count)
{
    return ShareToOutputExpertPrivatePayloadV1{PackProfilePrivate0(validTag, formatId),
                                               (fromRank << 32) | (localE & 0xFFFFFFFFULL), count};
}

__aicore__ inline constexpr ProfilePrivatePayloadRaw
ToProfilePrivatePayloadRaw(const ShareToOutputExpertPrivatePayloadV1 &payload)
{
    return MakeProfilePrivatePayloadRaw(payload.header, payload.fromRankLocalE, payload.count);
}

}  // namespace Cam

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_PAYLOAD_H
