#include "profiling/adapters/cam_moe_combine_normal/cam_moe_combine_normal_profile_traits.hpp"

#include <sstream>

#include "exception.hpp"

namespace deep_ep::profiling::cam_moe_combine_normal {

namespace {
// combine 无 group 语义，groupCountCapacity 固定为 1；聚合打点下每个 stage 每 launch 一条记录，
// multi-round 按 occurrence=round 记录，容量按最大轮数（64）预留。
constexpr uint32_t kRoundOccurrenceCapacity = 64U;
constexpr uint8_t kCombineTokenCountPrivateFormatV1 = 1U;
}  // namespace

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "cam_moe_combine_normal",
        kStageCount,
        Cam::PROFILE_ACTIVE_STAGE_CAPACITY,
        {Cam::PROFILE_AIC_COUNT_CAPACITY, Cam::PROFILE_AIV_COUNT_CAPACITY, Cam::PROFILE_LOGICAL_CORE_COUNT_CAPACITY},
        &GetStageName,
        &GetStageDisplayName,
        &GetPrivateDataJson,
    };
    return schema;
}

const ProfileOpRegistration &GetProfileRegistration()
{
    static const ProfileOpRegistration registration{
        "cam_moe_combine_normal",
        &GetProfileSchema,
        &GetLaunchEventName,
    };
    return registration;
}

const char *GetLaunchEventName()
{
    return "cam_moe_combine_normal_launch";
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::SendCopyToShare:
            return "send_copy_to_share";
        case ProfileStage::SendSetStatus:
            return "send_set_status";
        case ProfileStage::RecvWaitStatus:
            return "recv_wait_status";
        case ProfileStage::RecvReadAndSum:
            return "recv_read_and_sum";
        case ProfileStage::SetRoundStatus:
            return "set_round_status";
        case ProfileStage::WaitRoundStatus:
            return "wait_round_status";
        default:
            return "unknown";
    }
}

std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    // 聚合打点：每个 stage 每 launch（或每轮）一条记录，展示名即阶段名。
    return GetStageName(stageId);
}

std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    const auto payload = Cam::AsCombineTokenCountPrivatePayloadV1(record);
    if (Cam::GetProfilePrivateValidTag(payload.header) == Cam::PROFILE_PRIVATE_DATA_INVALID) {
        return {};
    }
    if (Cam::GetProfilePrivateFormatId(payload.header) != kCombineTokenCountPrivateFormatV1) {
        return {};
    }
    auto stage = static_cast<ProfileStage>(stageId);
    std::ostringstream oss;
    switch (stage) {
        case ProfileStage::SendCopyToShare:
        case ProfileStage::SendSetStatus:
            oss << ",\"send_token_count\":" << payload.tokenCount << ",\"send_rank\":" << payload.rank;
            break;
        case ProfileStage::RecvWaitStatus:
        case ProfileStage::RecvReadAndSum:
            oss << ",\"recv_token_count\":" << payload.tokenCount;
            break;
        default:
            return {};
    }
    return oss.str();
}

Cam::ProfileStageLayout BuildStageLayout()
{
    Cam::ProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(kStageCount);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::PROFILE_ACTIVE_STAGE_CAPACITY);
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SendCopyToShare),
                                                         kRoundOccurrenceCapacity),
                     "invalid send copy to share occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SendSetStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid send set status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::RecvWaitStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid recv wait status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::RecvReadAndSum),
                                                         kRoundOccurrenceCapacity),
                     "invalid recv read and sum occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SetRoundStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid set round status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WaitRoundStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid wait round status occurrence capacity.");
    return layout;
}

}  // namespace deep_ep::profiling::cam_moe_combine_normal
