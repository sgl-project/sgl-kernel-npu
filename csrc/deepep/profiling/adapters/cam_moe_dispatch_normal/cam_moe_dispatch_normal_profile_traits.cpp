#include "profiling/adapters/cam_moe_dispatch_normal/cam_moe_dispatch_normal_profile_traits.hpp"

#include <sstream>

#include "exception.hpp"

namespace deep_ep::profiling::cam_moe_dispatch_normal {

namespace {
// dispatch 无 group 语义，groupCountCapacity 固定为 1。
// 每轮一条的阶段（发送/等待/轮间 barrier/公共准备）occurrence=round，multi-round 按最大轮数（64）预留；
// ShareToOutputExpert 为 per-expert 打点，occurrenceId 同时编码轮次与本核内 expert 序号
// （occurrenceId = roundIndex * maxExpertsPerCore + expertIndex），容量按协议上限 64 预留，
// 超过容量的记录会被 kernel 侧安全丢弃（详见 kernel 侧 Record 的 occurrenceId 校验）。
constexpr uint32_t kRoundOccurrenceCapacity = 64U;
constexpr uint8_t kShareToOutputExpertPrivateFormatV1 = 1U;
}  // namespace

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "cam_moe_dispatch_normal",
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
        "cam_moe_dispatch_normal",
        &GetProfileSchema,
        &GetLaunchEventName,
    };
    return registration;
}

const char *GetLaunchEventName()
{
    return "cam_moe_dispatch_normal_launch";
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::InputToShare:
            return "input_to_share";
        case ProfileStage::SetStatus:
            return "set_status";
        case ProfileStage::WaitStatus:
            return "wait_status";
        case ProfileStage::ShareToOutputCommon:
            return "share_to_output_common";
        case ProfileStage::ShareToOutputExpert:
            return "share_to_output_expert";
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
    // 展示名即阶段名；ShareToOutputExpert 的 expert 级信息通过 private payload（fromRank/localE/count）携带。
    return GetStageName(stageId);
}

std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    const auto payload = Cam::AsShareToOutputExpertPrivatePayloadV1(record);
    if (Cam::GetProfilePrivateValidTag(payload.header) == Cam::PROFILE_PRIVATE_DATA_INVALID) {
        return {};
    }
    if (Cam::GetProfilePrivateFormatId(payload.header) != kShareToOutputExpertPrivateFormatV1) {
        return {};
    }
    auto stage = static_cast<ProfileStage>(stageId);
    std::ostringstream oss;
    switch (stage) {
        case ProfileStage::ShareToOutputExpert:
            // fromRank：源端 rank；localE：本 rank 内专家索引；count：该 expert 从源端接收的 token 数
            oss << ",\"from_rank\":" << Cam::GetShareToOutputExpertFromRank(payload) << ",\"local_e\":"
                << Cam::GetShareToOutputExpertLocalE(payload) << ",\"recv_token_count\":" << payload.count;
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
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::InputToShare),
                                                         kRoundOccurrenceCapacity),
                     "invalid input to share occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SetStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid set status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WaitStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid wait status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::ShareToOutputCommon),
                                                         kRoundOccurrenceCapacity),
                     "invalid share to output common occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::ShareToOutputExpert),
                                                         kRoundOccurrenceCapacity),
                     "invalid share to output expert occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::SetRoundStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid set round status occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WaitRoundStatus),
                                                         kRoundOccurrenceCapacity),
                     "invalid wait round status occurrence capacity.");
    return layout;
}

}  // namespace deep_ep::profiling::cam_moe_dispatch_normal
