#include "profiling/adapters/fused_deep_moe_a5/fused_deep_moe_a5_profile_traits.hpp"

#include <sstream>

#include "exception.hpp"

namespace deep_ep::profiling::fused_deep_moe_a5 {

namespace {
constexpr uint8_t kDispatchSendPrivateFormatV1 = 1U;
constexpr uint8_t kDispatchRecvPrivateFormatV1 = 1U;
}  // namespace

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "fused_deep_moe",
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
        "fused_deep_moe_a5",
        &GetProfileSchema,
        &GetLaunchEventName,
    };
    return registration;
}

const char *GetLaunchEventName()
{
    return "fused_deep_moe_launch";
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::DispatchSend:
            return "dispatch_send";
        case ProfileStage::DispatchRecv:
            return "dispatch_recv";
        case ProfileStage::Gmm1:
            return "gmm1";
        case ProfileStage::Swiglu:
            return "swiglu";
        case ProfileStage::Quant:
            return "quant";
        case ProfileStage::StageBarrier:
            return "stage_barrier";
        case ProfileStage::Gmm2:
            return "gmm2";
        case ProfileStage::Combine:
            return "combine";
        case ProfileStage::WeightSumAllToAllSend:
            return "weight_sum_all_to_all_send";
        case ProfileStage::WeightSumReducePermute:
            return "weight_sum_reduce_permute";
        case ProfileStage::WeightSumClean:
            return "weight_sum_clean";
        default:
            return "unknown";
    }
}

std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout)
{
    std::ostringstream oss;
    oss << GetStageName(stageId);
    uint32_t stageOccurrenceCount = Cam::GetProfileStageOccurrenceCount(stageLayout, static_cast<uint32_t>(stageId));
    auto stage = static_cast<ProfileStage>(stageId);
    if (stage == ProfileStage::DispatchRecv || stage == ProfileStage::Gmm1 || stage == ProfileStage::Swiglu ||
        stage == ProfileStage::Quant || stage == ProfileStage::Gmm2 || stage == ProfileStage::Combine) {
        oss << "[group=" << occurrenceId << "]";
    } else if (stageOccurrenceCount > 1U || occurrenceId != 0U) {
        oss << "[occ=" << occurrenceId << "]";
    }
    return oss.str();
}

std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    auto stage = static_cast<ProfileStage>(stageId);
    if (stage == ProfileStage::DispatchSend) {
        const auto payload = Cam::AsDispatchSendPrivatePayloadV1(record);
        if (Cam::GetProfilePrivateValidTag(payload.header) == Cam::PROFILE_PRIVATE_DATA_INVALID) {
            return {};
        }
        if (Cam::GetProfilePrivateFormatId(payload.header) != kDispatchSendPrivateFormatV1) {
            return {};
        }
        std::ostringstream oss;
        oss << ",\"dispatch_send_valid_token_count\":" << payload.validTokenCount;
        oss << ",\"dispatch_send_per_token_bytes\":" << payload.perTokenCommBytes;
        return oss.str();
    }
    if (stage == ProfileStage::DispatchRecv) {
        const auto payload = Cam::AsDispatchRecvPrivatePayloadV1(record);
        if (Cam::GetProfilePrivateValidTag(payload.header) == Cam::PROFILE_PRIVATE_DATA_INVALID) {
            return {};
        }
        if (Cam::GetProfilePrivateFormatId(payload.header) != kDispatchRecvPrivateFormatV1) {
            return {};
        }
        std::ostringstream oss;
        oss << ",\"dispatch_recv_aiv_token_count\":" << payload.aivTokenCount;
        return oss.str();
    }
    return {};
}

Cam::ProfileStageLayout BuildStageLayout(uint32_t groupCountCapacity)
{
    EP_HOST_ASSERT_S(groupCountCapacity >= 1U && groupCountCapacity <= Cam::PROFILE_MAX_GROUP_COUNT_CAPACITY,
                     "groupCountCapacity must be in [1, 64].");
    Cam::ProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(kStageCount);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::PROFILE_ACTIVE_STAGE_CAPACITY);
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::DispatchSend), 1U),
                     "invalid dispatch send occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::DispatchRecv),
                                                         groupCountCapacity),
                     "invalid dispatch receive occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Gmm1), groupCountCapacity),
        "invalid gmm1 occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Swiglu), groupCountCapacity),
        "invalid swiglu occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Quant), groupCountCapacity),
        "invalid quant occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::StageBarrier), 1U),
                     "invalid stage barrier occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Gmm2), groupCountCapacity),
        "invalid gmm2 occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::Combine), groupCountCapacity),
        "invalid combine occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WeightSumAllToAllSend), 1U),
        "invalid weight sum all-to-all send occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WeightSumReducePermute), 1U),
        "invalid weight sum reduce permute occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::WeightSumClean), 1U),
        "invalid weight sum clean occurrence capacity.");
    return layout;
}

uint32_t GetGroupCountCapacity(int64_t numExperts, int64_t numRanks)
{
    EP_HOST_ASSERT_S(numRanks > 0, "num_ranks must be positive for fused deep moe profiling.");
    EP_HOST_ASSERT_S(numExperts > 0, "num_experts must be positive for fused deep moe profiling.");
    EP_HOST_ASSERT_S(numExperts % numRanks == 0,
                     "num_experts must be divisible by num_ranks for fused deep moe profiling.");
    return static_cast<uint32_t>(numExperts / numRanks);
}

}  // namespace deep_ep::profiling::fused_deep_moe_a5
