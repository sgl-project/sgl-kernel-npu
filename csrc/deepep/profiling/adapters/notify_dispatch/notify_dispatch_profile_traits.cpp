#include "profiling/adapters/notify_dispatch/notify_dispatch_profile_traits.hpp"

#include <sstream>

#include "exception.hpp"

namespace deep_ep::profiling::notify_dispatch {

namespace {
// notify_dispatch 无 group / 无 per-expert payload。
// 每个 stage 都是「该 core 一条」：核心分派 stage 只有对应 core 产生记录，
// 多核分片 stage（InputToShareSlice/ShareToShareSlice）每核一条。
// 故所有阶段 occurrence 容量均为 1。
constexpr uint32_t kSingleOccurrenceCapacity = 1U;
}  // namespace

const ProfileSchema &GetProfileSchema()
{
    static const ProfileSchema schema{
        "notify_dispatch",
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
        "notify_dispatch",
        &GetProfileSchema,
        &GetLaunchEventName,
    };
    return registration;
}

const char *GetLaunchEventName()
{
    return "notify_dispatch_launch";
}

const char *GetStageName(uint64_t stageId)
{
    switch (static_cast<ProfileStage>(stageId)) {
        case ProfileStage::AssembleSendData:
            return "assemble_send_data";
        case ProfileStage::InputToShareSlice:
            return "input_to_share_slice";
        case ProfileStage::ShareToShareSlice:
            return "share_to_share_slice";
        case ProfileStage::BuildTotalRecvTokens:
            return "build_total_recv_tokens";
        case ProfileStage::BuildRecvCount:
            return "build_recv_count";
        case ProfileStage::BuildRecvOffset:
            return "build_recv_offset";
        case ProfileStage::BuildMaxBs:
            return "build_max_bs";
        case ProfileStage::BuildRecvTokenPerExp:
            return "build_recv_token_per_exp";
        case ProfileStage::BuildExpGlobalOffset:
            return "build_exp_global_offset";
        case ProfileStage::BuildSrcRankInExpOffset:
            return "build_src_rank_in_exp_offset";
        case ProfileStage::BuildRInSrcrankOffset:
            return "build_r_in_srcrank_offset";
        default:
            return "unknown";
    }
}

std::string GetStageDisplayName(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileStageLayout &stageLayout)
{
    (void)occurrenceId;
    (void)stageLayout;
    // 展示名即阶段名（无 group、无 per-expert 后缀）。
    return GetStageName(stageId);
}

std::string GetPrivateDataJson(uint64_t stageId, uint64_t occurrenceId, const Cam::ProfileRecord &record,
                               const Cam::ProfileStageLayout &stageLayout)
{
    (void)stageId;
    (void)occurrenceId;
    (void)record;
    (void)stageLayout;
    // 无 payload。
    return {};
}

Cam::ProfileStageLayout BuildStageLayout()
{
    Cam::ProfileStageLayout layout{};
    layout.stageCount = static_cast<uint16_t>(kStageCount);
    layout.activeStageCapacity = static_cast<uint16_t>(Cam::PROFILE_ACTIVE_STAGE_CAPACITY);
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::AssembleSendData),
                                                         kSingleOccurrenceCapacity),
                     "invalid assemble send data occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::InputToShareSlice),
                                                         kSingleOccurrenceCapacity),
                     "invalid input to share slice occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::ShareToShareSlice),
                                                         kSingleOccurrenceCapacity),
                     "invalid share to share slice occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(
                         layout, static_cast<uint32_t>(ProfileStage::BuildTotalRecvTokens), kSingleOccurrenceCapacity),
                     "invalid build total recv tokens occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::BuildRecvCount),
                                                         kSingleOccurrenceCapacity),
                     "invalid build recv count occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::BuildRecvOffset),
                                                         kSingleOccurrenceCapacity),
                     "invalid build recv offset occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::BuildMaxBs),
                                                         kSingleOccurrenceCapacity),
                     "invalid build max bs occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(
                         layout, static_cast<uint32_t>(ProfileStage::BuildRecvTokenPerExp), kSingleOccurrenceCapacity),
                     "invalid build recv token per exp occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(
                         layout, static_cast<uint32_t>(ProfileStage::BuildExpGlobalOffset), kSingleOccurrenceCapacity),
                     "invalid build exp global offset occurrence capacity.");
    EP_HOST_ASSERT_S(
        Cam::SetProfileStageOccurrenceCount(layout, static_cast<uint32_t>(ProfileStage::BuildSrcRankInExpOffset),
                                            kSingleOccurrenceCapacity),
        "invalid build src rank in exp offset occurrence capacity.");
    EP_HOST_ASSERT_S(Cam::SetProfileStageOccurrenceCount(
                         layout, static_cast<uint32_t>(ProfileStage::BuildRInSrcrankOffset), kSingleOccurrenceCapacity),
                     "invalid build r in srcrank offset occurrence capacity.");
    return layout;
}

}  // namespace deep_ep::profiling::notify_dispatch
