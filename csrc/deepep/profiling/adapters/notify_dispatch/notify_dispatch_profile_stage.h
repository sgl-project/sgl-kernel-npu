#ifndef DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H
#define DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::notify_dispatch {

// notify_dispatch 阶段定义（A5 专用）。
// stageId 顺序必须与 kernel 侧（ops/op_kernel/profiling/adapters/notify_dispatch/...）保持一致。
// 结构说明：AssembleSendData 与 8 个 Build* 是核心分派（仅一个 core 执行），
// InputToShareSlice / ShareToShareSlice 是多核均分分片；无 payload。
enum class ProfileStage : uint32_t {
    AssembleSendData = 0,
    InputToShareSlice = 1,
    ShareToShareSlice = 2,
    BuildTotalRecvTokens = 3,
    BuildRecvCount = 4,
    BuildRecvOffset = 5,
    BuildMaxBs = 6,
    BuildRecvTokenPerExp = 7,
    BuildExpGlobalOffset = 8,
    BuildSrcRankInExpOffset = 9,
    BuildRInSrcrankOffset = 10,
    Count = 11,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::notify_dispatch

#endif  // DEEPEP_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H
