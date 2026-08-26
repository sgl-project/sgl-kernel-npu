#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::notify_dispatch {

// notify_dispatch 阶段定义（A5 专用）。
// stageId 顺序与 host 侧 schema（profiling/adapters/notify_dispatch/...）保持一致。
// 打点方式：
//   - 本算子是「核心分派 + 多核分片」混合结构：
//       · AssembleSendData 与 8 个 Build* 是核心分派（仅对应一个 core 执行，其余核早退 return），
//         每个任务一个 stage，occurrence 全为 1，只在该 core 上产生记录；
//       · InputToShareSlice / ShareToShareSlice 是多核均分分片（blockIdx < coreNumPerStageX/Y），
//         每核一条，occurrence=1。
//   - 无 per-expert / per-group 附加信息，故无 payload。
enum class ProfileStage : uint32_t {
    AssembleSendData = 0,       // core 0：组装发送数据（核心分派）
    InputToShareSlice = 1,      // 多核分片：把本 rank 输入搬进 share
    ShareToShareSlice = 2,      // 多核分片：share → share 拷贝
    BuildTotalRecvTokens = 3,   // core 0：归约总接收 token 数（核心分派）
    BuildRecvCount = 4,         // core 1：前缀和接收计数（核心分派）
    BuildRecvOffset = 5,        // core 2：接收偏移（核心分派）
    BuildMaxBs = 6,             // core 3：最大 batch size（核心分派）
    BuildRecvTokenPerExp = 7,   // core 4：每 expert 接收 token 数（核心分派）
    BuildExpGlobalOffset = 8,   // core 5：expert 全局偏移（核心分派）
    BuildSrcRankInExpOffset = 9,// core 6：expert 内源 rank 偏移（核心分派）
    BuildRInSrcrankOffset = 10, // core 7：源 rank 内偏移（核心分派）
    Count = 11,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::notify_dispatch

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_NOTIFY_DISPATCH_PROFILE_STAGE_H