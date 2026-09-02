#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_STAGE_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::cam_moe_dispatch_normal {

// normal dispatch 阶段定义（A3 / A5 共用一套枚举）
// stageId 顺序与 host 侧 schema（profiling/adapters/cam_moe_dispatch_normal/...）保持一致。
// 打点方式：
//   - 发送阶段（InputToShare / SetStatus）与接收等待阶段（WaitStatus）：对函数整体计时，每轮一条；
//   - ShareToOutputLongSeq 拆分为两段：
//       · ShareToOutputCommon：读远端前的公共准备部分（expertGlobalOffset/srcrankInExpertOffset/rInSrcrankOffset
//         加载、tokenInParams 等），每轮一条；
//       · ShareToOutputExpert：按“每个 expert 一条”打点，occurrenceId 编码轮次与本核内 expert 序号，
//         payload 携带源端 rank（fromRank）、本 rank 内专家索引（localE）与接收 token 数（count）；
//   - multi-round 额外把轮间 barrier 拆分为 SetRoundStatus / WaitRoundStatus 两条独立记录。
enum class ProfileStage : uint32_t {
    InputToShare = 0,         // 发送：把 token 写入远端 HCCL window（函数整体计时）
    SetStatus = 1,            // 发送：置远端状态字（函数整体计时）
    WaitStatus = 2,           // 接收：轮询等待数据就绪（函数整体计时）
    ShareToOutputCommon = 3,  // 接收：读远端前的公共准备部分（每轮一条）
    ShareToOutputExpert = 4,  // 接收：per-expert（每个 expert 一条，payload 带 fromRank/localE/count）
    SetRoundStatus = 5,       // multi-round 轮间 barrier：置本轮完成标志
    WaitRoundStatus = 6,      // multi-round 轮间 barrier：轮询等待所有 rank 就绪
    Count = 7,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::cam_moe_dispatch_normal

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_DISPATCH_NORMAL_PROFILE_STAGE_H
