#ifndef DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_STAGE_H
#define DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_STAGE_H

#include <cstdint>

namespace deep_ep::profiling::cam_moe_combine_normal {

// combine 阶段定义（A3 / A5 / single-round / multi-round 共用一套枚举）
// stageId 顺序与 host 侧 schema（profiling/adapters/cam_moe_combine_normal/...）保持一致。
// 打点方式：发送/接收阶段分别把“本核（本轮）所有 token 的同类耗时”聚合为一条记录（occurrence=0/round）；
// multi-round 额外把轮间 barrier 拆分为 SetRoundStatus / WaitRoundStatus 两条独立记录。
enum class ProfileStage : uint32_t {
    SendCopyToShare = 0,  // 发送：把 token 写入远端 HCCL window（聚合 copy 耗时）
    SendSetStatus = 1,    // 发送：置远端状态字（聚合 status 耗时）
    RecvWaitStatus = 2,   // 接收：轮询等待数据就绪（聚合 wait 耗时）
    RecvReadAndSum = 3,   // 接收：读远端 + topk 加权求和写 XOut（聚合 sum 耗时）
    SetRoundStatus = 4,   // multi-round 轮间 barrier：置本轮完成标志（SetRoundStatus 耗时）
    WaitRoundStatus = 5,  // multi-round 轮间 barrier：轮询等待所有 rank 就绪（WaitRoundStatus 耗时）
    Count = 6,
};

constexpr uint32_t kStageCount = static_cast<uint32_t>(ProfileStage::Count);

}  // namespace deep_ep::profiling::cam_moe_combine_normal

#endif  // DEEPEP_OP_KERNEL_PROFILING_ADAPTERS_CAM_MOE_COMBINE_NORMAL_PROFILE_STAGE_H
