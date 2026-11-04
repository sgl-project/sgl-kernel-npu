#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <stdint.h>

namespace AscendTimer {
// ========== 1. 枚举定义（CPU/NPU共用） ==========
// 固定计时项
// AIV_COMBINE 已移至 DynamicTimingType (per-expert 拆分, 见阶段1);
enum FixedTiming {
    KERNEL_TIMING_IDX = 0,
    AIV_QUANT,
    AIV_INIT_ROUTING,
    AIV_RESET_DATA,
    AIV_ALLGATHER,
    AIC_WAIT_ALLGATHER,
    AIV_POST_COMBINE,
    AIV_UNPERMUTE,
    FIXED_TIMING_COUNT
};

// Per-expert wall-clock envelopes, including waits between repeated segments.
// They are not continuous busy time and must not be summed across pipelines.
enum DynamicTimingType {
    AIV_DISPATCH,
    AIC_GMM1,
    AIC_GMM2,
    AIV_COMBINE,
    AIV_GMM1_EPILOGUE,
    // 共享专家复用同一条 GMM 流水，但必须分槽打点：同名类型下共享专家与
    // 路由专家的 iter 编号会互相覆盖。actual iter = sharedExpertNum，
    // 0 表示本轮未启用共享专家（解析端跳过这些列）。
    AIC_SHARE_GMM1,
    AIC_SHARE_GMM2,
    AIV_SHARE_GMM1_EPILOGUE,
    DYNAMIC_TYPE_COUNT_ENUM
};

// ========== 2. 公共常量定义（CPU/NPU共用） ==========
// Cover the operator's entire expert limit, including BF16's 128 local experts.
// Capacity lives in the optional GM timer buffer, not a per-expert local array.
static constexpr int MAX_DYNAMIC_ITER = 1024;

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每个计时项占 2 个 int64 槽位: [start_cycle, end_cycle]
static constexpr int N_TIMING_ITEM_PER_CORE = FIXED_TIMING_COUNT + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER;
static constexpr int N_TIMING_COUNTER_PER_CORE = N_TIMING_ITEM_PER_CORE * 2;  // 每核原始计数器数
static constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN =
    (N_TIMING_COUNTER_PER_CORE + 15) / 16 *
    16;  // 16倍对齐后每核计数器数 128B对齐 每个核写入N_TIMING_COUNTER_PER_CORE_ALIGN个数
static constexpr int N_CORE_COUNT = 96;  // 核心总数 AIC+AIV核心数量
static constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;  // 总基础计数器数

// 动态迭代次数存储区对齐配置（每核动态项存储区16元素对齐）
static constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;  // 动态项类型总数
static constexpr int DYNAMIC_ITER_PER_CORE_ALIGN =
    (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;  // 每核动态迭代次数区16对齐 128B对齐
static constexpr int DYNAMIC_ITER_TOTAL_SIZE =
    DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;  // 动态迭代次数总存储区大小，每个核cacheline不重叠
}  // namespace AscendTimer
// ========== 4. 计时开关宏 ==========
#ifndef ENABLE_TIMER
#define ENABLE_TIMER 0  // Release builds compile profiling out; pass -DENABLE_TIMER=1 for a profile build.
#endif
#if ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do {                  \
        code;             \
    } while (0)
#else
#define TIMER_BLOCK(code) \
    do {                  \
    } while (0)
#endif

#endif  // ASCEND_TIMER_COMMON_H
