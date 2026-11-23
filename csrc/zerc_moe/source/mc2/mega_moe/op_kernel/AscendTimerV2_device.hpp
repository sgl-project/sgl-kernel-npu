#ifndef ASCEND_TIMER_DEVICE_H
#define ASCEND_TIMER_DEVICE_H

#include "AscendTimerV2.hpp"

#define ASCEND_TIMER_TOTAL_BUFFER_SIZE (AscendTimer::N_TIMING_COUNTER + AscendTimer::DYNAMIC_ITER_TOTAL_SIZE)

// ========== 1. NPU端专属类型定义 ==========
typedef struct AccumulateTag_ Accumulate;
struct AccumulateTag_ {};
typedef struct OverwriteTag_ Overwrite;
struct OverwriteTag_ {};
struct Span {};

// AscendTimerDevice: 完整实现仅在 ENABLE_TIMER 时编译进内核; release (ENABLE_TIMER=0)
// 下替换为空 stub (零字节、全 no-op), 让 timer/timerAll 成员可无条件声明。真实调用均在
// TIMER_BLOCK 内, release 下展开为空, stub 方法只为编译通过而存在。
#if ENABLE_TIMER
class AscendTimerDevice
{
public:
    // ========== 2. 成员变量 ==========
    GM_ADDR t_addr;
    int64_t start_cycle;
    int64_t dynamic_start_cycles[AscendTimer::DYNAMIC_TYPE_COUNT];
    int dynamic_start_iters[AscendTimer::DYNAMIC_TYPE_COUNT];
    bool dynamic_valid[AscendTimer::DYNAMIC_TYPE_COUNT];
    int block_id;
    int r;
    int group_id;
    int subblockid;
    int core_id;  // 当前核心ID

    // ========== 3. 构造函数 ==========
    __aicore__ inline AscendTimerDevice()
    {
        t_addr = NULL;
        start_cycle = 0;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = 0;
        r = 0;
        group_id = 0;
        subblockid = 0;
        core_id = 0;
    }

    __aicore__ inline AscendTimerDevice(GM_ADDR timeAddr)
    {
        t_addr = timeAddr;
        start_cycle = 0;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = AscendC::GetBlockIdx();
        r = AscendC::GetTaskRation();
        group_id = block_id;
        core_id = block_id;

        if ASCEND_IS_AIV {
            group_id = group_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            core_id = group_id * 3 + subblockid + 1;  // {1,2}, {4,5}
        } else {
            subblockid = 0;
            core_id = group_id * 3;  // 0,3,6
        }
    }

    // ========== 4. 拷贝构造 ==========
    __aicore__ inline AscendTimerDevice(const AscendTimerDevice &other)
    {
        t_addr = other.t_addr;
        start_cycle = other.start_cycle;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = other.dynamic_start_cycles[i];
            dynamic_start_iters[i] = other.dynamic_start_iters[i];
            dynamic_valid[i] = other.dynamic_valid[i];
        }
        block_id = other.block_id;
        r = other.r;
        group_id = other.group_id;
        subblockid = other.subblockid;
        core_id = other.core_id;
    }

    // ========== 5. 初始化函数 ==========
    __aicore__ inline void Init(const GM_ADDR timeAddr)
    {
        t_addr = (GM_ADDR)timeAddr;
        for (int i = 0; i < AscendTimer::DYNAMIC_TYPE_COUNT; ++i) {
            dynamic_start_cycles[i] = 0;
            dynamic_start_iters[i] = -1;
            dynamic_valid[i] = false;
        }
        block_id = AscendC::GetBlockIdx();
        r = AscendC::GetTaskRation();
        group_id = block_id;
        core_id = block_id;

        if ASCEND_IS_AIV {
            group_id = group_id / r;
            subblockid = AscendC::GetSubBlockIdx();
            core_id = group_id * 3 + subblockid + 1;
        } else {
            subblockid = 0;
            core_id = group_id * 3;
        }
    }

    __aicore__ inline void Init(const GM_ADDR timeAddr, int64_t startCycle)
    {
        Init(timeAddr);
        start_cycle = startCycle;
    }

    // ========== 6. 基础接口 ==========
    __aicore__ __inline__ GM_ADDR GetTAddr()
    {
        return t_addr;
    }
    __aicore__ __inline__ int64_t GetStartCycle()
    {
        return start_cycle;
    }
    __aicore__ __inline__ int GetCoreId()
    {
        return core_id;
    }  // 获取当前核心ID

    // ========== 7. 设置当前核心指定动态项的实际迭代次数 ==========
    // (zero/zeroDynamicActualIterAll 已移除: timer buffer 由 host 侧 torch.zeros
    //  清零, device 端无需重复初始化。原实现零调用, 删除以减少死代码。)
    __aicore__ __inline__ void setDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t iter)
    {
        if (t_addr == nullptr) {
            return;
        }
        // 校验核心ID和动态项类型
        if (core_id < 0 || core_id >= AscendTimer::N_CORE_COUNT) {
            AscendC::printf("[WARNING] NPU端核心ID %d 超出范围（0-%d）\n", core_id, AscendTimer::N_CORE_COUNT - 1);
            return;
        }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT) {
            AscendC::printf("[WARNING] NPU端动态项类型 %d 超出范围（0-%d）\n", (int)type,
                            AscendTimer::DYNAMIC_TYPE_COUNT - 1);
            return;
        }

        int32_t idx = AscendTimer::N_TIMING_COUNTER + core_id * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + (int)type;

        // 校验总偏移
        if (idx >= ASCEND_TIMER_TOTAL_BUFFER_SIZE) {
            AscendC::printf("[WARNING] NPU端动态迭代次数索引 %d 超出总缓冲区大小 %d\n", idx,
                            ASCEND_TIMER_TOTAL_BUFFER_SIZE);
            return;
        }

        if (iter < 0 || iter > AscendTimer::MAX_DYNAMIC_ITER) {
            AscendC::printf("[WARNING] Timer type %d has %d experts, capacity %d; trace is incomplete\n", (int)type,
                            static_cast<int32_t>(iter), AscendTimer::MAX_DYNAMIC_ITER);
        }

        // 设置迭代次数
        AscendC::GlobalTensor<int64_t> iterTensor;
        iterTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        // Preserve overflow so the host rejects an incomplete trace, rather
        // than silently treating the unrecorded experts as idle time.
        iterTensor(idx) = iter;
    }

    // ========== 8. 开始计时 ==========
    // (zeroDynamicActualIterAll 已移除: 同 zero, host 侧 torch.zeros 已清零。)
    __aicore__ __inline__ void Tik()
    {
        if (t_addr == nullptr) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        start_cycle = AscendC::GetSystemCycle();
    }

    __aicore__ __inline__ void Tik(int64_t start)
    {
        if (t_addr == nullptr) {
            return;
        }
        start_cycle = start;
    }

    __aicore__ __inline__ void Tik(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) {
            return;
        }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 ||
            iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        int64_t start = AscendC::GetSystemCycle();
        dynamic_start_cycles[(int)type] = start;
        dynamic_start_iters[(int)type] = iter;
        dynamic_valid[(int)type] = true;

        // Do not overwrite the stored first start when beginning another
        // segment. Tok<Span> merges its real endpoints into the zeroed slot.
    }

    // ========== 9. 结束计时 ==========
    // 默认 Overwrite (与 Tok() / Tok(type,iter) 一致); 需累加多段时显式 Tok<Accumulate>(idx)。
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(int32_t timing_idx)
    {
        if (t_addr == nullptr) {
            return;
        }
        if (timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) {
            AscendC::printf("[WARNING] NPU端Tok索引 %d 超出每核最大计时项数 %d\n", timing_idx,
                            AscendTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }
        PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t elapsed_cycle = end_cycle - start_cycle;

        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            accumulate_time(timing_idx, start_cycle, elapsed_cycle);
        } else if constexpr (std::is_same_v<ModeTag, Span>) {
            span_time(timing_idx, start_cycle, end_cycle);
        } else if constexpr (std::is_same_v<ModeTag, Overwrite>) {
            write_time(timing_idx, start_cycle, end_cycle);
        }
    }

    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok()
    {
        Tok<ModeTag>(AscendTimer::KERNEL_TIMING_IDX);
    }

    // 动态计时专用Tok（移除迭代次数更新）

    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(AscendTimer::DynamicTimingType type, int iter)
    {
        if (t_addr == nullptr) {
            return;
        }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 ||
            iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            return;
        }

        // 索引计算
        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;

        // 校验索引
        if (timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) {
            AscendC::printf("[WARNING] NPU端动态项 %d 迭代 %d 索引 %d 超出每核最大计时项数 %d\n", (int)type, iter,
                            timing_idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }

        if (!dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) {
            return;
        }

        PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle();
        int64_t start = dynamic_start_cycles[(int)type];
        int64_t elapsed_cycle = end_cycle - start;

        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            accumulate_time(timing_idx, start, elapsed_cycle);
        } else if constexpr (std::is_same_v<ModeTag, Span>) {
            span_time(timing_idx, start, end_cycle);
        } else if constexpr (std::is_same_v<ModeTag, Overwrite>) {
            write_time(timing_idx, start, end_cycle);
        }
        dynamic_valid[(int)type] = false;
        dynamic_start_iters[(int)type] = -1;
    }

    // ========== 9b. 扣减式结束计时 (无调用方, 保留作工具) ==========
    // 已弃用: 扣减会把 end 前移, timeline 上区间位置失真; 各打点已统一为干活区间口径
    // (掐头去尾, 直接 Tok<Overwrite>)。GetSystemCycle 是单条 MOV SYS_CNT asm, 无需 PipeBar。
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void TokSubtract(AscendTimer::DynamicTimingType type, int iter, int64_t subtractCycles)
    {
        if (t_addr == nullptr) {
            return;
        }
        if ((int)type < 0 || (int)type >= AscendTimer::DYNAMIC_TYPE_COUNT || iter < 0 ||
            iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            return;
        }

        int timing_idx = AscendTimer::FIXED_TIMING_COUNT + (int)type * AscendTimer::MAX_DYNAMIC_ITER + iter;

        if (timing_idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) {
            AscendC::printf("[WARNING] NPU端TokSubtract动态项 %d 迭代 %d 索引 %d 超出每核最大计时项数 %d\n", (int)type,
                            iter, timing_idx, AscendTimer::N_TIMING_ITEM_PER_CORE);
            return;
        }

        if (!dynamic_valid[(int)type] || dynamic_start_iters[(int)type] != iter) {
            return;
        }

        PipeBarrier<PIPE_ALL>();
        int64_t end_cycle = AscendC::GetSystemCycle() - subtractCycles;
        int64_t start = dynamic_start_cycles[(int)type];
        int64_t elapsed_cycle = end_cycle - start;

        if constexpr (std::is_same_v<ModeTag, Accumulate>) {
            accumulate_time(timing_idx, start, elapsed_cycle);
        } else if constexpr (std::is_same_v<ModeTag, Overwrite>) {
            write_time(timing_idx, start, end_cycle);
        }
        dynamic_valid[(int)type] = false;
        dynamic_start_iters[(int)type] = -1;
    }

    // ========== 10. 打印调试信息 ==========
    __aicore__ __inline__ void print()
    {
        AscendC::printf("AscendTimer: core_id=%d, block_id=%d, group_id=%d, subblockid=%d, t_addr=%p\n", core_id,
                        block_id, group_id, subblockid, t_addr);
    }

private:
    // ========== 11. 核内索引计算 ==========
    __aicore__ __inline__ int32_t calculate_index(int32_t idx)
    {
        if (idx >= AscendTimer::N_TIMING_ITEM_PER_CORE) {
            AscendC::printf("[WARNING] NPU端计算索引时，输入idx %d 超出每核最大计时项数 %d\n", idx,
                            AscendTimer::N_TIMING_ITEM_PER_CORE);
            return 0;
        }

        int32_t adjusted_idx = 0;
        if ASCEND_IS_AIV {
            adjusted_idx = group_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 +
                           AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * (1 + subblockid) + idx * 2;
        } else {
            adjusted_idx = group_id * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * 3 + idx * 2;
        }

        if (adjusted_idx + 1 >= AscendTimer::N_TIMING_COUNTER) {
            AscendC::printf("[WARNING] NPU端计算后索引 %d 超出总对齐最大值 %d\n", adjusted_idx,
                            AscendTimer::N_TIMING_COUNTER);
            return 0;
        }

        return adjusted_idx;
    }

    // Keep physical endpoints for timeline events, even with intervening idle
    // time or other experts. Accumulate below is only a duration accumulator.
    __aicore__ __inline__ void span_time(int32_t idx, int64_t start, int64_t end)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        int64_t current_start = timeTensor(adjusted_idx);
        int64_t current_end = timeTensor(adjusted_idx + 1);
        if (current_start == 0 || start < current_start) {
            timeTensor(adjusted_idx) = start;
        }
        if (end > current_end) {
            timeTensor(adjusted_idx + 1) = end;
        }
    }

    // ========== 12. 累加/覆盖计时值 ==========
    __aicore__ __inline__ void accumulate_time(int32_t idx, int64_t start, int64_t time)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        int64_t current_start = timeTensor(adjusted_idx);
        int64_t current_end = timeTensor(adjusted_idx + 1);
        if (current_start == 0 && current_end == 0) {
            timeTensor(adjusted_idx) = start;
            timeTensor(adjusted_idx + 1) = start + time;
        } else {
            timeTensor(adjusted_idx + 1) = current_end + time;
        }
    }

    __aicore__ __inline__ void write_time(int32_t idx, int64_t start, int64_t end)
    {
        int32_t adjusted_idx = calculate_index(idx);
        AscendC::GlobalTensor<int64_t> timeTensor;
        timeTensor.SetGlobalBuffer((__gm__ int64_t *)t_addr);
        timeTensor(adjusted_idx) = start;
        timeTensor(adjusted_idx + 1) = end;
    }
};
#else
// release stub: 与上面完整类同构的方法签名 (含 template Tok), 全 no-op。
// timer/timerAll 成员由此为空对象; 其调用点都在 TIMER_BLOCK 内, release 下不展开。
class AscendTimerDevice
{
public:
    __aicore__ inline AscendTimerDevice() {}
    __aicore__ inline AscendTimerDevice(GM_ADDR timeAddr)
    {
        (void)timeAddr;
    }
    __aicore__ inline AscendTimerDevice(const AscendTimerDevice &other)
    {
        (void)other;
    }
    __aicore__ inline void Init(const GM_ADDR timeAddr)
    {
        (void)timeAddr;
    }
    __aicore__ inline void Init(const GM_ADDR timeAddr, int64_t startCycle)
    {
        (void)timeAddr;
        (void)startCycle;
    }
    __aicore__ __inline__ GM_ADDR GetTAddr()
    {
        return NULL;
    }
    __aicore__ __inline__ int64_t GetStartCycle()
    {
        return 0;
    }
    __aicore__ __inline__ int GetCoreId()
    {
        return 0;
    }
    __aicore__ __inline__ void setDynamicActualIter(AscendTimer::DynamicTimingType type, int64_t iter)
    {
        (void)type;
        (void)iter;
    }
    __aicore__ __inline__ void Tik() {}
    __aicore__ __inline__ void Tik(int64_t start)
    {
        (void)start;
    }
    __aicore__ __inline__ void Tik(AscendTimer::DynamicTimingType type, int iter)
    {
        (void)type;
        (void)iter;
    }
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(int32_t timing_idx)
    {
        (void)timing_idx;
    }
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok()
    {}
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void Tok(AscendTimer::DynamicTimingType type, int iter)
    {
        (void)type;
        (void)iter;
    }
    template <typename ModeTag = Overwrite>
    __aicore__ __inline__ void TokSubtract(AscendTimer::DynamicTimingType type, int iter, int64_t subtractCycles)
    {
        (void)type;
        (void)iter;
        (void)subtractCycles;
    }
    __aicore__ __inline__ void print() {}
};
#endif  // ENABLE_TIMER

#endif  // ASCEND_TIMER_DEVICE_H
