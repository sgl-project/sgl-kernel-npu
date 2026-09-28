#ifndef FUSED_DEEP_MOE_UTILS_H
#define FUSED_DEEP_MOE_UTILS_H

#include <kernel_operator.h>
#include "../fused_deep_moe_a5_base.h"

namespace CVSoftSync {
constexpr uint32_t SOFT_SYNC_SPACE_SIZE = 512;
}

namespace FusedDeepMoeSync {
constexpr uint64_t GROUP_TOKEN_NUM_OFFSET = 932 * 1024;
constexpr uint32_t GROUP_INFO_SIZE = 32;
constexpr uint32_t INT32_COUNT_PER_BLOCK = 32 / sizeof(int32_t);
constexpr uint32_t X2_READY_SLOT_SIZE = 512;
constexpr uint32_t X2_READY_MAX_ROUTED_EXPERTS = 256;
constexpr uint32_t X2_READY_SIZE = X2_READY_SLOT_SIZE * X2_READY_MAX_ROUTED_EXPERTS;
constexpr uint32_t X2_READY_COUNTER_INDEX = 0;
static_assert(X2_READY_SLOT_SIZE % INT32_COUNT_PER_BLOCK == 0);
static_assert(X2_READY_SIZE == 128 * 1024);
constexpr uint32_t SHARED_X2_DONE_COUNT_INDEX = 16;
static_assert(GROUP_INFO_SIZE % INT32_COUNT_PER_BLOCK == 0);
static_assert(SHARED_X2_DONE_COUNT_INDEX % INT32_COUNT_PER_BLOCK == 0);
static_assert(SHARED_X2_DONE_COUNT_INDEX + INT32_COUNT_PER_BLOCK <= GROUP_INFO_SIZE);
}  // namespace FusedDeepMoeSync

template <typename T>
__aicore__ inline T FlushAndGetValue(AscendC::GlobalTensor<T> &globalTensor, uint64_t index)
{
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        globalTensor[index]);
    __asm__ __volatile__("");
    T value = globalTensor.GetValue(index);
    return value;
}

template <typename T>
__aicore__ inline void SetValueAndFlush(AscendC::GlobalTensor<T> &globalTensor, uint64_t index, T value)
{
    globalTensor.SetValue(index, value);
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        globalTensor[index]);
    __asm__ __volatile__("");
}

template <typename T>
__aicore__ inline T FlushAndSpinValue(AscendC::GlobalTensor<T> &globalTensor, uint64_t index)
{
    T value = FlushAndGetValue(globalTensor, index);
    if (value == 0) {
        SetValueAndFlush(globalTensor, index, 1);
    } else {
        SetValueAndFlush(globalTensor, index, 0);
    }
    return value;
}

__aicore__ inline void EncreaseSyncFlag(__gm__ int32_t *flagAddr, int32_t idx)
{
    // flag++, like set flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(flagAddr + idx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t));
    int32_t value = FlushAndGetValue<int32_t>(global, 0);
    SetValueAndFlush<int32_t>(global, 0, value + 1);
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CheckSyncFlag(__gm__ int32_t *flagAddr, int32_t idx, uint32_t target)
{
    //  check flag, like wait flag
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<int32_t> global;
    global.SetGlobalBuffer(flagAddr + idx * CVSoftSync::SOFT_SYNC_SPACE_SIZE / sizeof(int32_t));
    while (true) {
        int32_t value = FlushAndGetValue<int32_t>(global, 0);
        if (value >= target) {
            break;
        }
        SPIN_WAIT_CYCLES();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

#endif  // FUSED_DEEP_MOE_UTILS_H
