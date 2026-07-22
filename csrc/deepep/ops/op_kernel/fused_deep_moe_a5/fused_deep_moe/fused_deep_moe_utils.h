/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * Description: FusedDeepMoe operator kernel utils function header file, for a3
 * Create: 2026-05-12
 * Note:
 * History: 2026-05-12 create FusedDeepMoe operator kernel utils function header file, for a3
 */
#ifndef FUSED_DEEP_MOE_UTILS_H
#define FUSED_DEEP_MOE_UTILS_H

#include <kernel_operator.h>
#include "../fused_deep_moe_a5_base.h"

namespace CVSoftSync {
constexpr uint32_t SOFT_SYNC_SPACE_SIZE = 512;
}

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
