/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_common.h
 * \brief
 */
#ifndef MOE_V3_COMMON_H_REGBASE
#define MOE_V3_COMMON_H_REGBASE

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
#define OVERFLOW_MODE_CTRL 60
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t GATHER = 0;
constexpr int64_t SCATTER = 1;

constexpr uint16_t FLOAT_REG_TENSOR_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr int64_t KEY_VALUE_MODE = 2LL;
constexpr int64_t KEY_VALUE_MODE_DIM_NUM = 2LL;
constexpr int64_t CUMSUM_MODE = 0LL;
constexpr int64_t COUNT_MODE = 1LL;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}
template <HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    SetFlag<event>(EVENT_ID7);
    WaitFlag<event>(EVENT_ID7);
}

// 将 UB 中的专家序号(fp32)中小于 cmpScalar(=expertStart_)的项替换为 MIN_FP32 哨兵, 使排序时沉底。
// __simd_vf__ 函数头标记取代原先内嵌的 __VEC_SCOPE__ 包裹块(950/dav_3510 现行写法)。
__simd_vf__ inline void VfTagOutOfRangeExpert(__local_mem__ float *inUbAddr, uint16_t repeatTimes, uint32_t sreg,
                                              float cmpScalar)
{
    Reg::MaskReg maskRegLoop, cmpMaskReg;
    Reg::MaskReg pregMain = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::RegTensor<float> inRegToFloat, infFloat, vDstReg0;
    Reg::Duplicate(infFloat, MIN_FP32, pregMain);

    float negone = static_cast<float>(-1);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskRegLoop = Reg::UpdateMask<float>(sreg);
        Reg::DataCopy(inRegToFloat, inUbAddr + i * FLOAT_REG_TENSOR_LENGTH);
        Reg::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, inRegToFloat, cmpScalar, maskRegLoop);
        Reg::Muls(inRegToFloat, inRegToFloat, negone, maskRegLoop);
        Reg::Select(vDstReg0, infFloat, inRegToFloat, cmpMaskReg);
        Reg::DataCopy(inUbAddr + i * FLOAT_REG_TENSOR_LENGTH, vDstReg0, maskRegLoop);
    }
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_COMMON_H_REGBASE
