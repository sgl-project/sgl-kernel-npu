/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file apply_top_k_top_p_min_p_kernel.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../op_host/tiling/apply_top_k_top_p_min_p_tiling_data.h"

using namespace AscendC;
using namespace sglang::ATKTPMPHost;
#define TTILING_FP32_WITHOUT_MIN_P 10
#define TTILING_FP16_WITHOUT_MIN_P 20
#define TTILING_BF16_WITHOUT_MIN_P 30
#define TTILING_FP32_MIN_P 11
#define TTILING_FP16_MIN_P 21
#define TTILING_BF16_MIN_P 31

namespace sglang::npu_kernel::ApplyTopKTopPMinPKernel {
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T, int64_t IsMinPSampling>
class ApplyTopKTopPMinP {
public:
    __aicore__ inline ApplyTopKTopPMinP(){};
    __aicore__ inline void Init(
        const __gm__ ApplyTopKTopPMinPTilingData *tilingData, GM_ADDR probs, GM_ADDR k,
        GM_ADDR p, GM_ADDR min_p, GM_ADDR sampled_res, GM_ADDR workspace, TPipe *tPipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyInWithCast(
        LocalTensor<float>& localTensor, GlobalTensor<T>& globalTensor, int64_t gmOffset, int64_t copyNum);
    __aicore__ inline void InitWorkspace(LocalTensor<float>& workLocal);
    __aicore__ inline void GetFloatValue(GlobalTensor<T>& globalTensor, int64_t offset, float& value);
    __aicore__ inline void CumSumImpl(LocalTensor<float>& cumSumInput1Local, LocalTensor<float>& cumSumInput2Local);
    __aicore__ inline void CopyOutWithCast(
        GlobalTensor<T>& globalTensor, LocalTensor<float>& localTensor, int64_t gmOffset, int64_t copyNum);
    __aicore__ inline void TopPProcess(
        LocalTensor<float>& cumSumInput1Local, LocalTensor<float>& cumSumInput2Local, LocalTensor<float>& zeroLocal,
        LocalTensor<uint8_t>& maskLocal);
    __aicore__ inline void MinPProcess(
        LocalTensor<float>& workLocal, LocalTensor<float>& zeroLocal, LocalTensor<uint8_t>& maskLocal);
private:
    TBuf<TPosition::VECCALC> calBuf_;

    // tilingData
    int64_t batchSize_ = 0;
    int64_t vocabSize_ = 0;
    int64_t batchPerCore_ = 0;
    int64_t batchTailCore_ = 0;
    uint32_t blockIdx_ = 0;
    int64_t coreBatch_ = 0;
    int64_t batchOffset_ = 0;
    int64_t coreNum_ = 0;
    int64_t iterateTimes_ = 0;
    int64_t baseGmOffset_ = 0;
    int64_t probsGmOffset_ = 0;

    int32_t kValue_ = 0;
    float pValue_ = 0;
    float minPValue_ = 0;
    float maxValue_ = 0;
    float minPThresholds_ = 0;
    int64_t lastIndex_ = 0;
    int64_t kLoopNum_ = 0;
    int64_t kTailNum_ = 0;
    int64_t loopDataNum_ = 0;

    GlobalTensor<T> gmProbs_;
    GlobalTensor<int32_t> gmK_;
    GlobalTensor<T> gmP_;
    GlobalTensor<T> gmMinP_;
    GlobalTensor<T> gmSampledRes_;

    GlobalTensor<float> gmWk_;
};

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::Init(
    const __gm__ ApplyTopKTopPMinPTilingData *tilingData, GM_ADDR probs, GM_ADDR k,
    GM_ADDR p, GM_ADDR min_p, GM_ADDR sampled_res, GM_ADDR workspace, TPipe *tPipe)
{
    batchSize_ = tilingData->batchSize;
    vocabSize_ = tilingData->vocabSize;
    batchPerCore_ = tilingData->batchPerCore;
    batchTailCore_ = tilingData->batchTailCore;
    coreNum_ = tilingData->coreNum;
    uint32_t ubSize = static_cast<uint32_t>(tilingData->ubSize);
    loopDataNum_ = tilingData->loopDataNum;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= coreNum_) {
        return;
    }

    if (blockIdx_ < batchTailCore_) {
        coreBatch_ = batchPerCore_ + 1;
        batchOffset_ = blockIdx_ * coreBatch_;
    } else {
        coreBatch_ = batchPerCore_;
        batchOffset_ = blockIdx_ * batchPerCore_ + batchTailCore_;
    }
    baseGmOffset_ = batchOffset_ * vocabSize_;

    if (coreBatch_ == 0) {
        return;
    }

    gmProbs_.SetGlobalBuffer((__gm__ T *)probs + baseGmOffset_);
    gmK_.SetGlobalBuffer((__gm__ int32_t *)k + batchOffset_);
    gmP_.SetGlobalBuffer((__gm__ T *)p + batchOffset_);
    if constexpr (IsMinPSampling == 1) {
        gmMinP_.SetGlobalBuffer((__gm__ T *)min_p + batchOffset_);
    }
    gmSampledRes_.SetGlobalBuffer((__gm__ T *)sampled_res + baseGmOffset_);
    gmWk_.SetGlobalBuffer((__gm__ float *)workspace + baseGmOffset_);
    InitGlobalMemory(gmSampledRes_, coreBatch_ * vocabSize_, T(0));
    InitGlobalMemory(gmWk_, coreBatch_ * vocabSize_, float(0));
    tPipe->InitBuffer(calBuf_, ubSize);
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::GetFloatValue(
    GlobalTensor<T>& globalTensor, int64_t offset, float& value)
{
    if constexpr (IsSameType<T, float>::value) {
        value = globalTensor.GetValue(offset);
    } else if constexpr (IsSameType<T, half>::value) {
        value = static_cast<float>(globalTensor.GetValue(offset));
    } else {
        value = ToFloat(globalTensor.GetValue(offset));
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::Process()
{
    if (blockIdx_ >= coreNum_) {
        return;
    }
    uint32_t bufOffset = 0;
    LocalTensor<uint8_t> maskLocal = calBuf_.GetWithOffset<uint8_t>(loopDataNum_ / 8, bufOffset);
    bufOffset += loopDataNum_ / 8 * sizeof(uint8_t);
    LocalTensor<float> zeroLocal = calBuf_.GetWithOffset<float>(loopDataNum_, bufOffset);
    bufOffset += loopDataNum_ * sizeof(float);
    LocalTensor<float> cumSumInput1Local = calBuf_.GetWithOffset<float>(loopDataNum_, bufOffset);
    bufOffset += loopDataNum_ * sizeof(float);
    LocalTensor<float> cumSumInput2Local = calBuf_.GetWithOffset<float>(loopDataNum_, bufOffset);
    bufOffset += loopDataNum_ * sizeof(float);

    Duplicate(zeroLocal, float(0), loopDataNum_);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (int64_t batchLoop = 0; batchLoop < coreBatch_; batchLoop++) {
        probsGmOffset_ = batchLoop * vocabSize_;

        kValue_ = gmK_.GetValue(batchLoop);
        GetFloatValue(gmP_, batchLoop, pValue_);
        if constexpr (IsMinPSampling == 1) {
            GetFloatValue(gmMinP_, batchLoop, minPValue_);
            GetFloatValue(gmProbs_, probsGmOffset_, maxValue_);
            minPThresholds_ = maxValue_ * minPValue_;
        }
        if (kValue_ > vocabSize_ || kValue_ < 0) {
            lastIndex_ = vocabSize_;
        } else {
            lastIndex_ = kValue_;
        }
        kLoopNum_ = lastIndex_ / loopDataNum_;
        kTailNum_ = lastIndex_ - kLoopNum_ * loopDataNum_;

        InitWorkspace(cumSumInput1Local);

        TopPProcess(cumSumInput1Local, cumSumInput2Local, zeroLocal, maskLocal);

        if constexpr (IsMinPSampling == 1) {
            MinPProcess(cumSumInput1Local, zeroLocal, maskLocal);
        }
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::CopyInWithCast(
    LocalTensor<float>& localTensor, GlobalTensor<T>& globalTensor, int64_t gmOffset, int64_t copyNum)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(localTensor, globalTensor[gmOffset],
                    {1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    } else {
        DataCopyPad(localTensor.ReinterpretCast<T>()[loopDataNum_], globalTensor[gmOffset],
                    {1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Cast(localTensor, localTensor.ReinterpretCast<T>()[loopDataNum_], RoundMode::CAST_NONE, copyNum);
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::InitWorkspace(LocalTensor<float>& workLocal)
{
    for (int64_t vocabLoop = 0; vocabLoop < kLoopNum_; vocabLoop++) {
        CopyInWithCast(workLocal, gmProbs_, probsGmOffset_ + vocabLoop * loopDataNum_, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        } else {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }
        DataCopyPad(gmWk_[probsGmOffset_ + vocabLoop * loopDataNum_], workLocal,
                    {1, static_cast<uint32_t>(loopDataNum_ * sizeof(float)), 0, 0, 0});
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    if (kTailNum_ > 0) {
        CopyInWithCast(workLocal, gmProbs_, probsGmOffset_ + kLoopNum_ * loopDataNum_, kTailNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        } else {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }
        DataCopyPad(gmWk_[probsGmOffset_ + kLoopNum_ * loopDataNum_], workLocal,
                    {1, static_cast<uint32_t>(kTailNum_ * sizeof(float)), 0, 0, 0});
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::CumSumImpl(
    LocalTensor<float>& cumSumInput1Local, LocalTensor<float>& cumSumInput2Local)
{
    int64_t tmpValue = 1;
    iterateTimes_ = 0;
    while (tmpValue < lastIndex_) {
        tmpValue <<= 1;
        iterateTimes_++;
    }
    for (int64_t iterateTime = 0; iterateTime < iterateTimes_; iterateTime++) {
        int64_t iteratOffset = 1;
        for (int64_t powerIdx = 0; powerIdx < iterateTime; powerIdx++) {
            iteratOffset = iteratOffset * 2;
        }
        int64_t addLength = lastIndex_ - iteratOffset;
        int64_t innerLoopNum = addLength / loopDataNum_;
        int64_t dataTail = addLength - innerLoopNum * loopDataNum_;
        for (int64_t innerLoop = 0; innerLoop < innerLoopNum; innerLoop++) {
            int64_t innerLoopOffset = dataTail + (innerLoopNum - 1 - innerLoop) * loopDataNum_;
            DataCopyPad(cumSumInput1Local, gmWk_[probsGmOffset_ + innerLoopOffset],
                        {1, static_cast<uint32_t>(loopDataNum_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(cumSumInput2Local, gmWk_[probsGmOffset_ + innerLoopOffset + iteratOffset],
                        {1, static_cast<uint32_t>(loopDataNum_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Add(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, loopDataNum_);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(gmWk_[probsGmOffset_ + innerLoopOffset + iteratOffset], cumSumInput1Local,
                        {1, static_cast<uint32_t>(loopDataNum_ * sizeof(float)), 0, 0, 0});
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
        if (dataTail > 0) {
            DataCopyPad(cumSumInput1Local, gmWk_[probsGmOffset_],
                        {1, static_cast<uint32_t>(dataTail * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(cumSumInput2Local, gmWk_[probsGmOffset_ + iteratOffset],
                        {1, static_cast<uint32_t>(dataTail * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Add(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, dataTail);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(gmWk_[probsGmOffset_ + iteratOffset], cumSumInput1Local,
                        {1, static_cast<uint32_t>(dataTail * sizeof(float)), 0, 0, 0});
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::CopyOutWithCast(
    GlobalTensor<T>& globalTensor, LocalTensor<float>& localTensor, int64_t gmOffset, int64_t copyNum)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(globalTensor[gmOffset], localTensor, {1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0});
    } else {
        Cast(localTensor.ReinterpretCast<T>(), localTensor, RoundMode::CAST_ROUND, copyNum);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(globalTensor[gmOffset], localTensor.ReinterpretCast<T>(),
                    {1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0});
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::TopPProcess(
    LocalTensor<float>& cumSumInput1Local, LocalTensor<float>& cumSumInput2Local, LocalTensor<float>& zeroLocal,
    LocalTensor<uint8_t>& maskLocal)
{
    CumSumImpl(cumSumInput1Local, cumSumInput2Local);
    for (int64_t vocabLoop = 0; vocabLoop < kLoopNum_; vocabLoop++) {
        DataCopyPad(cumSumInput1Local, gmWk_[probsGmOffset_ + vocabLoop * loopDataNum_],
                    {1, static_cast<uint32_t>(loopDataNum_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        CopyInWithCast(cumSumInput2Local, gmProbs_, probsGmOffset_ + vocabLoop * loopDataNum_, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            PipeBarrier<PIPE_V>();
        }
        Sub(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, loopDataNum_);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, cumSumInput1Local, pValue_, CMPMODE::GT, loopDataNum_);
        PipeBarrier<PIPE_V>();
        Select(cumSumInput2Local, maskLocal, zeroLocal, cumSumInput2Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CopyOutWithCast(gmSampledRes_, cumSumInput2Local, probsGmOffset_ + vocabLoop * loopDataNum_, loopDataNum_);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    if (kTailNum_ > 0) {
        DataCopyPad(cumSumInput1Local, gmWk_[probsGmOffset_ + kLoopNum_ * loopDataNum_],
                    {1, static_cast<uint32_t>(kTailNum_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        CopyInWithCast(cumSumInput2Local, gmProbs_, probsGmOffset_ + kLoopNum_ * loopDataNum_, kTailNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            PipeBarrier<PIPE_V>();
        }
        Sub(cumSumInput1Local, cumSumInput1Local, cumSumInput2Local, loopDataNum_);
        PipeBarrier<PIPE_V>();
        CompareScalar(maskLocal, cumSumInput1Local, pValue_, CMPMODE::GT, loopDataNum_);
        PipeBarrier<PIPE_V>();
        Select(cumSumInput2Local, maskLocal, zeroLocal, cumSumInput2Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CopyOutWithCast(gmSampledRes_, cumSumInput2Local, probsGmOffset_ + kLoopNum_ * loopDataNum_, kTailNum_);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}

template <typename T, int64_t IsMinPSampling>
__aicore__ inline void ApplyTopKTopPMinP<T, IsMinPSampling>::MinPProcess(
    LocalTensor<float>& workLocal, LocalTensor<float>& zeroLocal, LocalTensor<uint8_t>& maskLocal)
{
    for (int64_t vocabLoop = 0; vocabLoop < kLoopNum_; vocabLoop++) {
        CopyInWithCast(workLocal, gmSampledRes_, probsGmOffset_ + vocabLoop * loopDataNum_, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CompareScalar(maskLocal, workLocal, minPThresholds_, CMPMODE::LT, loopDataNum_);
        PipeBarrier<PIPE_V>();
        Select(workLocal, maskLocal, zeroLocal, workLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CopyOutWithCast(gmSampledRes_, workLocal, probsGmOffset_ + vocabLoop * loopDataNum_, loopDataNum_);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    if (kTailNum_ > 0) {
        CopyInWithCast(workLocal, gmSampledRes_, probsGmOffset_ + kLoopNum_ * loopDataNum_, kTailNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CompareScalar(maskLocal, workLocal, minPThresholds_, CMPMODE::LT, loopDataNum_);
        PipeBarrier<PIPE_V>();
        Select(workLocal, maskLocal, zeroLocal, workLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, loopDataNum_);
        if constexpr (IsSameType<T, float>::value) {
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        } else {
            PipeBarrier<PIPE_V>();
        }
        CopyOutWithCast(gmSampledRes_, workLocal, probsGmOffset_ + kLoopNum_ * loopDataNum_, kTailNum_);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}
}  // namespace sglang::npu_kernel::ApplyTopKTopPMinPKernel

__global__ __aicore__ void apply_top_k_top_p_min_p(
    GM_ADDR probs, GM_ADDR k, GM_ADDR p, GM_ADDR min_p, GM_ADDR sampled_res,
    GM_ADDR workspace, GM_ADDR tiling)
{
#define INIT_AND_PROCESS                                                   \
    op.Init(tilingData, probs, k, p, min_p, sampled_res, userWS, &tPipe); \
    op.Process();

    AscendC::TPipe tPipe;
    using namespace sglang::npu_kernel::ApplyTopKTopPMinPKernel;

    // __gm__ uint8_t *userWorkspace = workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto tilingData = reinterpret_cast<__gm__ sglang::ATKTPMPHost::ApplyTopKTopPMinPTilingData *>(tiling);
    auto tilingKey = tilingData->tilingKey;
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (tilingKey == TTILING_FP32_WITHOUT_MIN_P) {
            ApplyTopKTopPMinP<float, 0> op;
            INIT_AND_PROCESS;
    } else if (tilingKey == TTILING_FP16_WITHOUT_MIN_P) {
            ApplyTopKTopPMinP<half, 0> op;
            INIT_AND_PROCESS;
    } else if (tilingKey == TTILING_BF16_WITHOUT_MIN_P) {
            ApplyTopKTopPMinP<bfloat16_t, 0> op;
            INIT_AND_PROCESS;
    } else if (tilingKey == TTILING_FP32_MIN_P) {
            ApplyTopKTopPMinP<float, 1> op;
            INIT_AND_PROCESS;
    } else if (tilingKey == TTILING_FP16_MIN_P) {
            ApplyTopKTopPMinP<half, 1> op;
            INIT_AND_PROCESS;
    } else if (tilingKey == TTILING_BF16_MIN_P) {
            ApplyTopKTopPMinP<bfloat16_t, 1> op;
            INIT_AND_PROCESS;
    }
}
