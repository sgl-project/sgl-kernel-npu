/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_shrink.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMV_SHRINK_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMV_SHRINK_H

#include "kernel_operator.h"

template <typename scalar_t>
class SGEMMVShrink
{
public:
    using X_T = scalar_t;
    using W_T = scalar_t;
    using Y_T = float;

    static constexpr uint64_t BUFFER_NUM = 1;
    static constexpr uint64_t TILE_LENGTH = 11776;  // optimal performance tile length

public:
    __aicore__ inline SGEMMVShrink(AscendC::TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR loraScales, uint32_t loraScalesSize, GM_ADDR y, uint32_t batchSize,
                                uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        inputHiddenDim_ = inputHiddenDim;
        maxLoRARank_ = maxLoRARank;
        singleLoRAWeightLen_ = inputHiddenDim_ * maxLoRARank_;
        incremental_ = inputHiddenDim_ > TILE_LENGTH;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(y));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        loraScalesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(loraScales), loraScalesSize);

        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, TILE_LENGTH * sizeof(X_T));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, TILE_LENGTH * sizeof(W_T));
        pipe_->InitBuffer(tmpBufferX_, TILE_LENGTH * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, TILE_LENGTH * sizeof(float));

        pipe_->InitBuffer(outQueueY_, 1, maxLoRARank_ * sizeof(Y_T));
        pipe_->InitBuffer(outBufferY_, maxLoRARank_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t startIdx = blockIdx * numTokensPerCore_;
        int64_t endIdx = startIdx + numTokensPerCore_;
        if (endIdx > batchSize_) {
            endIdx = batchSize_;
        }
        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            // set up LoRA index
            CopyInIndex(idx);
            if (reqLoRAIndex_ < 0) {
                continue;
            }

            reqLoRAWeightOffset_ = reqLoRAIndex_ * singleLoRAWeightLen_;
            reqLoRARank_ = loraRanksGm_.GetValue(reqLoRAIndex_);

            if (reqLoRARank_ == 0) {
                continue;
            }

            reqLoRAScale_ = loraScalesGm_.GetValue(reqLoRAIndex_);

            if (incremental_) {
                ProcessImpl<true>(idx);
            } else {
                ProcessImpl<false>(idx);
            }

            ScaleOutput();
            CopyOut(idx);
        }
    }

private:
    template <bool INCREMENTAL_MODE>
    __aicore__ inline void ProcessImpl(const int64_t idx)
    {
        AscendC::LocalTensor<float> yOutLocal = outBufferY_.Get<float>();
        if constexpr (!INCREMENTAL_MODE) {
            CopyInX(idx, 0, inputHiddenDim_);
            AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, inputHiddenDim_);
            pipe_barrier(PIPE_V);
            inQueueX_.FreeTensor(xLocal);
        }

        for (int i = 0; i < reqLoRARank_; i++) {
            float acc(0);
            for (int32_t j = 0; j < inputHiddenDim_ / TILE_LENGTH; j++) {
                if constexpr (INCREMENTAL_MODE) {
                    CopyInX(idx, j);
                }
                CopyInW(i, j);
                Compute<INCREMENTAL_MODE>(acc);
            }
            CopyAndComputeLastIteration<INCREMENTAL_MODE>(idx, i, acc);
            yOutLocal.SetValue(i, acc);
        }
    }

    __aicore__ inline void CopyInIndex(const int64_t idx)
    {
        // look up the LoRA index
        int64_t weightIdx = idx;
        uint64_t i = 0;
        for (; i < seqLenGm_.GetSize(); i++) {
            int64_t repeatValue = seqLenGm_.GetValue(i);
            if (weightIdx >= repeatValue) {
                weightIdx -= repeatValue;
                continue;
            }
            break;
        }
        reqLoRAIndex_ = (i < seqLenGm_.GetSize()) ? loraIndicesGm_.GetValue(i) : -1;
    }

    __aicore__ inline void CopyInX(const int64_t idx, int32_t colIdx, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();
        DataCopy(xLocal, xGm_[inputHiddenDim_ * idx + colIdx * TILE_LENGTH], numElements);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInW(int32_t rowIdx, int32_t colIdx, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wGm_[reqLoRAWeightOffset_ + rowIdx * inputHiddenDim_ + colIdx * TILE_LENGTH], numElements);
        inQueueW_.EnQue(wLocal);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void Compute(float &acc, int32_t numElements = TILE_LENGTH)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> xTmpTensor = tmpBufferX_.Get<float>();
        AscendC::LocalTensor<float> wTmpTensor = tmpBufferW_.Get<float>();

        if constexpr (INCREMENTAL_MODE) {
            AscendC::LocalTensor<X_T> xLocal = inQueueX_.DeQue<X_T>();
            Cast(xTmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, numElements);
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            pipe_barrier(PIPE_V);
            inQueueX_.FreeTensor(xLocal);
            inQueueW_.FreeTensor(wLocal);
        } else {
            Cast(wTmpTensor, wLocal, AscendC::RoundMode::CAST_NONE, numElements);
            pipe_barrier(PIPE_V);
            inQueueW_.FreeTensor(wLocal);
        }
        // dot product of the one tile of X and W
        Mul(wTmpTensor, xTmpTensor, wTmpTensor, numElements);
        pipe_barrier(PIPE_V);
        // reduce sum generate one number, which is the summation of all the dot product
        ReduceSum<float>(wTmpTensor, wTmpTensor, wTmpTensor, numElements);
        pipe_barrier(PIPE_V);

        acc += wTmpTensor.GetValue(0);
    }

    template <bool INCREMENTAL_MODE>
    __aicore__ inline void CopyAndComputeLastIteration(const int64_t idx, int32_t rowIdx, float &acc)
    {
        int32_t colIdx = inputHiddenDim_ / TILE_LENGTH;
        int32_t remaining = inputHiddenDim_ % TILE_LENGTH;
        if (remaining == 0) {
            return;
        }
        if constexpr (INCREMENTAL_MODE) {
            CopyInX(idx, colIdx, remaining);
        }
        CopyInW(rowIdx, colIdx, remaining);
        Compute<INCREMENTAL_MODE>(acc, remaining);
    }

    __aicore__ inline void ScaleOutput()
    {
        AscendC::LocalTensor<float> yLocal = outBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();

        Muls(yOutLocal, yLocal, reqLoRAScale_, reqLoRARank_);
        pipe_barrier(PIPE_V);

        outQueueY_.EnQue<Y_T>(yOutLocal);
    }

    __aicore__ inline void CopyOut(const int64_t idx)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        DataCopy(yOutGm_[maxLoRARank_ * idx], yOutLocal, reqLoRARank_);
        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_, inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferX_, tmpBufferW_, outBufferY_;
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<half> loraScalesGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;
    uint32_t batchSize_;
    uint32_t numTokensPerCore_;
    uint32_t inputHiddenDim_;
    uint32_t maxLoRARank_;
    uint32_t singleLoRAWeightLen_;

    uint64_t reqLoRAWeightOffset_;
    int32_t reqLoRAIndex_;
    int32_t reqLoRARank_;
    float reqLoRAScale_;

    bool incremental_;
};

#define SGEMMV_SHRINK_TYPE_DECLARE(TYPE)                                                                               \
    extern "C" __global__ __aicore__ void sgemmv_shrink_##TYPE(                                                        \
        GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize, \
        GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR loraScales, uint32_t loraScalesSize, GM_ADDR y,             \
        uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank)                  \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SGEMMVShrink<TYPE> op(&pipe);                                                                                  \
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, loraScales,     \
                loraScalesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank);                          \
        op.Process();                                                                                                  \
    }

// declare all dtype kernel
SGEMMV_SHRINK_TYPE_DECLARE(half)
#if (__CCE_AICORE__ >= 220)
SGEMMV_SHRINK_TYPE_DECLARE(bfloat16_t)
#endif

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMV_SHRINK_H
