/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/csrc/kernels/sgmv_expand.cpp
 */

#ifndef SGL_KERNEL_NPU_KERNEL_SGEMMV_EXPAND_H
#define SGL_KERNEL_NPU_KERNEL_SGEMMV_EXPAND_H

#include "kernel_operator.h"
#include "lora_common_kernel.h"

template <typename scalar_t>
class SGEMMVExpand
{
public:
    using X_T = float;
    using W_T = scalar_t;
    using Y_T = scalar_t;

    static constexpr int32_t LORA_RANK_8 = 8;
    static constexpr int32_t LORA_RANK_16 = 16;
    static constexpr int32_t LORA_RANK_32 = 32;
    static constexpr int32_t LORA_RANK_64 = 64;

    static constexpr int32_t BUFFER_NUM = 2;

    static constexpr int32_t DATA_VECTOR_BLOCK = 32;
    static constexpr int32_t NUM_BYTES_PER_REPEAT = 256;
    static constexpr int32_t NUM_BLOCKS_PER_REPEAT = 8;

    static constexpr int32_t NUM_ELEMENTS_PER_REPEAT = NUM_BYTES_PER_REPEAT / sizeof(float);

    static constexpr int32_t MASK_COUNT = NUM_BYTES_PER_REPEAT / sizeof(float);
    static constexpr int32_t W_IN_TILE_NUM_ELEMENTS = 4096;

    static constexpr int32_t W_IN_TILE_STORAGE_ELEMENTS = 8192;

    static constexpr int32_t Y_OUT_TILE_NUM_ELEMENTS = 512;

    static constexpr int32_t BLOCK_REDUCE_NUM_REPEATS = W_IN_TILE_NUM_ELEMENTS / NUM_ELEMENTS_PER_REPEAT;

    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_16 =
        (BLOCK_REDUCE_NUM_REPEATS * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;

    static constexpr int32_t PAIR_REDUCE_NUM_REPEATS_32 = (PAIR_REDUCE_NUM_REPEATS_16 + 1) / 2;

public:
    __aicore__ inline SGEMMVExpand(AscendC::TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize,
                                GM_ADDR seqLen, uint32_t seqLenSize, GM_ADDR loraRanks, uint32_t loraRanksSize,
                                GM_ADDR sliceOffsets, uint32_t sliceOffsetsSize, GM_ADDR yIn, GM_ADDR yOut,
                                uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank,
                                uint32_t outputFullDim)
    {
        batchSize_ = batchSize;
        numTokensPerCore_ = numTokensPerCore;
        maxLoRARank_ = maxLoRARank;
        outputFullDim_ = outputFullDim;

        if (sliceOffsetsSize >= 2) {
            sliceCount_ = sliceOffsetsSize - 1;
        } else {
            sliceCount_ = 0;
        }

        singleLoRAWeightLen_ = static_cast<uint64_t>(maxLoRARank_) * static_cast<uint64_t>(outputFullDim_);

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ W_T *>(weight));
        yInGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yIn));
        yOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y_T *>(yOut));
        loraIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraIndices), loraIndicesSize);
        seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLen), seqLenSize);
        loraRanksGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(loraRanks), loraRanksSize);
        sliceOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sliceOffsets), sliceOffsetsSize);

        pipe_->InitBuffer(inQueueX_, 1, NUM_ELEMENTS_PER_REPEAT * sizeof(float));
        pipe_->InitBuffer(inQueueW_, BUFFER_NUM, W_IN_TILE_STORAGE_ELEMENTS * sizeof(W_T));
        pipe_->InitBuffer(inQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, Y_OUT_TILE_NUM_ELEMENTS * sizeof(Y_T));
        pipe_->InitBuffer(dupBufferX_, W_IN_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBufferW_, W_IN_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(inBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(tmpBufferY_, Y_OUT_TILE_NUM_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (sliceCount_ == 0) {
            return;
        }

        const int64_t blockIdxSlice = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockIdx = blockIdxSlice / static_cast<int64_t>(sliceCount_);
        const int64_t startIdx = blockIdx * static_cast<int64_t>(numTokensPerCore_);
        int64_t endIdx = startIdx + static_cast<int64_t>(numTokensPerCore_);
        if (endIdx > static_cast<int64_t>(batchSize_)) {
            endIdx = static_cast<int64_t>(batchSize_);
        }

        reqSlice_ = static_cast<int32_t>(blockIdxSlice % static_cast<int64_t>(sliceCount_));

        sliceOffset_ = static_cast<uint32_t>(sliceOffsetsGm_.GetValue(reqSlice_));
        outputHiddenDim_ = static_cast<uint32_t>(sliceOffsetsGm_.GetValue(reqSlice_ + 1)) - sliceOffset_;

        lora_common::BlockIterator blockIterator(seqLenGm_);
        for (int64_t idx = startIdx; idx < endIdx; ++idx) {
            yOffset_ = static_cast<uint64_t>(outputFullDim_) * static_cast<uint64_t>(idx) +
                       static_cast<uint64_t>(sliceOffset_);

            const int64_t requestBlock = blockIterator.GetBlockIdx(idx);

            if (requestBlock < 0) {
                continue;
            }

            reqLoRAIndex_ = loraIndicesGm_.GetValue(requestBlock);

            if (reqLoRAIndex_ < 0) {
                continue;
            }

            reqLoRARank_ = loraRanksGm_.GetValue(reqLoRAIndex_);

            if (!IsSupportedRank(reqLoRARank_)) {
                continue;
            }
            reqLoRAWeightOffset_ = static_cast<uint64_t>(reqLoRAIndex_) * singleLoRAWeightLen_ +
                                   static_cast<uint64_t>(sliceOffset_) * static_cast<uint64_t>(maxLoRARank_);
            numOutputElementsPerInputTile_ = BLOCK_REDUCE_NUM_REPEATS * (NUM_ELEMENTS_PER_REPEAT / reqLoRARank_);
            numStreamInPerOutputTile_ =
                (Y_OUT_TILE_NUM_ELEMENTS + numOutputElementsPerInputTile_ - 1) / numOutputElementsPerInputTile_;

            CopyInX(idx);
            const int32_t numFullYTiles = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
            for (int32_t i = 0; i < numFullYTiles; ++i) {
                CopyInY(i, Y_OUT_TILE_NUM_ELEMENTS);
                ClearAccumulator();
                for (int32_t j = 0; j < numStreamInPerOutputTile_; ++j) {
                    CopyInW(i * numStreamInPerOutputTile_ + j, W_IN_TILE_NUM_ELEMENTS);
                    Compute(j * numOutputElementsPerInputTile_);
                }
                ScaleOutput(Y_OUT_TILE_NUM_ELEMENTS);
                CopyOut(i, Y_OUT_TILE_NUM_ELEMENTS);
            }

            ComputeLastIteration();
        }
    }

private:
    __aicore__ inline bool IsSupportedRank(int32_t rank)
    {
        return rank == LORA_RANK_8 || rank == LORA_RANK_16 || rank == LORA_RANK_32 || rank == LORA_RANK_64;
    }

    __aicore__ inline void CopyInX(int64_t idx)
    {
        AscendC::LocalTensor<X_T> xLocal = inQueueX_.AllocTensor<X_T>();

        const uint64_t xOffset =
            static_cast<uint64_t>(sliceCount_) * static_cast<uint64_t>(maxLoRARank_) * static_cast<uint64_t>(idx) +
            static_cast<uint64_t>(reqSlice_) * static_cast<uint64_t>(maxLoRARank_);

        if constexpr (std::is_same_v<X_T, float>) {
            DataCopy(xLocal, xGm_[xOffset], reqLoRARank_);

        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(reqLoRARank_ * sizeof(X_T)), 0, 0, 0};
            const uint32_t paddedElements = ((reqLoRARank_ * sizeof(X_T) + DATA_VECTOR_BLOCK - 1) / DATA_VECTOR_BLOCK) *
                                            DATA_VECTOR_BLOCK / sizeof(X_T);
            const uint32_t rightPadding = paddedElements - reqLoRARank_;
            AscendC::DataCopyPadExtParams<X_T> padParams{true, 0, static_cast<uint8_t>(rightPadding),
                                                         static_cast<X_T>(0)};
            DataCopyPad(xLocal, xGm_[xOffset], copyParams, padParams);
        }

        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.DeQue<X_T>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();

        if constexpr (std::is_same_v<X_T, float>) {
            for (int32_t j = 0; j < reqLoRARank_; ++j) {
                xDup.SetValue(j, xLocal.GetValue(j));
            }

        } else {
            Cast(xDup, xLocal, AscendC::RoundMode::CAST_NONE, reqLoRARank_);
            pipe_barrier(PIPE_V);
        }

        // TODO: can be vectorized
        for (int32_t i = maxLoRARank_; i < NUM_ELEMENTS_PER_REPEAT; i += maxLoRARank_) {
            for (int32_t j = 0; j < maxLoRARank_; j++) {
                float entry = xDup.GetValue(j);
                xDup.SetValue(i + j, entry);
            }
        }

        inQueueX_.FreeTensor(xLocal);
    }

     __aicore__ inline void CopyInY(int32_t progress, int32_t numElements = Y_OUT_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.AllocTensor<Y_T>();
        DataCopy(yInLocal, yInGm_[yOffset_ + progress * Y_OUT_TILE_NUM_ELEMENTS], numElements);
        inQueueY_.EnQue(yInLocal);
    }

    __aicore__ inline void CopyInW(int32_t progress, int32_t numElements = W_IN_TILE_NUM_ELEMENTS)
    {
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.AllocTensor<W_T>();
        DataCopy(wLocal, wGm_[reqLoRAWeightOffset_ + progress * (W_IN_TILE_NUM_ELEMENTS / reqLoRARank_) * maxLoRARank_],
                 {static_cast<uint16_t>(numElements / reqLoRARank_),
                  static_cast<uint16_t>((reqLoRARank_ * sizeof(W_T) + DATA_VECTOR_BLOCK - 1) / DATA_VECTOR_BLOCK),
                  static_cast<uint16_t>((maxLoRARank_ - reqLoRARank_) * sizeof(W_T) / DATA_VECTOR_BLOCK), 0});
        inQueueW_.EnQue(wLocal);
    }

    __aicore__ inline void ClearAccumulator()
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        Duplicate(yLocal, 0.0f, Y_OUT_TILE_NUM_ELEMENTS);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t blockReduceRepeatCount = BLOCK_REDUCE_NUM_REPEATS,
                                   int32_t /*pairReduceRepeat16*/ = 0, int32_t /*pairReduceRepeat32*/ = 0)
    {
        if (blockReduceRepeatCount <= 0) {
            return;
        }

        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<float> xDup = dupBufferX_.Get<float>();
        AscendC::LocalTensor<W_T> wLocal = inQueueW_.DeQue<W_T>();
        AscendC::LocalTensor<float> wTmp = tmpBufferW_.Get<float>();

        const int32_t rank = reqLoRARank_;
        const int32_t rows = (blockReduceRepeatCount * NUM_ELEMENTS_PER_REPEAT) / rank;
        const int32_t logicalElements = rows * rank;

        const uint32_t paddedRowBytes =
            ((static_cast<uint32_t>(rank) * sizeof(W_T) + DATA_VECTOR_BLOCK - 1) / DATA_VECTOR_BLOCK) *
            DATA_VECTOR_BLOCK;
        const int32_t paddedRowElements = static_cast<int32_t>(paddedRowBytes / sizeof(W_T));

        for (int32_t row = 0; row < rows; ++row) {
            AscendC::LocalTensor<W_T> src = wLocal[row * paddedRowElements];

            AscendC::LocalTensor<float> dst = wTmp[row * rank];

            Cast(dst, src, AscendC::RoundMode::CAST_NONE, rank);
        }

        if (logicalElements < W_IN_TILE_NUM_ELEMENTS) {
            Duplicate(wTmp[logicalElements], 0.0f, W_IN_TILE_NUM_ELEMENTS - logicalElements);

            pipe_barrier(PIPE_V);
        }

        pipe_barrier(PIPE_V);

        inQueueW_.FreeTensor(wLocal);

        Mul(wTmp, wTmp, xDup, MASK_COUNT, blockReduceRepeatCount, dotProductParams_);

        pipe_barrier(PIPE_V);

        // TODO: can be vectorized
        for (int32_t row = 0; row < rows; ++row) {
            const int32_t base = row * rank;
            float sum = 0.0f;

            for (int32_t k = 0; k < rank; ++k) {
                sum += wTmp.GetValue(base + k);
            }

            yLocal.SetValue(progress + row, sum);
        }

        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ScaleOutput(int32_t numElements)
    {
        AscendC::LocalTensor<float> yLocal = tmpBufferY_.Get<float>();
        AscendC::LocalTensor<Y_T> yInLocal = inQueueY_.DeQue<Y_T>();
        AscendC::LocalTensor<float> yInLocalFP32 = inBufferY_.Get<float>();
        Cast(yInLocalFP32, yInLocal, AscendC::RoundMode::CAST_NONE, numElements);
        pipe_barrier(PIPE_V);
        inQueueY_.FreeTensor(yInLocal);
        Add(yLocal, yLocal, yInLocalFP32, numElements);
        pipe_barrier(PIPE_V);
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.AllocTensor<Y_T>();
        Cast(yOutLocal, yLocal, AscendC::RoundMode::CAST_RINT, numElements);
        pipe_barrier(PIPE_V);
        outQueueY_.EnQue(yOutLocal);
    }

    __aicore__ inline void ComputeLastIteration()
    {
        const int32_t remainingY = outputHiddenDim_ % Y_OUT_TILE_NUM_ELEMENTS;
        if (remainingY == 0) {
            return;
        }

        const int32_t numStreamOut = outputHiddenDim_ / Y_OUT_TILE_NUM_ELEMENTS;
        const int32_t remainingW = remainingY * reqLoRARank_;
        const int32_t fullWTiles = remainingW / W_IN_TILE_NUM_ELEMENTS;
        const int32_t remainingWElements = remainingW % W_IN_TILE_NUM_ELEMENTS;
        CopyInY(numStreamOut, remainingY);
        ClearAccumulator();
        for (int32_t i = 0; i < fullWTiles; ++i) {
            CopyInW(numStreamOut * numStreamInPerOutputTile_ + i, W_IN_TILE_NUM_ELEMENTS);
            Compute(i * numOutputElementsPerInputTile_);
        }

        if (remainingWElements > 0) {
            const int32_t progress = numStreamOut * numStreamInPerOutputTile_ + fullWTiles;
            CopyInW(progress, remainingWElements);
            const int32_t lastRepeatCount =
                (remainingWElements + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
            const int32_t pairReduceRepeat16 =
                (lastRepeatCount * NUM_BLOCKS_PER_REPEAT + NUM_ELEMENTS_PER_REPEAT - 1) / NUM_ELEMENTS_PER_REPEAT;
            const int32_t pairReduceRepeat32 = (pairReduceRepeat16 + 1) / 2;
            const int32_t outputProgress = fullWTiles * numOutputElementsPerInputTile_;
            Compute(outputProgress, lastRepeatCount, pairReduceRepeat16, pairReduceRepeat32);
        }

        ScaleOutput(remainingY);
        CopyOut(numStreamOut, remainingY);
    }

    __aicore__ inline void CopyOut(int32_t progress, int32_t numElements)
    {
        AscendC::LocalTensor<Y_T> yOutLocal = outQueueY_.DeQue<Y_T>();
        const uint64_t offset = yOffset_ + static_cast<uint64_t>(progress) * Y_OUT_TILE_NUM_ELEMENTS;

        if (numElements == Y_OUT_TILE_NUM_ELEMENTS) {
            DataCopy(yOutGm_[offset], yOutLocal, numElements);

        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(numElements * sizeof(Y_T)), 0, 0, 0};

            DataCopyPad(yOutGm_[offset], yOutLocal, copyParams);
        }

        outQueueY_.FreeTensor(yOutLocal);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 8 * BUFFER_NUM> inQueueW_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferW_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dupBufferX_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBufferY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufferY_;

    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<W_T> wGm_;
    AscendC::GlobalTensor<Y_T> yInGm_;
    AscendC::GlobalTensor<Y_T> yOutGm_;

    AscendC::GlobalTensor<int32_t> loraIndicesGm_;
    AscendC::GlobalTensor<int32_t> seqLenGm_;
    AscendC::GlobalTensor<int32_t> loraRanksGm_;
    AscendC::GlobalTensor<int32_t> sliceOffsetsGm_;

    uint32_t batchSize_ = 0;
    uint32_t sliceCount_ = 0;
    uint32_t numTokensPerCore_ = 0;
    uint32_t maxLoRARank_ = 0;
    uint32_t outputHiddenDim_ = 0;
    uint32_t sliceOffset_ = 0;
    uint32_t outputFullDim_ = 0;
    uint64_t singleLoRAWeightLen_ = 0;
    int64_t reqLoRAIndex_ = -1;
    int32_t reqLoRARank_ = 0;
    uint64_t reqLoRAWeightOffset_ = 0;
    int32_t reqSlice_ = 0;
    uint32_t numOutputElementsPerInputTile_ = 0;
    uint32_t numStreamInPerOutputTile_ = 0;
    uint64_t yOffset_ = 0;

    AscendC::UnaryRepeatParams castParams_ = {1, 1, 8, 4};
    AscendC::BinaryRepeatParams dotProductParams_ = {1, 1, 1, 8, 0, 8};
};

#define SGEMMV_EXPAND_TYPE_DECLARE(TYPE)                                                                               \
    extern "C" __global__ __aicore__ void sgemmv_expand_##TYPE(                                                        \
        GM_ADDR x, GM_ADDR weight, GM_ADDR loraIndices, uint32_t loraIndicesSize, GM_ADDR seqLen, uint32_t seqLenSize, \
        GM_ADDR loraRanks, uint32_t loraRanksSize, GM_ADDR sliceOffsets, uint32_t sliceOffsetsSize, GM_ADDR yIn,       \
        GM_ADDR yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputFullDim)     \
    {                                                                                                                  \
        AscendC::TPipe pipe;                                                                                           \
        SGEMMVExpand<TYPE> op(&pipe);                                                                                  \
        op.Init(x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, loraRanks, loraRanksSize, sliceOffsets,   \
                sliceOffsetsSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputFullDim);                 \
        op.Process();                                                                                                  \
    }

SGEMMV_EXPAND_TYPE_DECLARE(half)

#if (__CCE_AICORE__ >= 220)
SGEMMV_EXPAND_TYPE_DECLARE(bfloat16_t)
#endif

#endif  // SGL_KERNEL_NPU_KERNEL_SGEMMV_EXPAND_H
