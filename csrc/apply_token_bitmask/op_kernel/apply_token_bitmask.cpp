// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BITS_PER_INT32 = 32;

template <typename T>
class KernelApplyTokenBitmask {
public:
    __aicore__ inline KernelApplyTokenBitmask() {}

    __aicore__ inline void Init(GM_ADDR logitsGmAddr, GM_ADDR bitmaskGmAddr,
                                 uint32_t numRows, uint32_t vocabSize,
                                 uint32_t logitsStride, uint32_t bitmaskStride,
                                 uint32_t rowsPerCore, uint32_t rowsLastCore,
                                 uint32_t tileLength, uint32_t blockDim,
                                 uint32_t dtypeSize) {
        this->numRows = numRows;
        this->vocabSize = vocabSize;
        this->logitsStride = logitsStride;
        this->bitmaskStride = bitmaskStride;
        this->tileLength = tileLength;
        this->dtypeSize = dtypeSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < blockDim - 1) {
            this->startRow = blockIdx * rowsPerCore;
            this->localRows = rowsPerCore;
        } else {
            this->startRow = blockIdx * rowsPerCore;
            this->localRows = rowsLastCore;
        }

        logitsGm.SetGlobalBuffer((__gm__ T*)logitsGmAddr);
        bitmaskGm.SetGlobalBuffer((__gm__ int32_t*)bitmaskGmAddr);

        // UB allocation per tile:
        // logitsQueue: tileLength * sizeof(T)
        // bitmaskQueue: (tileLength / 32) * sizeof(int32_t), min 8 elements
        // outQueueLogits: tileLength * sizeof(T)
        uint32_t bitmaskTileElems = tileLength / BITS_PER_INT32;
        if (bitmaskTileElems < 8) {
            bitmaskTileElems = 8;
        }

        pipe.InitBuffer(inQueueLogits, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueBitmask, BUFFER_NUM, bitmaskTileElems * sizeof(int32_t));
        pipe.InitBuffer(outQueueLogits, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < this->localRows; rowIdx++) {
            uint32_t batchId = this->startRow + rowIdx;
            uint32_t numTiles = (this->vocabSize + this->tileLength - 1) / this->tileLength;

            for (uint32_t tileIdx = 0; tileIdx < numTiles; tileIdx++) {
                uint32_t offset = tileIdx * this->tileLength;
                uint32_t curTileLen = this->tileLength;
                if (offset + curTileLen > this->vocabSize) {
                    curTileLen = this->vocabSize - offset;
                }
                CopyIn(batchId, offset, curTileLen);
                Compute(curTileLen);
                CopyOut(batchId, offset, curTileLen);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchId, uint32_t vocabOffset, uint32_t curTileLen)
    {
        AscendC::LocalTensor<T> logitsLocal = inQueueLogits.AllocTensor<T>();
        uint32_t logitsGmOffset = batchId * this->logitsStride + vocabOffset;
        AscendC::DataCopy(logitsLocal, logitsGm[logitsGmOffset], curTileLen);
        inQueueLogits.EnQue(logitsLocal);

        AscendC::LocalTensor<int32_t> bitmaskLocal = inQueueBitmask.AllocTensor<int32_t>();
        uint32_t bitmaskGmOffset = batchId * this->bitmaskStride + vocabOffset / BITS_PER_INT32;
        uint32_t bitmaskElems = (curTileLen + BITS_PER_INT32 - 1) / BITS_PER_INT32;
        uint32_t alignedBitmaskElems = AlignUp(bitmaskElems, 8);
        AscendC::DataCopy(bitmaskLocal, bitmaskGm[bitmaskGmOffset], alignedBitmaskElems);
        inQueueBitmask.EnQue(bitmaskLocal);
    }

    __aicore__ inline void Compute(uint32_t curTileLen)
    {
        AscendC::LocalTensor<T> logitsLocal = inQueueLogits.DeQue<T>();
        AscendC::LocalTensor<int32_t> bitmaskLocal = inQueueBitmask.DeQue<int32_t>();
        AscendC::LocalTensor<T> outLocal = outQueueLogits.AllocTensor<T>();

        // Copy logits to output
        AscendC::DataCopy(outLocal, logitsLocal, curTileLen);

        // Apply bitmask: process one int32 (32 bits) at a time
        // If bit is 0, set output to -inf
        T negInf = static_cast<T>(-1.0f / 0.0f);
        for (uint32_t intIdx = 0; intIdx < (curTileLen + BITS_PER_INT32 - 1) / BITS_PER_INT32; intIdx++) {
            int32_t packed = bitmaskLocal.GetValue(intIdx);
            if (packed == -1) continue; // All bits are 1, no masking needed
            for (uint32_t bitIdx = 0; bitIdx < BITS_PER_INT32; bitIdx++) {
                uint32_t i = intIdx * BITS_PER_INT32 + bitIdx;
                if (i >= curTileLen) break;
                if (((packed >> static_cast<int32_t>(bitIdx)) & 1) == 0) {
                    outLocal.SetValue(i, negInf);
                }
            }
        }

        outQueueLogits.EnQue(outLocal);
        inQueueLogits.FreeTensor(logitsLocal);
        inQueueBitmask.FreeTensor(bitmaskLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batchId, uint32_t vocabOffset, uint32_t curTileLen)
    {
        AscendC::LocalTensor<T> outLocal = outQueueLogits.DeQue<T>();
        uint32_t outputGmOffset = batchId * this->logitsStride + vocabOffset;
        AscendC::DataCopy(logitsGm[outputGmOffset], outLocal, curTileLen);
        outQueueLogits.FreeTensor(outLocal);
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return (value + align - 1) / align * align;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueLogits;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueBitmask;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueLogits;
    AscendC::GlobalTensor<T> logitsGm;
    AscendC::GlobalTensor<int32_t> bitmaskGm;

    uint32_t numRows;
    uint32_t vocabSize;
    uint32_t logitsStride;
    uint32_t bitmaskStride;
    uint32_t tileLength;
    uint32_t dtypeSize;
    uint32_t startRow;
    uint32_t localRows;
};

extern "C" __global__ __aicore__ void apply_token_bitmask_fp16(
    GM_ADDR logits, GM_ADDR bitmask,
    uint32_t numRows, uint32_t vocabSize,
    uint32_t logitsStride, uint32_t bitmaskStride,
    uint32_t rowsPerCore, uint32_t rowsLastCore,
    uint32_t tileLength, uint32_t blockDim,
    uint32_t dtypeSize) {
    KernelApplyTokenBitmask<half> op;
    op.Init(logits, bitmask, numRows, vocabSize,
            logitsStride, bitmaskStride,
            rowsPerCore, rowsLastCore,
            tileLength, blockDim, dtypeSize);
    op.Process();
}

extern "C" __global__ __aicore__ void apply_token_bitmask_fp32(
    GM_ADDR logits, GM_ADDR bitmask,
    uint32_t numRows, uint32_t vocabSize,
    uint32_t logitsStride, uint32_t bitmaskStride,
    uint32_t rowsPerCore, uint32_t rowsLastCore,
    uint32_t tileLength, uint32_t blockDim,
    uint32_t dtypeSize) {
    KernelApplyTokenBitmask<float> op;
    op.Init(logits, bitmask, numRows, vocabSize,
            logitsStride, bitmaskStride,
            rowsPerCore, rowsLastCore,
            tileLength, blockDim, dtypeSize);
    op.Process();
}

extern "C" __global__ __aicore__ void apply_token_bitmask_bf16(
    GM_ADDR logits, GM_ADDR bitmask,
    uint32_t numRows, uint32_t vocabSize,
    uint32_t logitsStride, uint32_t bitmaskStride,
    uint32_t rowsPerCore, uint32_t rowsLastCore,
    uint32_t tileLength, uint32_t blockDim,
    uint32_t dtypeSize) {
    KernelApplyTokenBitmask<bfloat16_t> op;
    op.Init(logits, bitmask, numRows, vocabSize,
            logitsStride, bitmaskStride,
            rowsPerCore, rowsLastCore,
            tileLength, blockDim, dtypeSize);
    op.Process();
}
