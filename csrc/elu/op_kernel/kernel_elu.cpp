// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_KERNEL_ELU_H
#define SGL_KERNEL_NPU_KERNEL_ELU_H

/* include file of ascendc */
#include "kernel_operator.h"

/* tensor num for each queue */
constexpr int32_t BUFFER_NUM = 2;

/*
 * ELU (Exponential Linear Unit) element-wise activation:
 *
 *   elu(x) = x                       if x > 0
 *          = alpha * (exp(x) - 1)    if x <= 0
 *
 * Instead of branching on the AICore, we decompose the formula into a pure
 * sequence of AscendC vector instructions using the identities:
 *
 *   max(x, 0) = x - min(x, 0)
 *
 * so that:
 *
 *   elu(x) = alpha * (exp(min(x, 0)) - 1) + (x - min(x, 0))
 *
 * This is correct for every x (including x == 0) and is overflow-safe for
 * fp16 because exp() is only ever evaluated on the non-positive branch
 * min(x, 0) <= 0.
 */
template <typename T>
class KernelElu
{
public:
    __aicore__ inline KernelElu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength, float alpha)
    {
        /*
         * The host guarantees that:
         *   - totalLength is divisible by GetBlockNum() and tileLength, so
         *     every block owns exactly blockLength elements laid out as
         *     `tileCount` full tiles (no tail handling is needed in-kernel);
         *   - tileLength * sizeof(T) is aligned to the 32B vector unit.
         */
        this->tileLength = tileLength;
        this->alpha = static_cast<T>(alpha);
        uint32_t blockLength = totalLength / AscendC::GetBlockNum();
        this->tileCount = blockLength / tileLength;

        uint32_t blockOffset = blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
        /* two single-buffered scratch tensors for the intermediate results */
        pipe.InitBuffer(tmpBufA, tileLength * sizeof(T));
        pipe.InitBuffer(tmpBufB, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileCount; i++) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<T> tmpA = tmpBufA.Get<T>();
        AscendC::LocalTensor<T> tmpB = tmpBufB.Get<T>();

        /* tmpA = min(x, 0) */
        AscendC::Mins(tmpA, xLocal, static_cast<T>(0), this->tileLength);
        /* tmpB = exp(min(x, 0)) */
        AscendC::Exp(tmpB, tmpA, this->tileLength);
        /* out  = exp(min(x, 0)) - 1 */
        AscendC::Adds(outLocal, tmpB, static_cast<T>(-1), this->tileLength);
        /* tmpB = alpha * (exp(min(x, 0)) - 1) */
        AscendC::Muls(tmpB, outLocal, this->alpha, this->tileLength);
        /* out  = x - min(x, 0) = max(x, 0) */
        AscendC::Sub(outLocal, xLocal, tmpA, this->tileLength);
        /* out  = alpha * (exp(min(x, 0)) - 1) + max(x, 0) = elu(x) */
        AscendC::Add(outLocal, outLocal, tmpB, this->tileLength);

        outQueueY.EnQue<T>(outLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<T> outLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueueY.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufA;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufB;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t tileLength;
    uint32_t tileCount;
    T alpha;
};

extern "C" __global__ __aicore__ void elu_fp16(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength,
                                               float alpha)
{
    KernelElu<half> op;
    op.Init(x, y, totalLength, tileLength, alpha);
    op.Process();
}

extern "C" __global__ __aicore__ void elu_fp32(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength,
                                               float alpha)
{
    KernelElu<float> op;
    op.Init(x, y, totalLength, tileLength, alpha);
    op.Process();
}

#endif  // SGL_KERNEL_NPU_KERNEL_ELU_H
