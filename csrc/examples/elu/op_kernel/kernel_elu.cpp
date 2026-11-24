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

#include "elu_tiling.h"
#include "kernel_operator.h"

/* number of buffers in the input and output queues */
constexpr int32_t BUFFER_NUM = 2;

/*
 * ELU (Exponential Linear Unit) element-wise activation:
 *
 *   elu(x) = x                       if x > 0
 *          = alpha * (exp(x) - 1)    if x <= 0
 *
 * Instead of branching on the AICore, the formula is rewritten with the
 * identity max(x, 0) = x - min(x, 0):
 *
 *   elu(x) = alpha * (exp(min(x, 0)) - 1) + (x - min(x, 0))
 *
 * so the vector unit executes one straight line of instructions per tile.
 * Exp only ever sees min(x, 0) <= 0, whose result stays in (0, 1], therefore
 * fp16 cannot overflow here.
 */
template <typename T>
class KernelElu
{
public:
    __aicore__ inline KernelElu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const optiling::SglEluTilingData &tiling)
    {
        /*
         * The Host splits totalLength evenly: the first extraBlocks cores own
         * baseLength + 1 elements, the others own baseLength. A core may own
         * fewer elements than one tile, and the last tile of a core is
         * generally shorter than tileLength; both are handled by CopyIn and
         * CopyOut with DataCopyPad, so the Kernel needs no padding contract.
         */
        this->tileLength = tiling.tileLength;
        this->alpha = static_cast<T>(tiling.alpha);
        this->blockIdx = AscendC::GetBlockIdx();

        uint32_t baseLength = tiling.baseLength;
        uint32_t extraBlocks = tiling.extraBlocks;
        if (this->blockIdx < extraBlocks) {
            this->start = this->blockIdx * (baseLength + 1);
            this->length = baseLength + 1;
        } else {
            this->start = extraBlocks * (baseLength + 1) + (this->blockIdx - extraBlocks) * baseLength;
            this->length = baseLength;
        }

        xGm.SetGlobalBuffer((__gm__ T *)x + this->start, this->length);
        yGm.SetGlobalBuffer((__gm__ T *)y + this->start, this->length);

        /*
         * Six tiles of UB are live at peak: the double-buffered input and
         * output queues (2 + 2) plus the two single-buffered scratch tensors
         * (1 + 1). The Host sizes tileLength from this same assumption.
         */
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBufA, this->tileLength * sizeof(T));
        pipe.InitBuffer(tmpBufB, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->length; offset += this->tileLength) {
            uint32_t curLength = this->length - offset;
            if (curLength > this->tileLength) {
                curLength = this->tileLength;
            }
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<T> tmpA = tmpBufA.Get<T>();
        AscendC::LocalTensor<T> tmpB = tmpBufB.Get<T>();

        /* tmpA = min(x, 0) */
        AscendC::Mins(tmpA, xLocal, static_cast<T>(0), curLength);
        /* tmpB = exp(min(x, 0)) */
        AscendC::Exp(tmpB, tmpA, curLength);
        /* out = exp(min(x, 0)) - 1 */
        AscendC::Adds(outLocal, tmpB, static_cast<T>(-1), curLength);
        /* tmpB = alpha * (exp(min(x, 0)) - 1) */
        AscendC::Muls(tmpB, outLocal, this->alpha, curLength);
        /* out = x - min(x, 0) = max(x, 0) */
        AscendC::Sub(outLocal, xLocal, tmpA, curLength);
        /* out = alpha * (exp(min(x, 0)) - 1) + max(x, 0) = elu(x) */
        AscendC::Add(outLocal, outLocal, tmpB, curLength);

        outQueueY.EnQue<T>(outLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<T> outLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad<T>(yGm[offset], outLocal, copyParams);
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
    uint32_t blockIdx;
    uint32_t start;
    uint32_t length;
    T alpha;
};

/*
 * One entry serves fp16 and fp32: the SDK compiles the Kernel once per dtype
 * combination declared by the OpDef and substitutes DTYPE_X accordingly.
 */
extern "C" __global__ __aicore__ void sgl_elu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::SglEluTilingData);
    GET_TILING_DATA(data, tiling);
    KernelElu<DTYPE_X> op;
    op.Init(x, y, data);
    op.Process();
}

#endif  // SGL_KERNEL_NPU_KERNEL_ELU_H
