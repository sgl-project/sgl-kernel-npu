##### Description of Helloworld
This is an extremely simple example of writing an op on Ascend NPU.

A typical op includes two major parts:
* Device part which really running device, i.e. NPU
* Host part which running on host, i.e. host CPU

Let's take a close look the code of helloworld op, this op is to add two tensor

Device part code:
```
/*
 * The following code runs on AICore and nothing to do with Torch:
 * a typical kernel major wrapped by a class and which have two major functions:
 * - Init(xxx), init everything first
 * - Process(), op logic
 *
 * a entry function of the kernel 'helloworld' which calls the kernel
 *
 */

#include "kernel_operator.h" /* include file of ascendc */

constexpr int32_t BUFFER_NUM = 2; /* tensor num for each queue */

class KernalHelloworld {
public:
    __aicore__ inline KernalHelloworld() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = 8;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void helloworld(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernalHelloworld op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
```

Host part code:
```
/*
 * Include this helper file located in utils,
 * this file includes the most importance MACRO 'EXEC_KERNAL_CMD'
 * which is tow launch the function with pytorch
 */
#include "torch_npu_helper.h"

/*
 * This file is generated automatically by compile tool
 */
#include "aclrtlaunch_helloworld.h"

namespace sglang {
namespace npu_kernel {

at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y)
{
    /* create a result tensor */
    at::Tensor z = at::empty_like(x);

    /* define the block dim */
    uint32_t blockDim = 8;

    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }

    /* launch the kernel function via torch */
    EXEC_KERNEL_CMD(add_custom, blockDim, x, y, z, totalLength);
    return z;
}
```
