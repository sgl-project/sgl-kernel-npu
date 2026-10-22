#include "apply_token_bitmask_impl.h"

extern "C" __global__ __aicore__ void sgl_apply_token_bit_mask(
    GM_ADDR logits, GM_ADDR bitmask, GM_ADDR logitsRef, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    KernelApplyTokenBitmask<DTYPE_LOGITS> op;
    op.Init(logits, bitmask, logitsRef, data.numRows, data.vocabSize, data.logitsStride,
            data.bitmaskStride, data.baseRows, data.extraCores, data.tileLength, data.blockDim, data.dtypeSize);
    op.Process();
}
