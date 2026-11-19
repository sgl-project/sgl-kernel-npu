#include "fused_dispatch_decode.h"

extern "C" __global__ __aicore__ void fused_dispatch_decode(
    GM_ADDR x, GM_ADDR ids, GM_ADDR values, GM_ADDR scales, GM_ADDR ends,
    GM_ADDR source, GM_ADDR counts, GM_ADDR total, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchTilingData);
    if (TILING_KEY_IS(23)) {
        FusedDispatchDecodeImpl::FusedDispatchDecode op;
        op.Init(x, ids, values, scales, ends, source, counts, total, tiling);
        op.Process();
    }
}
