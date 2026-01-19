#include "kernel_operator.h"
#include "moe_distribute_combine_a2_tiling.h"
#include "moe_distribute_combine_a2_layered.h"
#include <cstdio>

using namespace AscendC;
using namespace MoeDistributeCombineA2Impl;
extern "C" __global__ __aicore__ void moe_distribute_combine_a2(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
    GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales,
    GM_ADDR offsetInner, GM_ADDR offsetOuter, GM_ADDR countOuter, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineA2TilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(3000)) {
        MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expandIdx, epSendCount, offsetInner, offsetOuter, countOuter, expandScales, XOut, workspaceGM,
                &pipe, tilingGM);
        op.Process();
    }
#endif
}
