/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "shmem_moe_distribute_combine.h"
#include "shmem_moe_distribute_combine_tiling.h"

using namespace AscendC;
using namespace MoeDistributeCombineImpl;

extern "C" __global__ __aicore__ void shmem_moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount, GM_ADDR xActiveMask,
    GM_ADDR activationScale, GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR XOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ShmemMoeDistributeCombineTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
// #if (ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
// #if (ORIG_DTYPE_EXPAND_X == DT_BF16)
    if (TILING_KEY_IS(1100)) {  // tp=2
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeCombineTilingData, tilingData, tilingGM);
        ShmemMoeDistributeCombine<DTYPE_EXPAND_X, int32_t, true, false> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000)) {  // tp=1
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeCombineTilingData, tilingData, tilingGM);
        ShmemMoeDistributeCombine<DTYPE_EXPAND_X, int32_t, false, false> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(1020)) {  // tp=1, isQuant=true
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeCombineTilingData, tilingData, tilingGM);
        ShmemMoeDistributeCombine<DTYPE_EXPAND_X, int32_t, false, true> op;
        op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, scales, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
