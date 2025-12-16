/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "shmem_moe_distribute_dispatch.h"
#include "shmem_moe_distribute_dispatch_tiling.h"

using namespace AscendC;
using namespace MoeDistributeDispatchImpl;

/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化
 * 第2位（十位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：是否是共享专家卡:
 *     0: 不是, 1: 是
 * 第5位（万位）：无实际意义
 */

extern "C" __global__ __aicore__ void shmem_moe_distribute_dispatch(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                                                    GM_ADDR xActiveMask, GM_ADDR expandXOut,
                                                                    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                                                    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                                                    GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM,
                                                                    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(ShmemMoeDistributeDispatchTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    // #if (ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    // #if (ORIG_DTYPE_EXPAND_X == DT_BF16)
    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ShmemMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1100)) {
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ShmemMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(1002)) {
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ShmemMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1102)) {
        GET_TILING_DATA_WITH_STRUCT(ShmemMoeDistributeDispatchTilingData, tilingData, tilingGM);
        ShmemMoeDistributeDispatch<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(x, expertIds, scales, expandXOut, dynamicScalesOut, expandIdxOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
