#include "moe_distribute_dispatch_v2.h"
#include "kernel_operator.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "moe_distribute_dispatch_tiling.h"
#include "moe_distribute_dispatch_v2_a5.h"
#ifdef __DAV_C310__
#include "moe_distribute_dispatch_v2_ccu.h"
using namespace MoeDistributeDispatchA5CCUImpl;
#endif

using namespace AscendC;
using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2A5Impl;

/*
 * Tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化, 3: 量化类型: mxfp8, 4: 量化类型: mxfp4
 * 第2位（十位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：是否走fullmesh_v2模板:
 *     0: 不做, 1: 做
 * 第5位（万位）: A2/A3/A5
 *     20000: A2, 30000: A3, 50000: A5
 */

extern "C" __global__ __aicore__ void moe_low_latency_dispatch_v2(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                                                  GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                                                  GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                                                  GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
                                                                  GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
                                                                  GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    TPipe pipe;

#ifdef __DAV_C310__
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(30000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30100)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(50000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2A5<DTYPE_X, DTYPE_EXPAND_X, DTYPE_DYNAMIC_SCALES, false, false, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#ifdef __DAV_C310__
    if (TILING_KEY_IS(60000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, 0, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(30011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30111)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, true, false, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30102)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(30112)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, false, true, true, true> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(50002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2A5<DTYPE_X, DTYPE_EXPAND_X, DTYPE_DYNAMIC_SCALES, false, true, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#ifdef __DAV_C310__
    if (TILING_KEY_IS(60002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, 2, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#elif (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2)
#ifdef __DAV_C310__
    if (TILING_KEY_IS(50005)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2A5<DTYPE_X, DTYPE_EXPAND_X, DTYPE_DYNAMIC_SCALES, false, true, false, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(50003)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2A5<DTYPE_X, DTYPE_EXPAND_X, DTYPE_DYNAMIC_SCALES, false, true, true, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#elif (ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_EXPAND_X == DT_FLOAT4_E1M2)
#ifdef __DAV_C310__
    if (TILING_KEY_IS(50004)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
        MoeDistributeDispatchV2A5<DTYPE_X, DTYPE_EXPAND_X, DTYPE_DYNAMIC_SCALES, false, true, true, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#endif
}
