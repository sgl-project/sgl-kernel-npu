#include "kernel_operator.h"
#include "cam_moe_dispatch_normal_tiling.h"
#include "cam_moe_dispatch_normal.h"
#include "cam_moe_dispatch_normal_a5.h"

using namespace AscendC;
using namespace CamMoeDispatchNormalImpl;
using namespace CamMoeDispatchNormalA5Impl;

#define TILINGKEY_A3_NO_QUANT 13000
#define TILINGKEY_A3_QUANT 13002
#define TILINGKEY_A5_NO_QUANT 15000
#define TILINGKEY_A5_QUANT 15002
#define TILINGKEY_A5_MXFP8_QUANT 15003
#define TILINGKEY_A5_MXFP4_QUANT 15004
#define TILINGKEY_A5_SCALAR_FP8_QUANT 15005

extern "C" __global__ __aicore__ void cam_moe_dispatch_normal(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset, GM_ADDR send_token_idx, GM_ADDR recv_offset, GM_ADDR recv_count,
    GM_ADDR expert_global_offset, GM_ADDR srcrank_in_expert_offset, GM_ADDR r_in_srcrank_offset, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR assist_info_for_combine, GM_ADDR waitRecvCostStatsOut, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(CamMoeDispatchNormalTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    if (TILING_KEY_IS(TILINGKEY_A3_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TILINGKEY_A5_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormalA5<DTYPE_X, DTYPE_RECV_X, DTYPE_X_SCALES, false, false, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_RECV_X == DT_INT8)
    if (TILING_KEY_IS(TILINGKEY_A3_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormal<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TILINGKEY_A5_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormalA5<DTYPE_X, DTYPE_RECV_X, DTYPE_X_SCALES, true, false, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif

#ifdef __DAV_C310__
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#if (ORIG_DTYPE_RECV_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_RECV_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_RECV_X == DT_HIFLOAT8)
    if (TILING_KEY_IS(TILINGKEY_A5_MXFP8_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormalA5<DTYPE_X, DTYPE_RECV_X, DTYPE_X_SCALES, true, true, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILINGKEY_A5_SCALAR_FP8_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormalA5<DTYPE_X, DTYPE_RECV_X, DTYPE_X_SCALES, true, false, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#if (ORIG_DTYPE_RECV_X == DT_FLOAT4_E2M1 || ORIG_DTYPE_RECV_X == DT_FLOAT4_E1M2)
    if (TILING_KEY_IS(TILINGKEY_A5_MXFP4_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDispatchNormalTilingData, tilingData, tilingGM);
        CamMoeDispatchNormalA5<DTYPE_X, DTYPE_RECV_X, DTYPE_X_SCALES, true, true, false, false> op;
        op.Init(x, expertIds, send_offset, send_token_idx, recv_offset, recv_count, expert_global_offset,
                srcrank_in_expert_offset, r_in_srcrank_offset, expandXOut, dynamicScalesOut, assist_info_for_combine,
                waitRecvCostStatsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
        return;
    }
#endif
#endif  // __DAV_C310__
}
