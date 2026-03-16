#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "cam_moe_combine_normal.h"
#include "cam_moe_combine_normal_a5.h"
#include "cam_moe_combine_normal_multi_round.h"
#include "cam_moe_combine_normal_tiling.h"
using namespace AscendC;
using namespace CamMoeCombineNormalMultiRoundImpl;
using namespace CamMoeCombineNormalImpl;
using namespace CamMoeCombineNormalA5Impl;

/*
  5: INIT
  4: A5/A3/A2
*/
#define TILINGKEY_A3_MULTI_ROUND 13001
#define TILINGKEY_A3_SINGLE_ROUND 13000
#define TILINGKEY_A5_MULTI_ROUND 15001
#define TILINGKEY_A5_SINGLE_ROUND 15000

extern "C" __global__ __aicore__ void cam_moe_combine_normal(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                                             GM_ADDR topkWeights, GM_ADDR tokenIdx, GM_ADDR tpRecvCount,
                                                             GM_ADDR XOut, GM_ADDR sendCostStatsOut,
                                                             GM_ADDR workspaceGM, GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(CamMoeCombineNormalTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    GET_TILING_DATA_WITH_STRUCT(CamMoeCombineNormalTilingData, tilingData, tilingGM);
    if (TILING_KEY_IS(TILINGKEY_A3_MULTI_ROUND)) {
        CamMoeCombineNormalMultiRound<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_A3_SINGLE_ROUND)) {
        CamMoeCombineNormal<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_A5_SINGLE_ROUND)) {
        CamMoeCombineNormalA5<DTYPE_RECV_X, DTYPE_X, int32_t> op;
        op.Init(recvX, tokenSrcInfo, epRecvCount, topkWeights, tokenIdx, tpRecvCount, XOut, sendCostStatsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
