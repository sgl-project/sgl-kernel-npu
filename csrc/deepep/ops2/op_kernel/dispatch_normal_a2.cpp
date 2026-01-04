#include "kernel_operator.h"
// #include "notify_dispatch_a2.h"
// #include "notify_dispatch_tiling_a2.h"
// #include "a2/a2.h"
#include "a2/cam_moe_distribute_dispatch_a2_layered.h"
#include "cam_moe_distribute_dispatch_tiling.h"

#define TILING_KEY_FLOAT16 20
#define TILING_KEY_BFLOAT16 21
#define TILING_KEY_FLOAT 22
#define TILING_KEY_INT 23
#define TILING_KEY_A2_FLOAT16 120
#define TILING_KEY_A2_BFLOAT16 121
#define TILING_KEY_A2_FLOAT 122
#define TILING_KEY_A2_INT 123

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

using namespace AscendC;
using namespace MoeDistributeDispatchA2Impl;
using namespace Cam;

extern "C" __global__ __aicore__ void dispatch_normal_a2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR tokenServerIdx,
    GM_ADDR tokenServerCnt, GM_ADDR epRankTokenCnt, GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx,
    GM_ADDR recvX, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountOut,
    GM_ADDR expandScalesOut, GM_ADDR dispatchWaitRecvCostStatsOut, GM_ADDR workspace, GM_ADDR tiling)
{
    printf("[dispatch_normal_a2] blockId: %d\n", GetBlockIdx());
    REGISTER_TILING_DEFAULT(CamMoeDistributeDispatchA2TilingData);
    // GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tiling);

    // hcomm will set magic later in init
    uint32_t magic = 1;
    GM_ADDR commArgs = nullptr;

    // fill in unused args
    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int64_t scaleCount = 0;
    GM_ADDR offset = nullptr;
    int blockNum = GetBlockNum();

    TPipe pipe;
    if (TILING_KEY_IS(2100001000)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tiling);
        CamMoeDistributeDispatchA2Layered<bfloat16_t, bfloat16_t, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
                srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, recvX, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epRecvCountOut, expandScalesOut, workspace, &pipe, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2100001002)) {
        GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tiling);
        CamMoeDistributeDispatchA2Layered<bfloat16_t, int8_t, false, true, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
                srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, recvX, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epRecvCountOut, expandScalesOut, workspace, &pipe, tiling);
        op.Process();
    }
}
