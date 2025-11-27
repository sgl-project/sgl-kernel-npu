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
    // REGISTER_TILING_DEFAULT(NotifyDispatchA2TilingData);
    // GET_TILING_DATA_WITH_STRUCT(NotifyDispatchA2TilingData, tilingData, tiling);
    REGISTER_TILING_DEFAULT(CamMoeDistributeDispatchA2TilingData);
    GET_TILING_DATA_WITH_STRUCT(CamMoeDistributeDispatchA2TilingData, tilingData, tiling);

    // hcomm will set magic later in init
    uint32_t magic = 1;
    GM_ADDR commArgs = nullptr;

    // int localRank = tilingData.notifyDispatchInfoA2.localRankId;
    // int localRankSize = tilingData.notifyDispatchInfoA2.localRankSize;
    // int rank = tilingData.notifyDispatchInfoA2.rankId;
    // int rankSize = tilingData.notifyDispatchInfoA2.rankSize;
    // int64_t len = tilingData.notifyDispatchInfoA2.sendCount;
    // int64_t numTokens = tilingData.notifyDispatchInfoA2.numTokens;
    // int64_t topkNum = tilingData.notifyDispatchInfoA2.topkNum;
    // int64_t numExperts = tilingData.notifyDispatchInfoA2.numExperts;

    // GM_ADDR sendDataInput = sendData;
    // GM_ADDR tokenPerExpertDataInput = tokenPerExpertData;
    // GM_ADDR sendDataOffsetOutput = sendDataOffset;
    // GM_ADDR recvDataOutput = recvData;
    // GM_ADDR tokenServerIdxOutput = tokenServerIdx;
    // GM_ADDR tokensUniquePerServerOutput = tokensUniquePerServer;
    // GM_ADDR epRankTokenCntOutput = epRankTokenCnt;
    // GM_ADDR localEpTokenCntOutput = localEpTokenCnt;
    // GM_ADDR srcOffsetRankTokenIdxOutput = srcOffsetRankTokenIdx;
    // GM_ADDR dstOffsetRankTokenIdxOutput = dstOffsetRankTokenIdx;
    // GM_ADDR offsetInnerOutput = offsetInner;
    // GM_ADDR countOuterOutput = countOuter;

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
        // NotifyDispatchA2<int> opKernel(rank, rankSize, extraFlag);
        // opKernel.Init(KERNELS_ARGS_CALL_A2_ALL2ALL());
        // opKernel.Process();
        CamMoeDistributeDispatchA2Layered<bfloat16_t, bfloat16_t, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
                srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, recvX, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epRecvCountOut, expandScalesOut, workspace, &pipe, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2000000000)) {
        // NotifyDispatchA2<int> opKernel(rank, rankSize, extraFlag);
        // opKernel.Init(KERNELS_ARGS_CALL_A2_ALL2ALL());
        // opKernel.Process();
        CamMoeDistributeDispatchA2Layered<bfloat16_t, bfloat16_t, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
                srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, recvX, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epRecvCountOut, expandScalesOut, workspace, &pipe, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2000001000)) {
        // NotifyDispatchA2<int> opKernel(rank, rankSize, extraFlag);
        // opKernel.Init(KERNELS_ARGS_CALL_A2_ALL2ALL());
        // opKernel.Process();
        CamMoeDistributeDispatchA2Layered<bfloat16_t, bfloat16_t, false, false, false> op;
        op.Init(x, expertIds, scales, expertScales, tokenServerIdx, tokenServerCnt, epRankTokenCnt,
                srcOffsetRankTokenIdx, dstOffsetRankTokenIdx, recvX, dynamicScalesOut, expandIdxOut, expertTokenNumsOut,
                epRecvCountOut, expandScalesOut, workspace, &pipe, tiling);
        op.Process();
    }
}
