#include "kernel_operator.h"
#include "notify_dispatch_a2.h"
#include "notify_dispatch_tiling_a2.h"

#define TILING_KEY_FLOAT16 20
#define TILING_KEY_BFLOAT16 21
#define TILING_KEY_FLOAT 22
#define TILING_KEY_INT 23
#define TILING_KEY_A2_FLOAT16 120
#define TILING_KEY_A2_BFLOAT16 121
#define TILING_KEY_A2_FLOAT 122
#define TILING_KEY_A2_INT 123

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

extern "C" __global__ __aicore__ void notify_dispatch_a2(GM_ADDR sendData, GM_ADDR tokenPerExpertData, GM_ADDR tmpData,
                                                         GM_ADDR sendDataOffset, GM_ADDR recvData,
                                                         GM_ADDR tokenServerIdx, GM_ADDR tokensUniquePerServer,
                                                         GM_ADDR epRankTokenCnt, GM_ADDR localEpTokenCnt,
                                                         GM_ADDR srcOffsetRankTokenIdx, GM_ADDR dstOffsetRankTokenIdx,
                                                         GM_ADDR offsetInner, GM_ADDR countOuter, GM_ADDR expandIdx,
                                                         GM_ADDR totalRecvTokens, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NotifyDispatchA2TilingData);
    GET_TILING_DATA_WITH_STRUCT(NotifyDispatchA2TilingData, tilingData, tiling);

    int localRank = tilingData.notifyDispatchInfoA2.localRankId;
    int localRankSize = tilingData.notifyDispatchInfoA2.localRankSize;
    int rank = tilingData.notifyDispatchInfoA2.rankId;
    int rankSize = tilingData.notifyDispatchInfoA2.rankSize;
    int64_t len = tilingData.notifyDispatchInfoA2.sendCount;
    int64_t numTokens = tilingData.notifyDispatchInfoA2.numTokens;
    int64_t topkNum = tilingData.notifyDispatchInfoA2.topkNum;
    int64_t numExperts = tilingData.notifyDispatchInfoA2.numExperts;

    GM_ADDR sendDataInput = sendData;
    GM_ADDR tokenPerExpertDataInput = tokenPerExpertData;
    GM_ADDR tmpDataInput = tmpData;

    GM_ADDR sendDataOffsetOutput = sendDataOffset;
    GM_ADDR recvDataOutput = recvData;
    GM_ADDR tokenServerIdxOutput = tokenServerIdx;
    GM_ADDR tokensUniquePerServerOutput = tokensUniquePerServer;
    GM_ADDR epRankTokenCntOutput = epRankTokenCnt;
    GM_ADDR localEpTokenCntOutput = localEpTokenCnt;
    GM_ADDR srcOffsetRankTokenIdxOutput = srcOffsetRankTokenIdx;
    GM_ADDR dstOffsetRankTokenIdxOutput = dstOffsetRankTokenIdx;
    GM_ADDR offsetInnerOutput = offsetInner;
    GM_ADDR countOuterOutput = countOuter;
    GM_ADDR expandIdxOutput = expandIdx;
    GM_ADDR totalRecvTokensOutput = totalRecvTokens;

    // fill in unused args
    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int64_t scaleCount = 0;
    GM_ADDR offset = nullptr;
    int blockNum = GetBlockNum();

    if (TILING_KEY_IS(TILING_KEY_A2_INT)) {
        NotifyDispatchA2<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_A2_ALL2ALL());
        opKernel.Process();
    }
}
