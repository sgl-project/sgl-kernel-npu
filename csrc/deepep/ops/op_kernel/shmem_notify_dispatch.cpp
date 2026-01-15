#include "kernel_operator.h"
#include "shmem_notify_dispatch.h"
#include "shmem_notify_dispatch_tiling.h"

#define TILING_KEY_INT_SHMEM 223

#define KERNEL_USE_WORKSPACE (1 * 1024 * 1024)

using namespace ShmemNotifyDispatchImpl;

extern "C" __global__ __aicore__ void shmem_notify_dispatch(GM_ADDR tokenPerExpertData, GM_ADDR recvData,
                                                            GM_ADDR totalRecvTokens, GM_ADDR maxBs,
                                                            GM_ADDR recvTokensPerExpert, GM_ADDR putOffset,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ShmemNotifyDispatchTilingData);
    GET_TILING_DATA_WITH_STRUCT(ShmemNotifyDispatchTilingData, tilingData, tiling);

    int64_t len = tilingData.notifyDispatchInfo.sendCount;
    int localRank = tilingData.notifyDispatchInfo.localRankId;
    int localRankSize = tilingData.notifyDispatchInfo.localRankSize;
    int rank = tilingData.notifyDispatchInfo.rankId;
    int rankSize = tilingData.notifyDispatchInfo.rankSize;
    uint32_t topkNum = tilingData.notifyDispatchInfo.topkNum;
    uint64_t shmemPtr = tilingData.shmemPtr;

    GM_ADDR tokenPerExpertDataInput = tokenPerExpertData;
    GM_ADDR recvDataOutput = recvData;

    // fill in unused args
    uint32_t extraFlag = 0;
    GM_ADDR scale = nullptr;
    int root = 0;
    int op = 0;
    int cycleCount = 0;
    int64_t scaleCount = 0;
    GM_ADDR offset = nullptr;
    int blockNum = GetBlockNum();

    if (TILING_KEY_IS(TILING_KEY_INT_SHMEM)) {
        ShmemNotifyDispatch<int> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL_ALLGATHER());
        opKernel.Process();
    }
}
