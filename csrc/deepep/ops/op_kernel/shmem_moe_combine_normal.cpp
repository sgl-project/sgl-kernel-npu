#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "shmem_moe_combine_normal.h"
#include "shmem_moe_combine_normal_tiling.h"
using namespace AscendC;
using namespace ShmemMoeCombineNormalImpl;

extern "C" __global__ __aicore__ void shmem_moe_combine_normal(GM_ADDR recvX, GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                                               GM_ADDR topkIdx, GM_ADDR sendTokenIdx,
                                                               GM_ADDR balanceMatrix, GM_ADDR XOut,
                                                               GM_ADDR sendCostStatsOut, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(ShmemMoeCombineNormalTilingData);
    TPipe pipe;

#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    GET_TILING_DATA_WITH_STRUCT(ShmemMoeCombineNormalTilingData, tilingData, tilingGM);
    ShmemMoeCombineNormal<DTYPE_RECV_X, DTYPE_X, int32_t> op;
    op.Init(recvX, epRecvCount, topkWeights, topkIdx, sendTokenIdx, balanceMatrix, XOut, sendCostStatsOut, workspaceGM,
            &pipe, &tilingData);
    op.Process();
#endif
}
