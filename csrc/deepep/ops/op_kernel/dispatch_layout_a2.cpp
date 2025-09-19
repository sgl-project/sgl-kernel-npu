#include "kernel_operator.h"
#include "dispatch_layout_a2.h"
#include "dispatch_layout_tiling_a2.h"

extern "C" __global__ __aicore__ void dispatch_layout_a2(GM_ADDR topkIdx, GM_ADDR numTokensPerRank,
                                                         GM_ADDR numTokensPerExpert, GM_ADDR isTokenInRank,
                                                         GM_ADDR localTokenServerOffset, GM_ADDR localTokenServerUniqCount,
                                                         GM_ADDR localTokenServerTotalCount, GM_ADDR localTokenServerNum,
                                                         GM_ADDR expertRankTokenIdx, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DispatchLayoutA2TilingData);
    GET_TILING_DATA_WITH_STRUCT(DispatchLayoutA2TilingData, tilingData, tiling);

    TPipe pipe;

    DispatchLayoutA2<int32_t> op;
    op.Init(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, localTokenServerOffset,
            localTokenServerUniqCount, localTokenServerTotalCount, localTokenServerNum,
            expertRankTokenIdx, workspace, &pipe, &tilingData);
    op.Process();
}
