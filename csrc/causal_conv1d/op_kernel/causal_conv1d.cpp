/*!
 * \file causal_conv1d.cpp
 * \brief causal_conv1d kernel entry point
 */

#include "causal_conv1d.h"

namespace {

using sglang::npu_kernel::CausalConv1dTilingData;

template <typename T>
__aicore__ inline void RunCausalConv1d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                       GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y,
                                       const CausalConv1dTilingData *tilingData)
{
    NsCausalConv1d::CausalConv1d<T> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, tilingData);
    op.Process();
}

__aicore__ inline void InitTilingData(GM_ADDR tiling, CausalConv1dTilingData &tiling_data)
{
    auto gm_tiling = reinterpret_cast<__gm__ CausalConv1dTilingData *>(tiling);
    tiling_data.dim = gm_tiling->dim;
    tiling_data.cuSeqlen = gm_tiling->cuSeqlen;
    tiling_data.seqLen = gm_tiling->seqLen;
    tiling_data.inputMode = gm_tiling->inputMode;
    tiling_data.width = gm_tiling->width;
    tiling_data.stateLen = gm_tiling->stateLen;
    tiling_data.numCacheLines = gm_tiling->numCacheLines;
    tiling_data.batch = gm_tiling->batch;
    tiling_data.activationMode = gm_tiling->activationMode;
    tiling_data.padSlotId = gm_tiling->padSlotId;
    tiling_data.hasBias = gm_tiling->hasBias;
    tiling_data.dimTileSize = gm_tiling->dimTileSize;
    tiling_data.blocksPerSeq = gm_tiling->blocksPerSeq;
}

}  // namespace

extern "C" __global__ __aicore__ void causal_conv1d_bfloat16_t(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                               GM_ADDR convStates, GM_ADDR queryStartLoc,
                                                               GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    CausalConv1dTilingData tiling_data;
    InitTilingData(tiling, tiling_data);
    RunCausalConv1d<bfloat16_t>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y,
                                &tiling_data);
}

extern "C" __global__ __aicore__ void causal_conv1d_half(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                         GM_ADDR hasInitialState, GM_ADDR y, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    (void)workspace;
    CausalConv1dTilingData tiling_data;
    InitTilingData(tiling, tiling_data);
    RunCausalConv1d<half>(x, weight, bias, convStates, queryStartLoc, cacheIndices, hasInitialState, y, &tiling_data);
}
