/*!
 * \file causal_conv1d_update.cpp
 * \brief causal_conv1d_update kernel entry point
 */

#include "causal_conv1d_update.h"
#include "causal_conv1d_update_struct.h"

extern "C" __global__ __aicore__ void causal_conv1d_update_bfloat16_t(GM_ADDR x, GM_ADDR weight, GM_ADDR conv_state,
                                                                      GM_ADDR conv_state_indices, GM_ADDR bias,
                                                                      GM_ADDR num_accepted_tokens,
                                                                      GM_ADDR query_start_loc, GM_ADDR y,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dUpdateOp::CausalConv1dUpdate<bfloat16_t> op;
    op.Init(x, weight, conv_state, conv_state_indices, bias, num_accepted_tokens, query_start_loc, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void causal_conv1d_update_half(GM_ADDR x, GM_ADDR weight, GM_ADDR conv_state,
                                                                GM_ADDR conv_state_indices, GM_ADDR bias,
                                                                GM_ADDR num_accepted_tokens, GM_ADDR query_start_loc,
                                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    CausalConv1dUpdateOp::CausalConv1dUpdate<half> op;
    op.Init(x, weight, conv_state, conv_state_indices, bias, num_accepted_tokens, query_start_loc, y, tiling);
    op.Process();
}
