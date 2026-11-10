// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Framework adaptation of csrc/mega_chunk_gdn; PTO computation is unchanged.

#include "kernel_operator.h"
// CANN's generated MIX wrapper calls matmul::clearWorkspace.
#include "lib/matmul_intf.h"
#include "mega_chunk_gdn_tiling_data.h"

// Load dependency headers globally before including the unchanged implementation
// in a PTO scope. Preserve its MEMORY_BASE setting before the first PTO include.
#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <type_traits>

// Matmul headers import AscendC names globally. Keep PTO lookup inside pto so
// DYNAMIC, GlobalTensor and other common names resolve to the original PTO types.
namespace pto::sgl_mega {
#include "mega_kernel_impl.h"
}  // namespace pto::sgl_mega

extern "C" __global__ __aicore__ void sgl_mega_chunk_gdn(
    GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr, GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr,
    GM_ADDR msk_full_ptr, GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr_input, GM_ADDR g_sum_ptr_input,
    GM_ADDR g_t_ptr_input, GM_ADDR beta_t_ptr_input, GM_ADDR A_ptr_input, GM_ADDR A_inv_f32_ptr_input,
    GM_ADDR A_inv_ptr_input, GM_ADDR w_ptr_input, GM_ADDR u_ptr_input, GM_ADDR s_ptr_input, GM_ADDR v_new_ptr_input,
    GM_ADDR fs_ptr_input, GM_ADDR h0_ptr, GM_ADDR kkt_ws_ptr_input, GM_ADDR wy_ws_a1_ptr_input,
    GM_ADDR wy_ws_a2_ptr_input, GM_ADDR h_ws_ptr_input, GM_ADDR o_ws_qk_ptr_input, GM_ADDR o_ws_qs_ptr_input,
    GM_ADDR o_ws_gated_ptr_input, GM_ADDR o_ptr, GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr,
    GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr, GM_ADDR u_ptr, GM_ADDR s_ptr, GM_ADDR v_new_ptr,
    GM_ADDR fs_ptr, GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr, GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr,
    GM_ADDR o_ws_qk_ptr, GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(sgl_mega_gdn::TilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const auto has_initial_state = data.has_initial_state;
    const auto batch_size = data.batch_size;
    const auto seq_len = data.seq_len;
    const auto total_tokens = data.total_tokens;
    const auto num_matrices = data.num_matrices;
    const auto num_heads = data.num_heads;
    const auto num_key_heads = data.num_key_heads;

#define DISPATCH_MEGA_H(H)                                                                                             \
    case H:                                                                                                            \
        pto::sgl_mega::mega_kernel_impl<H>(                                                                            \
            q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr, minus_id_ptr, cu_seqlens_ptr, o_ptr, \
            g_sum_ptr, g_t_ptr, beta_t_ptr, A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr,   \
            h0_ptr, has_initial_state, kkt_ws_ptr, wy_ws_a1_ptr, wy_ws_a2_ptr, h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr,     \
            o_ws_gated_ptr, num_key_heads, batch_size, seq_len, total_tokens, num_matrices);                           \
        return

    switch (num_heads) {
        DISPATCH_MEGA_H(8);
        DISPATCH_MEGA_H(12);
        DISPATCH_MEGA_H(16);
        DISPATCH_MEGA_H(24);
        DISPATCH_MEGA_H(32);
        DISPATCH_MEGA_H(48);
        DISPATCH_MEGA_H(64);
        default:
            return;
    }

#undef DISPATCH_MEGA_H
}
