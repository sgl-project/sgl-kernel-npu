// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.

#include <cstdint>
#include <limits>
#include <stdexcept>

// TP=1, global key heads=16
#include "aclrtlaunch_launch_mega_kernel_h16_hg16.h"
#include "aclrtlaunch_launch_mega_kernel_h32_hg16.h"
#include "aclrtlaunch_launch_mega_kernel_h48_hg16.h"
#include "aclrtlaunch_launch_mega_kernel_h64_hg16.h"
// TP=2, local key heads=8
#include "aclrtlaunch_launch_mega_kernel_h16_hg8.h"
#include "aclrtlaunch_launch_mega_kernel_h32_hg8.h"
// TP=4, local key heads=4
#include "aclrtlaunch_launch_mega_kernel_h16_hg4.h"
#include "defines.h"
#include "torch_helper.h"

#define SGLANG_FOR_EACH_MEGA_CHUNK_GDN_VARIANT(MACRO) \
    MACRO(16, 16)                                     \
    MACRO(32, 16)                                     \
    MACRO(48, 16)                                     \
    MACRO(64, 16)                                     \
    MACRO(16, 8)                                      \
    MACRO(32, 8)                                      \
    MACRO(16, 4)

namespace sglang {
namespace npu_kernel {

namespace {
constexpr int64_t kGlobalKeyHeads = 16;
constexpr int64_t kHeadDim = 128;

bool is_supported_global_value_heads(int64_t value_heads)
{
    return value_heads == 16 || value_heads == 32 || value_heads == 48 || value_heads == 64;
}

bool is_supported_tp_degree(int64_t tp_degree)
{
    return tp_degree == 1 || tp_degree == 2 || tp_degree == 4 || tp_degree == 8;
}

bool is_supported_head_pair(int64_t value_heads, int64_t key_heads)
{
    if (key_heads <= 0 || kGlobalKeyHeads % key_heads != 0) {
        return false;
    }
    int64_t tp_degree = kGlobalKeyHeads / key_heads;
    return is_supported_tp_degree(tp_degree) && is_supported_global_value_heads(value_heads * tp_degree);
}

void check_shape(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                 const at::Tensor &beta, const at::Tensor &cu_seqlens, const at::Tensor &initial_state,
                 bool has_initial_state)
{
    TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, Hg, D]");
    TORCH_CHECK(k.dim() == 4, "k must have shape [B, T, Hg, D]");
    TORCH_CHECK(v.dim() == 4, "v must have shape [B, T, H, D]");
    TORCH_CHECK(g.dim() == 3, "g must have shape [B, T, H]");
    TORCH_CHECK(beta.dim() == 3, "beta must have shape [B, T, H]");

    TORCH_CHECK(q.size(0) == 1, "mega_chunk_gdn currently supports packed B=1 input");
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
    TORCH_CHECK(q.size(1) == v.size(1), "q/k and v sequence lengths must match");
    TORCH_CHECK(is_supported_head_pair(v.size(2), q.size(2)),
                "unsupported mega_chunk_gdn (NumValueHeads, NumKeyHeads) pair");
    TORCH_CHECK(q.size(3) == kHeadDim && v.size(3) == kHeadDim, "mega_chunk_gdn supports head dimension 128");

    TORCH_CHECK(g.size(0) == 1 && beta.size(0) == 1, "g and beta must use packed B=1 layout");
    TORCH_CHECK(g.size(1) == q.size(1) && beta.size(1) == q.size(1), "g/beta sequence lengths must match q");
    TORCH_CHECK(g.size(2) == v.size(2) && beta.size(2) == v.size(2), "g/beta must use the same NumValueHeads as v");

    TORCH_CHECK(q.scalar_type() == at::kHalf, "q must be float16");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
    TORCH_CHECK(v.scalar_type() == at::kHalf, "v must be float16");
    TORCH_CHECK(beta.scalar_type() == at::kHalf, "beta must be float16");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");

    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");
    TORCH_CHECK(g.is_contiguous() && beta.is_contiguous() && cu_seqlens.is_contiguous(),
                "g, beta, and cu_seqlens must be contiguous");

    if (has_initial_state) {
        TORCH_CHECK(initial_state.dim() == 4, "initial_state must have shape [N, H, D, D]");
        TORCH_CHECK(initial_state.size(0) == cu_seqlens.numel() - 1,
                    "initial_state.size(0) must match cu_seqlens sequences");
        TORCH_CHECK(initial_state.size(1) == v.size(2), "initial_state.size(1) must match NumValueHeads");
        TORCH_CHECK(initial_state.size(2) == kHeadDim && initial_state.size(3) == kHeadDim,
                    "initial_state must use head dimensions 128 x 128");
        TORCH_CHECK(initial_state.scalar_type() == at::kHalf, "initial_state must be float16");
        TORCH_CHECK(initial_state.is_contiguous(), "initial_state must be contiguous");
    }
}
}  // namespace

HOST_API void mega_chunk_gdn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                             const at::Tensor &beta, const at::Tensor &mask_lower, const at::Tensor &mask_full,
                             const at::Tensor &minus_identity, const at::Tensor &cu_seqlens, at::Tensor &out,
                             at::Tensor &g_sum, at::Tensor &g_t, at::Tensor &beta_t, at::Tensor &a,
                             at::Tensor &a_inv_f32, at::Tensor &a_inv, at::Tensor &w, at::Tensor &u, at::Tensor &s,
                             at::Tensor &v_new, at::Tensor &final_state, const at::Tensor &initial_state,
                             bool has_initial_state, at::Tensor &kkt_workspace, at::Tensor &wy_workspace_a1,
                             at::Tensor &wy_workspace_a2, at::Tensor &h_workspace, at::Tensor &o_workspace_qk,
                             at::Tensor &o_workspace_qs, at::Tensor &o_workspace_gated, int64_t block_dim,
                             int64_t batch_size, int64_t seq_len, int64_t total_tokens, int64_t num_matrices)
{
    check_shape(q, k, v, g, beta, cu_seqlens, initial_state, has_initial_state);
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(), "block_dim is out of uint32 range");
    TORCH_CHECK(batch_size == cu_seqlens.numel() - 1, "batch_size must match cu_seqlens");
    TORCH_CHECK(seq_len == q.size(1), "seq_len must match q.shape[1]");
    TORCH_CHECK(total_tokens == q.size(1), "total_tokens must match packed token count");
    TORCH_CHECK(num_matrices >= 0 && num_matrices <= std::numeric_limits<uint32_t>::max(),
                "num_matrices is out of uint32 range");

    uint32_t num_matrices_u32 = static_cast<uint32_t>(num_matrices);
    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    int64_t has_initial_state_i64 = has_initial_state ? 1 : 0;

#define LAUNCH_MEGA_CHUNK_GDN(H, HG)                                                                             \
    EXEC_KERNEL_CMD(launch_mega_kernel_h##H##_hg##HG, block_dim_u32, q, k, v, g, beta, mask_lower, mask_full,    \
                    minus_identity, cu_seqlens, out, g_sum, g_t, beta_t, a, a_inv_f32, a_inv, w, u, s, v_new,    \
                    final_state, initial_state, has_initial_state_i64, kkt_workspace, wy_workspace_a1,           \
                    wy_workspace_a2, h_workspace, o_workspace_qk, o_workspace_qs, o_workspace_gated, batch_size, \
                    seq_len, total_tokens, num_matrices_u32)

#define DISPATCH_MEGA_CHUNK_GDN(H, HG)       \
    if (v.size(2) == H && q.size(2) == HG) { \
        LAUNCH_MEGA_CHUNK_GDN(H, HG);        \
        return;                              \
    }

    SGLANG_FOR_EACH_MEGA_CHUNK_GDN_VARIANT(DISPATCH_MEGA_CHUNK_GDN)
    TORCH_CHECK(false, "unsupported mega_chunk_gdn (NumValueHeads, NumKeyHeads) pair");

#undef DISPATCH_MEGA_CHUNK_GDN
#undef LAUNCH_MEGA_CHUNK_GDN
}

}  // namespace npu_kernel
}  // namespace sglang

#undef SGLANG_FOR_EACH_MEGA_CHUNK_GDN_VARIANT
