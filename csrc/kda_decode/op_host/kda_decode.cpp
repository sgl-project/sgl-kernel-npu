// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Host wrapper for the vector-only PTO-ISA KDA (Kimi Delta Attention) recurrent
// decode kernel.
// Reached from python only when KDA_DECODE_PTO_BACKEND=1; the default path stays
// on the triton kernel in fla/fused_sigmoid_gating_recurrent.py.

#include <cstdint>
#include <limits>

#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_launch_kda_decode.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

namespace {
constexpr int64_t kHeadDim = 128;

void check_shape(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &A_log,
                 const at::Tensor &a, const at::Tensor &dt_bias, const at::Tensor &b, const at::Tensor &state,
                 const at::Tensor &out, const at::Tensor &state_indices, const at::Tensor &cu_seqlens)
{
    const char *err = "Unset KDA_DECODE_PTO_BACKEND to fall back to the triton decode kernel.";
    auto check = [&err](bool condition, const char *message) { TORCH_CHECK(condition, message, " ", err); };

    check(q.dim() == 4, "q must have shape [B, T, H, K]");
    check(k.sizes() == q.sizes(), "k must have q's shape");
    check(v.dim() == 4, "v must have shape [B, T, H, V]");
    check(v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(2) == q.size(2),
          "v must share q's B/T/H -- this kernel does not implement GQA grouping (HV must equal H)");
    check(q.size(3) == kHeadDim && v.size(3) == kHeadDim, "kda_decode supports head dimension 128");
    check(out.sizes() == v.sizes(), "out must have v's shape");

    // The gating is fused, so the kernel takes the raw parameters rather than a
    // precomputed g/beta.  a is indexed exactly like q -- [tokens, H * K] flat.
    const int64_t tokens = q.size(1), heads = q.size(2);
    check(A_log.numel() == heads, "A_log must hold one value per head");
    check(a.numel() == tokens * heads * kHeadDim, "a must hold [tokens, H * K] values");
    check(dt_bias.numel() == heads * kHeadDim, "dt_bias must hold [H * K] values");
    check(b.numel() == tokens * heads, "b must hold [tokens, H] values");

    check(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    check(k.scalar_type() == at::kBFloat16, "k must be bfloat16");
    check(v.scalar_type() == at::kBFloat16, "v must be bfloat16");
    check(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
    check(A_log.scalar_type() == at::kFloat, "A_log must be float32");
    check(a.scalar_type() == at::kFloat, "a must be float32");
    check(dt_bias.scalar_type() == at::kFloat, "dt_bias must be float32");
    check(b.scalar_type() == at::kFloat, "b must be float32");

    // V-major [slots, H, V, K], matching sglang's temporal_state pool.
    check(state.dim() == 4, "state must have shape [slots, H, V, K]");
    check(state.size(1) == v.size(2), "state.size(1) must match the head count");
    check(state.size(2) == kHeadDim && state.size(3) == kHeadDim, "state must be [.., 128, 128]");
    check(state.scalar_type() == at::kFloat, "state must be float32");

    check(state_indices.dim() == 1, "state_indices must be a 1-D int32 tensor");
    check(state_indices.scalar_type() == at::kInt, "state_indices must be int32");
    check(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2, "cu_seqlens must be int32 [N+1]");
    check(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    check(cu_seqlens.numel() - 1 == state_indices.numel(),
          "cu_seqlens must describe exactly state_indices.numel() sequences");
    check(q.size(0) == 1, "cu_seqlens addressing requires the packed B=1 layout");

    check(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");
    check(out.is_contiguous(), "out must be contiguous");
    check(A_log.is_contiguous() && a.is_contiguous() && dt_bias.is_contiguous() && b.is_contiguous(),
          "A_log, a, dt_bias, and b must be contiguous");
    // A non-contiguous pool would make the in-place state update land in a copy.
    check(state.is_contiguous(), "state must be contiguous so the in-place update reaches the pool");
    check(state_indices.is_contiguous() && cu_seqlens.is_contiguous(),
          "state_indices and cu_seqlens must be contiguous");
}
}  // namespace

HOST_API void kda_decode(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &A_log,
                         const at::Tensor &a, const at::Tensor &dt_bias, const at::Tensor &b, at::Tensor &state,
                         at::Tensor &out, const at::Tensor &state_indices, const at::Tensor &cu_seqlens, double scale,
                         bool use_qk_l2norm, double softplus_beta, double softplus_threshold)
{
    check_shape(q, k, v, A_log, a, dt_bias, b, state, out, state_indices, cu_seqlens);
    (void)softplus_threshold;
    TORCH_CHECK(softplus_beta > 0, "softplus_beta must be positive");

    // One work item per (sequence, head): v_tile == 128 covers the whole [V, K]
    // state in one pass, so there is no v-tile axis to split.  The kernel is
    // vector-only, so block_dim counts AIV cores and each block is one worker.
    int64_t total_work = (cu_seqlens.numel() - 1) * v.size(2);
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    int32_t block_dim = std::min(max_aiv, static_cast<int32_t>(std::min<int64_t>(total_work, max_aiv)));
    if (block_dim <= 0) {
        block_dim = 1;
    }

    // EXEC_KERNEL_CMD binds its arguments by non-const reference, so every
    // scalar has to be a named local rather than a temporary.
    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    int64_t num_sequences = cu_seqlens.numel() - 1;
    int64_t seq_len = q.size(1);
    int32_t num_heads = static_cast<int32_t>(v.size(2));
    int32_t num_state_slots = static_cast<int32_t>(state.size(0));
    float scale_f = static_cast<float>(scale);
    int32_t l2norm = use_qk_l2norm ? 1 : 0;
    float softplus_beta_f = static_cast<float>(softplus_beta);

    EXEC_KERNEL_CMD(launch_kda_decode, block_dim_u32, q, k, v, A_log, a, dt_bias, b, state, out, state_indices,
                    cu_seqlens, num_sequences, seq_len, num_heads, num_state_slots, scale_f, l2norm, softplus_beta_f);
}

}  // namespace npu_kernel
}  // namespace sglang
