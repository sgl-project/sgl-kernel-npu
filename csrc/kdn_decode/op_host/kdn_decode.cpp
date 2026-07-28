// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Host wrapper for the vector-only PTO-ISA KDA/KDN recurrent decode kernel.
// Reached from python only when KDN_DECODE_PTO_BACKEND=1; the default path stays
// on the triton kernel in fla/fused_sigmoid_gating_recurrent.py.

#include <cstdint>
#include <limits>

#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_launch_kdn_decode.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

namespace {
constexpr int64_t kHeadDim = 128;

void check_shape(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                 const at::Tensor &beta, const at::Tensor &state, const at::Tensor &out,
                 const at::Tensor &state_indices, const at::Tensor &cu_seqlens)
{
    const char *err = "Unset KDN_DECODE_PTO_BACKEND to fall back to the triton decode kernel.";
    auto check = [&err](bool condition, const char *message) { TORCH_CHECK(condition, message, " ", err); };

    check(q.dim() == 4, "q must have shape [B, T, H, K]");
    check(k.sizes() == q.sizes(), "k must have q's shape");
    check(g.sizes() == q.sizes(), "g must have q's shape [B, T, H, K] (per-channel KDA gate)");
    check(v.dim() == 4, "v must have shape [B, T, H, V]");
    check(v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(2) == q.size(2),
          "v must share q's B/T/H -- this kernel does not implement GQA grouping (HV must equal H)");
    check(q.size(3) == kHeadDim && v.size(3) == kHeadDim, "kdn_decode supports head dimension 128");
    check(beta.dim() == 3 && beta.size(0) == q.size(0) && beta.size(1) == q.size(1) && beta.size(2) == q.size(2),
          "beta must have shape [B, T, H]");
    check(out.sizes() == v.sizes(), "out must have v's shape");

    // fp16 is the C220 vector TCVT wire format; the launcher converts model bf16 once.
    check(q.scalar_type() == at::kHalf, "q must be float16");
    check(k.scalar_type() == at::kHalf, "k must be float16");
    check(v.scalar_type() == at::kHalf, "v must be float16");
    check(g.scalar_type() == at::kHalf, "g must be float16 (log-space gate)");
    check(beta.scalar_type() == at::kHalf, "beta must be float16");
    check(out.scalar_type() == at::kHalf, "out must be float16");

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

    check(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous(),
          "q, k, v, and g must be contiguous");
    check(beta.is_contiguous() && out.is_contiguous(), "beta and out must be contiguous");
    // A non-contiguous pool would make the in-place state update land in a copy.
    check(state.is_contiguous(), "state must be contiguous so the in-place update reaches the pool");
    check(state_indices.is_contiguous() && cu_seqlens.is_contiguous(),
          "state_indices and cu_seqlens must be contiguous");
}
}  // namespace

HOST_API void kdn_decode(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                         const at::Tensor &beta, at::Tensor &state, at::Tensor &out,
                         const at::Tensor &state_indices, const at::Tensor &cu_seqlens, double scale,
                         bool use_qk_l2norm)
{
    check_shape(q, k, v, g, beta, state, out, state_indices, cu_seqlens);

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

    EXEC_KERNEL_CMD(launch_kdn_decode, block_dim_u32, q, k, v, g, beta, state, out, state_indices, cu_seqlens,
                    num_sequences, seq_len, num_heads, num_state_slots, scale_f, l2norm);
}

}  // namespace npu_kernel
}  // namespace sglang
