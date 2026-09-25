// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Host entry for the fused W4A4 INT4 MoE "mega" kernel (Qwen3.x-MoE, Ascend 910B).
// One MIX launch runs the whole routed-expert path: int4 quant + routing scatter ->
// int4 gate_up -> SwiGLU + int4 requant -> int4 down -> unpermute/top-k combine, with
// the AscendC MatmulImpl<int4b_t> cube stages overlapped against the PTO-ISA vec stages.
//
// All workspaces, the two TCubeTiling structs, and the int4-NZ weight repack are prepared
// on the Python side (python/sgl_kernel_npu/sgl_kernel_npu/moe/mega_moe_w4a4.py), mirroring
// the mega_chunk_gdn split: this host function only validates shapes and launches.

#include <cstdint>
#include <limits>

#include "aclrtlaunch_launch_mega_moe_w4a4.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

namespace {
// Per-rank shapes the kernel is COMPILED for (constants in the .cpp: H_DIM, I_DIM_OVERRIDE).
constexpr int64_t kHidden = 2048;  // H_DIM
constexpr int64_t kInter = 128;    // I_DIM (per-rank intermediate, I_DIM_OVERRIDE=128)
constexpr int64_t kNGu = 2 * kInter;

void check_shape(const at::Tensor &x, const at::Tensor &w13, const at::Tensor &w2, const at::Tensor &group_list,
                 int64_t M_total, int64_t E, int64_t top_k, int64_t T_orig)
{
    const char *err = "Run the vendor-separated W4A4 MoE path by setting SGLANG_NPU_MEGA_MOE_W4A4=0.";
    auto check = [&err](bool condition, const char *message) { TORCH_CHECK(condition, message, " ", err); };

    check(x.is_contiguous(), "x must be contiguous (fix this in Python)");
    check(w13.is_contiguous(), "w13 must be contiguous (fix this in Python)");
    check(w2.is_contiguous(), "w2 must be contiguous (fix this in Python)");
    check(group_list.is_contiguous(), "group_list must be contiguous (fix this in Python)");

    check(x.dim() == 2 && x.size(1) == kHidden, "x must have shape [T_orig, 2048] (H_DIM)");
    // x is the raw routed activations; this build is the self-contained variant — the
    // block-diagonal Hadamard is applied in-kernel on the cube (Stage 0 -> xrot_ws, with
    // the H-64 blocks in b1), and the matching rotation is baked into w13 offline.
    check(x.scalar_type() == at::kHalf, "x must be float16 (routed activations)");
    check(group_list.scalar_type() == at::kLong, "group_list must be int64 (cumulative per-expert token counts)");
    check(group_list.numel() == E, "group_list must have E entries");
    check(E > 0 && E <= std::numeric_limits<uint32_t>::max(), "E out of uint32 range (must be positive)");
    check(E % 2 == 0, "E must be even (SAFESYNC NC=2 expert-chunk split)");
    check(M_total >= 0 && M_total <= std::numeric_limits<uint32_t>::max(), "M_total out of uint32 range");
    check(T_orig == x.size(0), "T_orig must match x.shape[0]");
    check(top_k >= 1 && top_k <= std::numeric_limits<uint32_t>::max(), "top_k out of uint32 range");
    // w13 / w2 are FRACTAL_NZ-packed int4 (int8 carrier, two int4 per byte); the Python
    // repack guarantees the NZ layout, so we only sanity-check the carrier dtype here.
    check(w13.scalar_type() == at::kChar || w13.scalar_type() == at::kQUInt4x2,
          "w13 must be int4-packed (int8 carrier or quint4x2)");
    check(w2.scalar_type() == at::kChar || w2.scalar_type() == at::kQUInt4x2,
          "w2 must be int4-packed (int8 carrier or quint4x2)");
}
}  // namespace

// Tensors are prepared by the Python wrapper; argument order matches the kernel entry
// W4A4_MEGA_KERNEL_NAME (== launch_mega_moe_w4a4).
HOST_API void mega_moe_w4a4(const at::Tensor &x, const at::Tensor &w13, const at::Tensor &w13_scale,
                            const at::Tensor &w2, const at::Tensor &w2_scale, const at::Tensor &group_list,
                            const at::Tensor &sort_idx, const at::Tensor &topk_w, at::Tensor &xq_ws, at::Tensor &xs_ws,
                            at::Tensor &gu_ws, at::Tensor &iq_ws, at::Tensor &is_ws, at::Tensor &d_ws, at::Tensor &y,
                            const at::Tensor &tiling_gu, const at::Tensor &tiling_dn, const at::Tensor &b1,
                            at::Tensor &xrot_ws, int64_t M_total, int64_t E, int64_t top_k, int64_t T_orig,
                            int64_t block_dim)
{
    check_shape(x, w13, w2, group_list, M_total, E, top_k, T_orig);
    // topk_w is read as half in the combine stage (stage5); fp32 weights would be byte-misread.
    TORCH_CHECK(topk_w.scalar_type() == at::kHalf && topk_w.is_contiguous(),
                "topk_w must be a contiguous float16 tensor (read as half in the combine stage). "
                "Run the vendor-separated W4A4 MoE path by setting SGLANG_NPU_MEGA_MOE_W4A4=0.");
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(), "block_dim out of uint32 range");

    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    uint32_t M_total_u32 = static_cast<uint32_t>(M_total);
    uint32_t E_u32 = static_cast<uint32_t>(E);
    uint32_t top_k_u32 = static_cast<uint32_t>(top_k);
    uint32_t T_orig_u32 = static_cast<uint32_t>(T_orig);

    EXEC_KERNEL_CMD(launch_mega_moe_w4a4, block_dim_u32, x, w13, w13_scale, w2, w2_scale, group_list, sort_idx, topk_w,
                    xq_ws, xs_ws, gu_ws, iq_ws, is_ws, d_ws, y, tiling_gu, tiling_dn, b1, xrot_ws, M_total_u32, E_u32,
                    top_k_u32, T_orig_u32);
}

}  // namespace npu_kernel
}  // namespace sglang
