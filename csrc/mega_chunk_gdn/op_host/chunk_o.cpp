// Minimal host launcher for debugging the PTO chunk_o kernel.

#include <cstdint>
#include <limits>

#include "aclrtlaunch_launch_chunk_o_debug.h"
#include "defines.h"
#include "runtime/rt_ffts.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

HOST_API void chunk_o_debug(const at::Tensor &q,
                            const at::Tensor &k,
                            const at::Tensor &v,
                            const at::Tensor &s,
                            const at::Tensor &g_t,
                            const at::Tensor &mask,
                            at::Tensor &workspace_qk,
                            at::Tensor &workspace_qs,
                            at::Tensor &workspace_gated,
                            at::Tensor &out,
                            const at::Tensor &cu_seqlens,
                            int64_t block_dim,
                            int64_t batch_size,
                            int64_t seq_len,
                            int64_t total_tokens)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q and k must have shape [1, T, Hg, D]");
    TORCH_CHECK(v.dim() == 4 && out.dim() == 4, "v and out must have shape [1, T, H, D]");
    TORCH_CHECK(s.dim() == 3, "s must have shape [total_chunks * H, D, D]");
    TORCH_CHECK(g_t.dim() == 2, "g_t must have shape [H, T]");
    TORCH_CHECK(mask.dim() == 2, "mask must have shape [128, 128]");
    TORCH_CHECK(q.size(0) == 1 && k.size(0) == 1 && v.size(0) == 1,
                "chunk_o_debug expects packed B=1 input");
    TORCH_CHECK(q.size(2) == 16 && k.size(2) == 16, "chunk_o_debug was built for Hg=16");
    TORCH_CHECK(v.size(2) == 16 && out.size(2) == 16, "chunk_o_debug was built for H=16");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128,
                "chunk_o_debug was built for D=128");
    TORCH_CHECK(g_t.size(0) == 16 && g_t.size(1) == total_tokens,
                "g_t must have shape [16, total_tokens]");
    TORCH_CHECK(mask.size(0) == 128 && mask.size(1) == 128,
                "mask must have shape [128, 128]");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf &&
                    v.scalar_type() == at::kHalf && s.scalar_type() == at::kHalf &&
                    out.scalar_type() == at::kHalf,
                "q, k, v, s, and out must be float16");
    TORCH_CHECK(g_t.scalar_type() == at::kFloat && mask.scalar_type() == at::kFloat,
                "g_t and mask must be float32");
    TORCH_CHECK(workspace_qk.scalar_type() == at::kHalf &&
                    workspace_qs.scalar_type() == at::kHalf &&
                    workspace_gated.scalar_type() == at::kHalf,
                "workspaces must be float16");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens.numel() == batch_size + 1, "batch_size must match cu_seqlens");
    TORCH_CHECK(seq_len == q.size(1), "seq_len must match q.shape[1]");
    TORCH_CHECK(total_tokens == g_t.size(1), "total_tokens must match g_t.shape[1]");
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(),
                "block_dim is out of uint32 range");

    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);

    EXEC_KERNEL_CMD(launch_chunk_o_debug, block_dim_u32, q, k, v, s, g_t, mask,
                    workspace_qk, workspace_qs, workspace_gated, out, cu_seqlens,
                    batch_size, seq_len, total_tokens, ffts_addr);
}

}  // namespace npu_kernel
}  // namespace sglang
