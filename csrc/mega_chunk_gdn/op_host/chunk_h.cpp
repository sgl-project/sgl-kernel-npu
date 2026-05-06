// Minimal host launcher for debugging the PTO chunk_h kernel.

#include <cstdint>
#include <limits>

#include "aclrtlaunch_launch_chunk_h_debug.h"
#include "defines.h"
#include "runtime/rt_ffts.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

HOST_API void chunk_h_debug(const at::Tensor &k,
                            const at::Tensor &w,
                            const at::Tensor &u,
                            const at::Tensor &g_t,
                            at::Tensor &s,
                            at::Tensor &v_new,
                            at::Tensor &final_state,
                            at::Tensor &workspace,
                            const at::Tensor &cu_seqlens,
                            int64_t block_dim,
                            int64_t batch_size,
                            int64_t seq_len,
                            int64_t total_tokens)
{
    TORCH_CHECK(k.dim() == 4, "k must have shape [1, T, Hg, D]");
    TORCH_CHECK(w.dim() == 4 && u.dim() == 4, "w and u must have shape [1, T, H, D]");
    TORCH_CHECK(g_t.dim() == 2, "g_t must have shape [H, T]");
    TORCH_CHECK(k.size(0) == 1 && w.size(0) == 1 && u.size(0) == 1,
                "chunk_h_debug expects packed B=1 input");
    TORCH_CHECK(k.size(2) == 16, "chunk_h_debug was built for Hg=16");
    TORCH_CHECK(w.size(2) == 16 && u.size(2) == 16, "chunk_h_debug was built for H=16");
    TORCH_CHECK(k.size(3) == 128 && w.size(3) == 128 && u.size(3) == 128,
                "chunk_h_debug was built for D=128");
    TORCH_CHECK(g_t.size(0) == 16 && g_t.size(1) == total_tokens,
                "g_t must have shape [16, total_tokens]");
    TORCH_CHECK(k.scalar_type() == at::kHalf, "k must be float16");
    TORCH_CHECK(w.scalar_type() == at::kHalf, "w must be float16");
    TORCH_CHECK(u.scalar_type() == at::kHalf, "u must be float16");
    TORCH_CHECK(g_t.scalar_type() == at::kFloat, "g_t must be float32");
    TORCH_CHECK(s.scalar_type() == at::kHalf, "s must be float16");
    TORCH_CHECK(v_new.scalar_type() == at::kHalf, "v_new must be float16");
    TORCH_CHECK(final_state.scalar_type() == at::kHalf, "final_state must be float16");
    TORCH_CHECK(workspace.scalar_type() == at::kHalf, "workspace must be float16");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens.numel() == batch_size + 1, "batch_size must match cu_seqlens");
    TORCH_CHECK(seq_len == k.size(1), "seq_len must match k.shape[1]");
    TORCH_CHECK(total_tokens == g_t.size(1), "total_tokens must match g_t.shape[1]");
    TORCH_CHECK(k.is_contiguous() && w.is_contiguous() && u.is_contiguous() &&
                    g_t.is_contiguous() && s.is_contiguous() && v_new.is_contiguous() &&
                    final_state.is_contiguous() && workspace.is_contiguous() &&
                    cu_seqlens.is_contiguous(),
                "all chunk_h_debug tensors must be contiguous");
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(),
                "block_dim is out of uint32 range");

    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);

    EXEC_KERNEL_CMD(launch_chunk_h_debug, block_dim_u32, k, w, u, g_t, s, v_new,
                    final_state, workspace, cu_seqlens, batch_size, seq_len,
                    total_tokens, ffts_addr);
}

}  // namespace npu_kernel
}  // namespace sglang
