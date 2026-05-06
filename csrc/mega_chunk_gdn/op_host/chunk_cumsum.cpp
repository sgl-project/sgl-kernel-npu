// Minimal host launcher for debugging the PTO chunk_cumsum kernel.

#include <cstdint>
#include <limits>

#include "aclrtlaunch_launch_chunk_cumsum.h"
#include "defines.h"
#include "runtime/rt_ffts.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

HOST_API void chunk_cumsum_debug(const at::Tensor &g,
                                 at::Tensor &g_sum,
                                 const at::Tensor &cu_seqlens,
                                 int64_t block_dim,
                                 int64_t batch_size,
                                 int64_t seq_len)
{
    TORCH_CHECK(g.dim() == 3, "g must have shape [1, T, H]");
    TORCH_CHECK(g.size(0) == 1, "chunk_cumsum_debug expects packed B=1 input");
    TORCH_CHECK(g.size(2) == 16, "chunk_cumsum_debug was built for H=16");
    TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32");
    TORCH_CHECK(g_sum.sizes() == g.sizes(), "g_sum must have the same shape as g");
    TORCH_CHECK(g_sum.scalar_type() == at::kFloat, "g_sum must be float32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least [0, T]");
    TORCH_CHECK(cu_seqlens.numel() == batch_size + 1, "batch_size must match cu_seqlens");
    TORCH_CHECK(seq_len == g.size(1), "seq_len must match g.shape[1]");
    TORCH_CHECK(g.is_contiguous() && g_sum.is_contiguous() && cu_seqlens.is_contiguous(),
                "g, g_sum, and cu_seqlens must be contiguous");
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(),
                "block_dim is out of uint32 range");

    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);

    EXEC_KERNEL_CMD(launch_chunk_cumsum, block_dim_u32, g, g_sum, cu_seqlens,
                    batch_size, seq_len, ffts_addr);
}

}  // namespace npu_kernel
}  // namespace sglang
