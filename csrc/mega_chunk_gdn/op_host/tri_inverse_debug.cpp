// Minimal host launcher for debugging tri_inverse_impl through torch.ops.

#include <cstdint>
#include <limits>

#include "aclrtlaunch_launch_tri_inverse_debug.h"
#include "defines.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

HOST_API void tri_inverse_debug(at::Tensor &tensor_out,
                                const at::Tensor &tensor_in,
                                const at::Tensor &minus_identity,
                                const at::Tensor &cu_seqlens,
                                int64_t block_dim,
                                int64_t matrix_size,
                                int64_t num_matrices,
                                int64_t num_bsnd_heads,
                                bool is_lower)
{
    TORCH_CHECK(tensor_out.scalar_type() == at::kFloat, "tensor_out must be float32");
    TORCH_CHECK(tensor_in.scalar_type() == at::kHalf, "tensor_in must be float16");
    TORCH_CHECK(minus_identity.scalar_type() == at::kHalf, "minus_identity must be float16");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(tensor_out.sizes() == tensor_in.sizes(), "tensor_out shape must match tensor_in");
    TORCH_CHECK(matrix_size == 16 || matrix_size == 32 || matrix_size == 64 || matrix_size == 128,
                "matrix_size must be one of 16, 32, 64, 128");
    TORCH_CHECK(minus_identity.numel() == matrix_size * matrix_size,
                "minus_identity must have matrix_size * matrix_size elements");
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(),
                "block_dim is out of uint32 range");
    TORCH_CHECK(num_matrices >= 0 && num_matrices <= std::numeric_limits<uint32_t>::max(),
                "num_matrices is out of uint32 range");
    TORCH_CHECK(num_bsnd_heads >= 0 && num_bsnd_heads <= 0xFFFF,
                "num_bsnd_heads is out of range");
    TORCH_CHECK(tensor_out.is_contiguous() && tensor_in.is_contiguous() &&
                    minus_identity.is_contiguous() && cu_seqlens.is_contiguous(),
                "all tri_inverse_debug tensors must be contiguous");

    uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
    uint32_t matrix_size_u32 = static_cast<uint32_t>(matrix_size);
    uint32_t num_matrices_u32 = static_cast<uint32_t>(num_matrices);
    uint32_t heads_with_flag = static_cast<uint32_t>(num_bsnd_heads);
    if (is_lower) {
        heads_with_flag |= 0x10000u;
    }

    EXEC_KERNEL_CMD(launch_tri_inverse_debug, block_dim_u32, tensor_out, tensor_in,
                    minus_identity, cu_seqlens, matrix_size_u32, num_matrices_u32,
                    heads_with_flag);
}

}  // namespace npu_kernel
}  // namespace sglang
