// Minimal standalone wrapper for debugging tri_inverse_impl through torch.ops.

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include "tri_inverse_impl.cpp"

extern "C" __global__ AICORE void launch_tri_inverse_debug(
    __gm__ uint8_t* tensor_out_ptr,
    __gm__ uint8_t* tensor_in_ptr,
    __gm__ uint8_t* minus_identity_ptr,
    __gm__ uint8_t* cu_seqlens_ptr,
    uint32_t matrix_size,
    uint32_t num_matrices,
    uint32_t num_bsnd_heads)
{
    const uint32_t is_lower = (num_bsnd_heads >> 16) & 1u;
    const uint32_t actual_heads = num_bsnd_heads & 0xFFFFu;
    if (actual_heads == 0) {
        if (num_matrices <= get_block_num()) {
            run_tri_inv_rec_unroll<half, 1, false>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        } else if (num_matrices <= 2u * get_block_num()) {
            run_tri_inv_rec_unroll<half, 2, false>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        } else {
            run_tri_inv_rec_unroll<half, 4, false>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        }
    } else {
        if (num_matrices <= get_block_num()) {
            run_tri_inv_rec_unroll<half, 1, true>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        } else if (num_matrices <= 2u * get_block_num()) {
            run_tri_inv_rec_unroll<half, 2, true>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        } else {
            run_tri_inv_rec_unroll<half, 4, true>(
                reinterpret_cast<__gm__ float*>(tensor_out_ptr),
                reinterpret_cast<__gm__ half*>(tensor_in_ptr),
                reinterpret_cast<__gm__ half*>(minus_identity_ptr),
                matrix_size, num_matrices, actual_heads,
                reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr), is_lower);
        }
    }
}
