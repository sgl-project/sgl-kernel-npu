// Minimal standalone wrapper for debugging chunk_o through torch.ops.

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include "chunk_o.cpp"

extern "C" __global__ AICORE void launch_chunk_o_debug(
    __gm__ uint8_t* q_ptr,
    __gm__ uint8_t* k_ptr,
    __gm__ uint8_t* v_ptr,
    __gm__ uint8_t* s_ptr,
    __gm__ uint8_t* g_ptr,
    __gm__ uint8_t* mask_ptr,
    __gm__ uint8_t* workspace_qk_ptr,
    __gm__ uint8_t* workspace_qs_ptr,
    __gm__ uint8_t* workspace_gated_ptr,
    __gm__ uint8_t* o_ptr,
    __gm__ uint8_t* cu_seqlens_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens)
{
    chunk_o_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
        reinterpret_cast<__gm__ half*>(q_ptr),
        reinterpret_cast<__gm__ half*>(k_ptr),
        reinterpret_cast<__gm__ half*>(v_ptr),
        reinterpret_cast<__gm__ half*>(s_ptr),
        reinterpret_cast<__gm__ float*>(g_ptr),
        reinterpret_cast<__gm__ float*>(mask_ptr),
        reinterpret_cast<__gm__ half*>(workspace_qk_ptr),
        reinterpret_cast<__gm__ half*>(workspace_qs_ptr),
        reinterpret_cast<__gm__ half*>(workspace_gated_ptr),
        reinterpret_cast<__gm__ half*>(o_ptr),
        reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens);
}
