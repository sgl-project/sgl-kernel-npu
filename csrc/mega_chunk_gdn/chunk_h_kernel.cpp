// Minimal standalone wrapper for debugging chunk_h through torch.ops.

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include "chunk_h.cpp"

extern "C" __global__ AICORE void launch_chunk_h_debug(
    __gm__ uint8_t* k_ptr,
    __gm__ uint8_t* w_ptr,
    __gm__ uint8_t* u_ptr,
    __gm__ uint8_t* g_ptr,
    __gm__ uint8_t* s_ptr,
    __gm__ uint8_t* v_ptr,
    __gm__ uint8_t* fs_ptr,
    __gm__ uint8_t* workspace_ptr,
    __gm__ uint8_t* cu_seqlens_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens,
    uint64_t sync_addr)
{
    chunk_h_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
        reinterpret_cast<__gm__ half*>(k_ptr),
        reinterpret_cast<__gm__ half*>(w_ptr),
        reinterpret_cast<__gm__ half*>(u_ptr),
        reinterpret_cast<__gm__ float*>(g_ptr),
        reinterpret_cast<__gm__ half*>(s_ptr),
        reinterpret_cast<__gm__ half*>(v_ptr),
        reinterpret_cast<__gm__ half*>(fs_ptr),
        reinterpret_cast<__gm__ half*>(workspace_ptr),
        reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
        batch_size, seq_len, total_tokens, sync_addr);
}
