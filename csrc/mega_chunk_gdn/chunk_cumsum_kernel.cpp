// Minimal standalone wrapper for debugging chunk_cumsum through torch.ops.

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include "chunk_cumsum.cpp"

extern "C" __global__ AICORE void launch_chunk_cumsum(
    __gm__ uint8_t* g_ptr,
    __gm__ uint8_t* g_sum_ptr,
    __gm__ uint8_t* cu_seqlens_ptr,
    int64_t batch_size,
    int64_t seq_len,
    uint64_t sync_addr)
{
    cumsum_kernel<GDN_H, GDN_C>(
        reinterpret_cast<__gm__ float*>(g_ptr),
        reinterpret_cast<__gm__ float*>(g_sum_ptr),
        reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
        batch_size, seq_len, sync_addr);
}
