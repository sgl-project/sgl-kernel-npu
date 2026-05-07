#include <cstdint>

extern "C" uint32_t aclrtlaunch_launch_chunk_o_debug(
    uint32_t blockDim,
    void* stream,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    void* s_ptr,
    void* g_ptr,
    void* mask_ptr,
    void* workspace_qk_ptr,
    void* workspace_qs_ptr,
    void* workspace_gated_ptr,
    void* o_ptr,
    void* cu_seqlens_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens);

extern "C" uint32_t call_chunk_o_debug_stub(
    uint32_t blockDim,
    void* stream,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    void* s_ptr,
    void* g_ptr,
    void* mask_ptr,
    void* workspace_qk_ptr,
    void* workspace_qs_ptr,
    void* workspace_gated_ptr,
    void* o_ptr,
    void* cu_seqlens_ptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t total_tokens) {
  return aclrtlaunch_launch_chunk_o_debug(
      blockDim, stream, q_ptr, k_ptr, v_ptr, s_ptr, g_ptr, mask_ptr,
      workspace_qk_ptr, workspace_qs_ptr, workspace_gated_ptr, o_ptr,
      cu_seqlens_ptr, batch_size, seq_len, total_tokens);
}
