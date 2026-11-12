#include <ATen/ATen.h>
#include <torch/library.h>
#include <string>
namespace sglang::npu_kernel {
at::Tensor kv_quant_sparse_attn_sharedkv_metadata_host(int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
                                                       const std::string &layout_q, const std::string &layout_kv,
                                                       const c10::optional<at::Tensor> &cu_seqlens_q,
                                                       const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size,
                                                       int64_t max_seqlen_q, int64_t max_seqlen_kv, int64_t cmp_topk,
                                                       int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
                                                       int64_t ori_win_left, int64_t ori_win_right, bool has_ori_kv,
                                                       bool has_cmp_kv, int64_t kv_quant_mode, int64_t tile_size,
                                                       int64_t rope_head_dim);
}
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "kv_quant_sparse_attn_sharedkv_metadata_host("
        "int num_heads_q, int num_heads_kv, int head_dim, "
        "str layout_q, str layout_kv, "
        "Tensor? cu_seqlens_q=None, Tensor? seqused_kv=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_kv=0, "
        "int cmp_topk=0, int cmp_ratio=1, int ori_mask_mode=4, int cmp_mask_mode=3, "
        "int ori_win_left=127, int ori_win_right=0, "
        "bool has_ori_kv=True, bool has_cmp_kv=True, "
        "int kv_quant_mode=1, int tile_size=64, int rope_head_dim=64) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, CatchAll, m)
{
    m.impl("kv_quant_sparse_attn_sharedkv_metadata_host",
           TORCH_FN(sglang::npu_kernel::kv_quant_sparse_attn_sharedkv_metadata_host));
}
