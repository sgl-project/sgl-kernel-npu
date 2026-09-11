#include <algorithm>
#include <cstdint>
#include <string>

#include "torch_helper.h"
#include "sgl_kenel_npu_ops.h"

namespace sglang::npu_kernel {

at::Tensor kv_quant_sparse_attn_sharedkv_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t /*kv_quant_mode*/,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size, int64_t max_seqlen_q,
    int64_t max_seqlen_kv, int64_t ori_topk, int64_t cmp_topk, int64_t /*tile_size*/,
    int64_t /*rope_head_dim*/, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv,
    bool has_ori_kv, bool has_cmp_kv, c10::string_view device) {
    TORCH_CHECK(seqused_kv.has_value() || (cu_seqlens_ori_kv.has_value() && std::string(layout_kv) == "TND"),
                "kv_quant_sparse_attn_sharedkv_metadata: KV sequence lengths are required");
    at::Device output_device{std::string(device)};
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q->device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv->device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv->device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q->device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv->device();
    }
    auto cpu_q = cu_seqlens_q.has_value()
                     ? c10::optional<at::Tensor>(cu_seqlens_q->to(at::kCPU).contiguous())
                     : c10::nullopt;
    auto cpu_seq_q = seqused_q.has_value()
                         ? c10::optional<at::Tensor>(seqused_q->to(at::kCPU).contiguous())
                         : c10::nullopt;
    auto cpu_ori_kv = cu_seqlens_ori_kv.has_value()
                          ? c10::optional<at::Tensor>(cu_seqlens_ori_kv->to(at::kCPU).contiguous())
                          : c10::nullopt;
    auto cpu_kv = seqused_kv.has_value()
                      ? c10::optional<at::Tensor>(seqused_kv->to(at::kCPU).contiguous())
                      : c10::nullopt;
    uint32_t max_s2_g_base_num = 0;
    auto old = sparse_attn_sharedkv_metadata_host_with_max_s2(
        num_heads_q, num_heads_kv, head_dim, std::string(layout_q), std::string(layout_kv), cpu_q, cpu_kv,
        batch_size, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        has_ori_kv, has_cmp_kv, &max_s2_g_base_num,
        cpu_seq_q.has_value() ? cpu_seq_q->data_ptr<int32_t>() : nullptr,
        cpu_ori_kv.has_value() ? cpu_ori_kv->data_ptr<int32_t>() : nullptr,
        static_cast<int32_t>(max_seqlen_q), static_cast<int32_t>(max_seqlen_kv)).to(at::kCPU);

    constexpr int32_t AIC = 36;
    constexpr int32_t AIV = 72;
    constexpr int32_t OLD_FA = 8;
    constexpr int32_t NEW_FA = 9;
    auto host = at::zeros({1024}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    const auto *src = old.data_ptr<int32_t>();
    auto *dst = host.data_ptr<int32_t>();
    int32_t max_s2 = static_cast<int32_t>(max_s2_g_base_num);
    if (max_seqlen_kv == 0) {
        int32_t kv_len = 0;
        if (cpu_kv.has_value()) {
            kv_len = cpu_kv->max().item<int32_t>();
        } else if (cpu_ori_kv.has_value()) {
            kv_len = cpu_ori_kv->view(-1)[-1].item<int32_t>();
        }
        max_s2 = std::max(max_s2, (kv_len + 127) / 128);
    }
    const bool is_n128 = num_heads_q == 128;
    const int32_t scheduler_aic = is_n128 ? AIC / 2 : AIC;
    for (int32_t i = 0; i < scheduler_aic; ++i) {
        const int32_t records = is_n128 ? 2 : 1;
        for (int32_t copy = 0; copy < records; ++copy) {
            for (int32_t j = 0; j < OLD_FA; ++j) {
                dst[(records * i + copy) * NEW_FA + j] = src[i * OLD_FA + j];
            }
            dst[(records * i + copy) * NEW_FA + 8] = max_s2;
        }
    }
    for (int32_t i = 0; i < AIV * OLD_FA; ++i) dst[AIC * NEW_FA + i] = src[AIC * OLD_FA + i];
    return host.to(output_device);
}
}  // namespace sglang::npu_kernel
