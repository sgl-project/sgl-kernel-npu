// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#pragma once

#include <ATen/ATen.h>

namespace sglang::npu_kernel {

void LaunchFusedCopySfa(const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &hbm_ckv,
                        at::Tensor *hbm_rope, const at::Tensor &actual_seq_lengths_query,
                        const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens,
                        const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
                        const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
                        const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts,
                        const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table, int64_t dram_kv_dev_ptr,
                        at::Tensor &attention_out, double scale_value, bool full_visible, bool resident_refresh_tail,
                        const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
                        const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
                        const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
                        int64_t host_token_capacity, int64_t dram_rope_dev_ptr);

void fused_scatter_copy_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &combined_hbm_cache,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_kv,
    const at::Tensor &num_cache_tokens, const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids, const at::Tensor &miss_dst_slots,
    const at::Tensor &miss_counts, const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table,
    int64_t dram_kv_dev_ptr, at::Tensor &attention_out, double scale_value, bool full_visible,
    bool resident_refresh_tail, const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
    const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
    const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
    int64_t host_token_capacity, int64_t dram_rope_dev_ptr);

void fused_scatter_copy_sparse_flash_attention_split(
    const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &combined_hbm_cache, at::Tensor &split_hbm_rope,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_kv,
    const at::Tensor &num_cache_tokens, const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids, const at::Tensor &miss_dst_slots,
    const at::Tensor &miss_counts, const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table,
    int64_t dram_kv_dev_ptr, at::Tensor &attention_out, double scale_value, bool full_visible,
    bool resident_refresh_tail, const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
    const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
    const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
    int64_t host_token_capacity, int64_t dram_rope_dev_ptr);

}  // namespace sglang::npu_kernel
