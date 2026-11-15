// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

namespace sglang::npu_kernel {

void fused_quant_lightning_indexer_manage(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, const at::Tensor &cache_slots_pool, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_dst_slots, const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts);

}  // namespace sglang::npu_kernel
