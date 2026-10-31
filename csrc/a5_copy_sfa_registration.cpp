// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// A5-only normal-wheel registration. This file deliberately has no LIM or
// external runtime dependency.
#include "a5_copy_sfa_ops.h"
#include "torch_helper.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "fused_quant_scatter_copy_sparse_flash_attention("
        "Tensor query, Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
        "int dram_kv_dev_ptr, Tensor hbm_block_table, "
        "Tensor dram_block_table, Tensor topk_dst_slots, "
        "Tensor topk_src_ids, Tensor topk_miss_counts, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
        "Tensor num_cache_tokens, Tensor miss_src_ids, "
        "Tensor miss_dst_slots, Tensor miss_counts, float scale_value, "
        "Tensor(b!) attention_out) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
}

}  // namespace
