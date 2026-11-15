// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// A5-only normal-wheel registration; no vLLM Python extension dependency.

#include "a5_c8_ops.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
c10::Dict<std::string, std::string> resolve_a5_c8_original_libraries();
}

namespace {

std::string A5C8Backend()
{
    return "original";
}

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("a5_c8_backend() -> str", &A5C8Backend);
    m.def("resolve_a5_c8_original_libraries() -> Dict(str, str)",
          &sglang::npu_kernel::resolve_a5_c8_original_libraries);
    m.def(
        "fused_quant_lightning_indexer_manage(Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
        "Tensor index_key_dequant_scale, Tensor index_key_cache, Tensor index_block_table, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
        "Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
        "Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, "
        "Tensor(e!) miss_src_ids, Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_quant_lightning_indexer_manage", TORCH_FN(sglang::npu_kernel::fused_quant_lightning_indexer_manage));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("fused_quant_lightning_indexer_manage", TORCH_FN(sglang::npu_kernel::fused_quant_lightning_indexer_manage));
}

}  // namespace
