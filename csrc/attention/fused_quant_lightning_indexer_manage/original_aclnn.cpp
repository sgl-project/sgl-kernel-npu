// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Production bridge to the pinned original A5 ACLNN adapter.

#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <string>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "a5_c8_ops.h"
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

namespace sglang::npu_kernel {

void fused_quant_lightning_indexer_manage(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, const at::Tensor &cache_slots_pool, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_dst_slots, const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts)
{
    if (query.is_meta()) {
        sgl_a5_original::check_fused_quant_lightning_indexer_manage_arch35(
            index_weights, query_dequant_scale, query, index_key_dequant_scale, index_key_cache, index_block_table,
            actual_seq_lengths_query, actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens, request_state,
            req_pool_entries, cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
            miss_dst_slots, miss_counts);
    }
    // The original adapter validates once before submission. Keep only the
    // extra addressing guards here and use its validation-only path for Meta.
    TORCH_CHECK(index_weights.dim() == 2 && index_key_cache.dim() == 4 && index_key_dequant_scale.dim() == 3,
                "weights/key/key-scale dimensions must be 2/4/3");
    TORCH_CHECK(index_key_cache.stride(0) >= 128 * 128 && index_key_dequant_scale.stride(0) >= 128,
                "key and key-scale blocks must not overlap.");
    for (auto stride : {index_weights.stride(0), index_key_cache.stride(0), index_key_dequant_scale.stride(0)}) {
        TORCH_CHECK(stride <= UINT32_MAX, "quant LIM stride exceeds uint32");
    }
    if (query.is_meta()) {
        return;
    }
    c10_npu::OptionalNPUGuard guard(query.device());
    sgl_a5_original::npu_fused_quant_lightning_indexer_manage_arch35(
        index_weights, query_dequant_scale, query, index_key_dequant_scale, index_key_cache, index_block_table,
        actual_seq_lengths_query, actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens, request_state,
        req_pool_entries, cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
        miss_dst_slots, miss_counts);
}

c10::Dict<std::string, std::string> resolve_a5_c8_original_libraries()
{
    c10::Dict<std::string, std::string> result;
    for (const char *symbol : {"aclnnFusedQuantLightningIndexerManageWithStridesGetWorkspaceSize",
                               "aclnnFusedQuantLightningIndexerManageWithStrides"}) {
        void *address = GetOpApiFuncAddr(symbol);
        TORCH_CHECK(address != nullptr, "Could not resolve original ACLNN symbol: ", symbol);
        Dl_info info{};
        TORCH_CHECK(dladdr(address, &info) != 0 && info.dli_fname != nullptr,
                    "Could not identify defining library for ", symbol);
        char resolved[PATH_MAX];
        TORCH_CHECK(realpath(info.dli_fname, resolved) != nullptr, "Could not resolve library path for ", symbol);
        result.insert(std::string(symbol), std::string(resolved));
    }
    return result;
}

}  // namespace sglang::npu_kernel
