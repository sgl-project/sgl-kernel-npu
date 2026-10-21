// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0.
#include <algorithm>
#include <limits>
#include "defines.h"
#include "ge_helper.h"
#include "torch_helper.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_lightning_indexer_manage_tiling_data.h"
#include "../op_kernel/upstream/fused_lightning_indexer_manage_constants.h"
#include "aclrtlaunch_fused_lightning_indexer_manage.h"

namespace sglang::npu_kernel {
HOST_API void fused_lightning_indexer_manage(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, at::Tensor cache_slots_pool, at::Tensor topk_src_ids, at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts, at::Tensor miss_src_ids, at::Tensor miss_dst_slots, at::Tensor miss_counts)
{
    constexpr int64_t kTopK = LIMConfig::TOPK;
    constexpr int64_t kMissCapacity = LIMConfig::MISS_CAPACITY;
    constexpr int64_t kBlockSize = LIMConfig::CACHE_BLOCK_SIZE;
    TORCH_CHECK(query.dim() == 3 &&
                    (query.size(1) == LIMConfig::QUERY_HEADS_SMALL || query.size(1) == LIMConfig::QUERY_HEADS_LARGE) &&
                    query.size(2) == LIMConfig::HEAD_DIM && query.size(0) > 0,
                "LIM-MTP query must be [T, H, 128], H=32 or 64 and T>0.");
    TORCH_CHECK(query.device().is_privateuseone(), "LIM-MTP tensors must be on NPU.");
    const int64_t total_queries = query.size(0);
    TORCH_CHECK(
        index_weights.dim() == 2 && index_weights.size(0) == total_queries && index_weights.size(1) == query.size(1),
        "index_weights must be [T, H].");
    TORCH_CHECK(query_dequant_scale.sizes() == index_weights.sizes(), "query_dequant_scale must be [T, H].");
    TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 && index_key_cache.size(1) == kBlockSize &&
                    index_key_cache.size(2) == LIMConfig::KEY_HEADS && index_key_cache.size(3) == LIMConfig::HEAD_DIM,
                "index_key_cache must be [blocks, 128, 1, 128].");
    TORCH_CHECK(index_key_dequant_scale.dim() == 3 && index_key_dequant_scale.size(0) == index_key_cache.size(0) &&
                    index_key_dequant_scale.size(1) == kBlockSize &&
                    index_key_dequant_scale.size(2) == LIMConfig::KEY_HEADS,
                "index_key_dequant_scale must be [blocks, 128, 1].");
    TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) > 0 && index_block_table.size(1) > 0 &&
                    index_block_table.size(1) <= LIMConfig::MAX_BLOCKS_PER_REQUEST,
                "index_block_table must be non-empty [B, max_blocks], max_blocks<=16384.");
    const int64_t batch_size = index_block_table.size(0);
    TORCH_CHECK(total_queries >= batch_size && total_queries <= batch_size * LIMConfig::MAX_ROUTES,
                "total_queries must satisfy B <= T <= 14B.");
    auto check_batch_vector = [batch_size](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch_size, name, " must be [B].");
    };
    check_batch_vector(actual_seq_lengths_query, "actual_seq_lengths_query");
    check_batch_vector(actual_seq_lengths_key, "actual_seq_lengths_key");
    check_batch_vector(offload_seq_lengths_key, "offload_seq_lengths_key");
    check_batch_vector(num_cache_tokens, "num_cache_tokens");
    check_batch_vector(request_state, "request_state");
    check_batch_vector(req_pool_entries, "req_pool_entries");
    TORCH_CHECK(cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
                    cache_slots_pool.size(1) == index_block_table.size(1) * kBlockSize,
                "cache_slots_pool width must equal max_blocks*128.");
    TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(0) == total_queries && topk_src_ids.size(1) == 1 &&
                    topk_src_ids.size(2) == kTopK && topk_dst_slots.sizes() == topk_src_ids.sizes(),
                "topk outputs must be [T, 1, 2048].");
    TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries,
                "topk_miss_counts must be [T].");
    TORCH_CHECK(miss_src_ids.dim() == 2 && miss_src_ids.size(0) == batch_size &&
                    miss_src_ids.size(1) == kMissCapacity && miss_dst_slots.sizes() == miss_src_ids.sizes(),
                "miss outputs must be [B, 32768].");
    check_batch_vector(miss_counts, "miss_counts");

    TORCH_CHECK(query.scalar_type() == index_key_cache.scalar_type() &&
                    query.scalar_type() == index_weights.scalar_type() &&
                    (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16),
                "query/index_key_cache/index_weights must share fp16 or bf16 dtype.");
    TORCH_CHECK(query_dequant_scale.scalar_type() == at::kFloat && index_key_dequant_scale.scalar_type() == at::kFloat,
                "dequant scales must be fp32.");
    const at::Tensor int_tensors[] = {index_block_table,      actual_seq_lengths_query,
                                      actual_seq_lengths_key, offload_seq_lengths_key,
                                      num_cache_tokens,       request_state,
                                      req_pool_entries,       cache_slots_pool,
                                      topk_src_ids,           topk_dst_slots,
                                      topk_miss_counts,       miss_src_ids,
                                      miss_dst_slots,         miss_counts};
    for (const auto &tensor : int_tensors) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt, "metadata and outputs must be int32.");
    }
    const at::Tensor all_tensors[] = {index_weights,
                                      query_dequant_scale,
                                      query,
                                      index_key_dequant_scale,
                                      index_key_cache,
                                      index_block_table,
                                      actual_seq_lengths_query,
                                      actual_seq_lengths_key,
                                      offload_seq_lengths_key,
                                      num_cache_tokens,
                                      request_state,
                                      req_pool_entries,
                                      cache_slots_pool,
                                      topk_src_ids,
                                      topk_dst_slots,
                                      topk_miss_counts,
                                      miss_src_ids,
                                      miss_dst_slots,
                                      miss_counts};
    const auto device = query.device();
    for (const auto &tensor : all_tensors) {
        TORCH_CHECK(tensor.device() == device, "all LIM-MTP tensors must be on the same NPU.");
        TORCH_CHECK(tensor.is_contiguous(), "all LIM-MTP tensors must be contiguous.");
    }

    c10_npu::OptionalNPUGuard guard(query.device());
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "LIM: Ascend platform is unavailable");
    auto soc = platform->GetSocVersion();
    TORCH_CHECK(soc == platform_ascendc::SocVersion::ASCEND910B || soc == platform_ascendc::SocVersion::ASCEND910_93,
                "LIM requires Ascend A2/A3");
    uint32_t aic = platform->GetCoreNumAic();
    uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic > 0 && aiv > 0, "LIM: no usable Ascend cores");
    TORCH_CHECK(batch_size <= UINT32_MAX && total_queries <= UINT32_MAX && cache_slots_pool.size(0) <= UINT32_MAX,
                "LIM dimensions exceed uint32");
    const uint32_t schedule = batch_size % aic == 0 ? 2U : 1U;
    const uint32_t owners = schedule == 1U ? aic : std::min<uint32_t>(batch_size, aic);
    const uint32_t blockDim = platform->CalcTschBlockDim(std::min(aiv, owners * 2U), aic, aiv);
    FusedLightningIndexerManageTilingData tiling{};
    tiling.bSize = batch_size;
    tiling.tSize = total_queries;
    tiling.s2Size = cache_slots_pool.size(1);
    tiling.usedCoreNum = blockDim;
    tiling.blockSize = 128;
    tiling.maxBlockNumPerBatch = index_block_table.size(1);
    tiling.poolSize = cache_slots_pool.size(0);
    tiling.n1Size = query.size(1);
    tiling.cacheSlotsSize = cache_slots_pool.size(1);
    tiling.scheduleMode = schedule;
    tiling.dtype = query.scalar_type() == at::kHalf ? 1U : 27U;
    auto tilingTensor = ge_helper::TilingTensorCache<FusedLightningIndexerManageTilingData>::Get(
        tiling, "fused_lightning_indexer_manage");
    const uint64_t bytes = LimWorkspaceBytes(tiling, platform->GetLibApiWorkSpaceSize());
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    EXEC_KERNEL_CMD(fused_lightning_indexer_manage, blockDim, index_weights, query_dequant_scale, query,
                    index_key_dequant_scale, index_key_cache, index_block_table, actual_seq_lengths_query,
                    actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens, request_state, req_pool_entries,
                    cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
                    miss_counts, cache_slots_pool, workspace, tilingTensor);
}
}  // namespace sglang::npu_kernel
