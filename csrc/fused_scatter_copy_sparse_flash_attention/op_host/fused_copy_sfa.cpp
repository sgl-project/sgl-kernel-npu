// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>

#include "aclrtlaunch_fused_scatter_copy_sparse_flash_attention_combined.h"
#include "fused_copy_sfa_tiling_data.h"
#include "ge_helper.h"
#include "sparse_flash_attention_def.h"
#include "sparse_flash_attention_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kCacheCapacity = 2048;
constexpr int64_t kPageSize = 128;
constexpr int64_t kKvDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kTokenStride = 576;

void CheckInt32(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckNpuTensor(const at::Tensor &tensor, const at::Tensor &query, const char *name)
{
    TORCH_CHECK(tensor.device() == query.device(), name, " must be on query's NPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void LaunchCombined(uint32_t blockDim, const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                    const at::Tensor &sparseIndices, const at::Tensor &cacheTokens, const at::Tensor &hbmBlockTable,
                    const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv,
                    const at::Tensor &queryRope, void *hbmKeyRope, void *dramKeyRope, void *dramKvCache,
                    const at::Tensor &dramBlockTable, const at::Tensor &topkSourceIds, const at::Tensor &topkMissCounts,
                    const at::Tensor &missSourceIds, const at::Tensor &missDstSlots, const at::Tensor &missCounts,
                    const at::Tensor &attentionOut, const at::Tensor &workspace, const at::Tensor &tiling)
{
    EXEC_KERNEL_CMD(fused_scatter_copy_sparse_flash_attention_combined, blockDim, query, key, value, sparseIndices,
                    cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope,
                    dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts, missSourceIds,
                    missDstSlots, missCounts, attentionOut, workspace, tiling);
}

}  // namespace

void fused_scatter_copy_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &combined_hbm_cache,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_kv,
    const at::Tensor &num_cache_tokens, const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids, const at::Tensor &miss_dst_slots,
    const at::Tensor &miss_counts, const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table,
    int64_t dram_kv_dev_ptr, at::Tensor &attention_out, double scale_value)
{
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be float16 or bfloat16");
    TORCH_CHECK(query.device().type() == DEVICE_TYPE, "query must be an NPU tensor");
    TORCH_CHECK(
        query.dim() == 3 && query.size(0) > 0 && query.size(1) > 0 && query.size(2) == kKvDim && query.is_contiguous(),
        "query must be TND [B, H, 512]");
    TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == query.size(0) && query_rope.size(1) == query.size(1) &&
                    query_rope.size(2) == kRopeDim,
                "query_rope must be [B, H, 64]");
    TORCH_CHECK(combined_hbm_cache.scalar_type() == query.scalar_type() && combined_hbm_cache.dim() == 4 &&
                    combined_hbm_cache.size(1) == kPageSize && combined_hbm_cache.size(2) == 1 &&
                    combined_hbm_cache.size(3) == kTokenStride && combined_hbm_cache.is_contiguous(),
                "combined_hbm_cache must be [R*16, 128, 1, 576] and match query dtype");
    TORCH_CHECK(combined_hbm_cache.size(0) % (kCacheCapacity / kPageSize) == 0,
                "combined_hbm_cache page count must be divisible by 16");
    TORCH_CHECK(attention_out.sizes() == query.sizes() && attention_out.scalar_type() == query.scalar_type(),
                "attention_out must match query shape and dtype");
    TORCH_CHECK(dram_kv_dev_ptr > 0, "dram_kv_dev_ptr must be a registered host pointer");
    TORCH_CHECK(hbm_block_table.dim() == 2 && hbm_block_table.size(0) == query.size(0) &&
                    hbm_block_table.size(1) == kCacheCapacity / kPageSize,
                "hbm_block_table must be [B, 16]");
    TORCH_CHECK(
        dram_block_table.dim() == 2 && dram_block_table.size(0) == query.size(0) && dram_block_table.size(1) > 0,
        "dram_block_table must be [B, dram_pages_per_request]");
    TORCH_CHECK(topk_src_ids.sizes() == topk_dst_slots.sizes() && topk_src_ids.dim() == 3 &&
                    topk_src_ids.size(0) == query.size(0) && topk_src_ids.size(1) == 1 &&
                    topk_src_ids.size(2) == kCacheCapacity,
                "top-k source and destination arrays must be [B, 1, 2048]");
    TORCH_CHECK(actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_query.numel() == query.size(0),
                "actual_seq_lengths_query must be cumulative Q1 lengths [B]");
    TORCH_CHECK(actual_seq_lengths_kv.dim() == 1 && actual_seq_lengths_kv.numel() == query.size(0),
                "actual_seq_lengths_kv must contain one 2048 length per request [B]");
    TORCH_CHECK(num_cache_tokens.dim() == 1 && num_cache_tokens.numel() == query.size(0),
                "num_cache_tokens must be [B]");
    TORCH_CHECK(miss_src_ids.sizes() == miss_dst_slots.sizes() && miss_src_ids.dim() == 2 &&
                    miss_src_ids.size(0) == query.size(0) && miss_src_ids.size(1) == kCacheCapacity,
                "request miss arrays must be [B, 2048]");
    TORCH_CHECK(topk_miss_counts.numel() == query.size(0) && miss_counts.numel() == query.size(0) &&
                    num_cache_tokens.numel() == query.size(0),
                "per-request count arrays must have B elements");

    const at::Tensor *intTensors[] = {
        &actual_seq_lengths_query, &actual_seq_lengths_kv, &num_cache_tokens, &topk_src_ids, &topk_dst_slots,
        &topk_miss_counts,         &miss_src_ids,          &miss_dst_slots,   &miss_counts,  &hbm_block_table,
        &dram_block_table};
    const char *intNames[] = {
        "actual_seq_lengths_query", "actual_seq_lengths_kv", "num_cache_tokens", "topk_src_ids", "topk_dst_slots",
        "topk_miss_counts",         "miss_src_ids",          "miss_dst_slots",   "miss_counts",  "hbm_block_table",
        "dram_block_table"};
    for (size_t i = 0; i < std::size(intTensors); ++i) {
        CheckInt32(*intTensors[i], intNames[i]);
        CheckNpuTensor(*intTensors[i], query, intNames[i]);
    }
    CheckNpuTensor(query_rope, query, "query_rope");
    CheckNpuTensor(query, query, "query");
    CheckNpuTensor(combined_hbm_cache, query, "combined_hbm_cache");
    CheckNpuTensor(attention_out, query, "attention_out");
    TORCH_CHECK(query_rope.scalar_type() == query.scalar_type(), "query_rope dtype must match query");

    const int64_t hbmBlockCount = combined_hbm_cache.size(0);
    const int64_t rowStride = kTokenStride;
    auto cacheKey = at::as_strided(combined_hbm_cache, {hbmBlockCount, kPageSize, 1, kKvDim},
                                   {kPageSize * rowStride, rowStride, rowStride, 1}, 0);
    auto cacheRope = at::as_strided(combined_hbm_cache, {hbmBlockCount, kPageSize, 1, kRopeDim},
                                    {kPageSize * rowStride, rowStride, rowStride, 1}, kKvDim);

    const auto &attentionOut = attention_out;
    auto softmaxMax = at::empty({0}, query.options().dtype(at::kFloat));
    auto softmaxSum = at::empty({0}, query.options().dtype(at::kFloat));
    SFAHost::SparseFlashAttention op("sparse_flash_attention");
    op.SetAttrAny("scale_value", static_cast<float>(scale_value));
    op.SetAttrAny("sparse_block_size", int64_t{1});
    op.SetAttrStr("layout_query", "TND");
    op.SetAttrStr("layout_kv", "PA_BSND");
    op.SetAttrAny("sparse_mode", int64_t{3});
    op.SetAttrAny("pre_tokens", INT64_MAX);
    op.SetAttrAny("next_tokens", INT64_MAX);
    op.SetAttrAny("attention_mode", int64_t{2});
    op.SetAttrAny("return_softmax_lse", false);
    auto context = std::make_shared<ge_helper::TilingContext>("sparse_flash_attention");
    auto queryDtype = query.scalar_type();
    op.SetToContext(context, queryDtype);
    context->RegisterTensor(query, true);
    context->RegisterTensor(cacheKey, true);
    context->RegisterTensor(cacheKey, true);
    context->RegisterTensor(topk_dst_slots, true);
    context->RegisterTensor(hbm_block_table, true);
    context->RegisterTensor(actual_seq_lengths_query, true);
    context->RegisterTensor(actual_seq_lengths_kv, true);
    context->RegisterTensor(query_rope, true);
    context->RegisterTensor(cacheRope, true);
    context->RegisterTensor(attentionOut, false);
    context->RegisterTensor(softmaxMax, false);
    context->RegisterTensor(softmaxSum, false);

    optiling::SFATilingInfo info;
    optiling::SFAInfoParser parser(context.get());
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS, "failed to parse sparse attention tiling inputs");
    optiling::SFATilingCheck checker(info);
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS, "sparse attention tiling validation failed");
    optiling::SFAMlaTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, "sparse attention tiling failed");

    FusedCopySfaTilingData fusedTilingData{};
    auto &sfaData = tiling.GetTilingData();
    TORCH_CHECK(sfaData.GetDataSize() == sizeof(SparseFlashAttentionTilingDataMla),
                "unexpected SFA tiling prefix size");
    sfaData.SaveToBuffer(reinterpret_cast<uint8_t *>(&fusedTilingData), sizeof(SparseFlashAttentionTilingDataMla));
    fusedTilingData.copyCap = kCacheCapacity;
    fusedTilingData.missCap = kCacheCapacity;
    fusedTilingData.hbmMaxBlockNum = static_cast<uint32_t>(hbm_block_table.size(1));
    fusedTilingData.dramMaxBlockNum = static_cast<uint32_t>(dram_block_table.size(1));
    std::array<uint8_t, sizeof(FusedCopySfaTilingData)> tilingBytes{};
    std::memcpy(tilingBytes.data(), &fusedTilingData, sizeof(fusedTilingData));
    auto tilingTensor = context->GetTilingTensor(tilingBytes);
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, query.options().dtype(at::kByte));
    auto *dramKv = reinterpret_cast<void *>(static_cast<uintptr_t>(dram_kv_dev_ptr));
    auto *dramRope = reinterpret_cast<void *>(static_cast<uintptr_t>(dram_kv_dev_ptr + kKvDim * 2));
    auto *hbmRope = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(combined_hbm_cache.data_ptr()) + kKvDim * 2);

    LaunchCombined(tiling.GetBlockDim(), query, cacheKey, cacheKey, topk_dst_slots, num_cache_tokens, hbm_block_table,
                   actual_seq_lengths_query, actual_seq_lengths_kv, query_rope, hbmRope, dramRope, dramKv,
                   dram_block_table, topk_src_ids, topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts,
                   attention_out, workspace, tilingTensor);
}

}  // namespace sglang::npu_kernel
