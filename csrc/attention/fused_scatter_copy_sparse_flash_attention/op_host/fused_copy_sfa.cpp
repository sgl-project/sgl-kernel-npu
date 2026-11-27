// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#include "fused_copy_sfa.h"
#include <c10/core/DeviceGuard.h>
#include <iterator>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kTopK = 2048;
constexpr int64_t kPageSize = 128;
constexpr int64_t kMaxQueriesPerRequest = 16;
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

}  // namespace

static void
FusedCopySfaImpl(const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &hbm_ckv, at::Tensor &hbm_rope,
                 const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_kv,
                 const at::Tensor &num_cache_tokens, const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
                 const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids, const at::Tensor &miss_dst_slots,
                 const at::Tensor &miss_counts, const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table,
                 int64_t dram_kv_dev_ptr, at::Tensor &attention_out, double scale_value, bool full_visible,
                 bool resident_refresh_tail, const c10::optional<at::Tensor> &current,
                 const c10::optional<at::Tensor> &positions, const c10::optional<at::Tensor> &valid,
                 const c10::optional<at::Tensor> &request_slots, const c10::optional<at::Tensor> &req_to_token,
                 const c10::optional<at::Tensor> &resident_end, int64_t host_token_capacity, int64_t dram_rope_dev_ptr)
{
    const bool resident = current.has_value();
    TORCH_CHECK(resident == positions.has_value() && resident == valid.has_value() &&
                    resident == request_slots.has_value() && resident == req_to_token.has_value() &&
                    resident == resident_end.has_value(),
                "resident staging inputs must be supplied together");
    TORCH_CHECK(!resident_refresh_tail || full_visible,
                "resident_refresh_tail requires full_visible=True and a native LIM suffix plan");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query must be float16 or bfloat16");
    TORCH_CHECK(query.device().type() == at::DeviceType::PrivateUse1, "query must be an NPU tensor");
    TORCH_CHECK(
        query.dim() == 3 && query.size(0) > 0 && query.size(1) > 0 && query.size(2) == kKvDim && query.is_contiguous(),
        "query must be contiguous TND [T, H, 512]");
    TORCH_CHECK(query.size(1) <= 64 || query.size(1) == 128, "Copy-SFA supports 1..64 or 128 query heads");
    c10::DeviceGuard guard(query.device());
    TORCH_CHECK(num_cache_tokens.dim() == 1 && num_cache_tokens.numel() > 0, "num_cache_tokens must be [B] with B > 0");
    const int64_t batchSize = num_cache_tokens.numel();
    const int64_t totalQueryTokens = query.size(0);
    if (resident) {
        TORCH_CHECK(full_visible, "resident staging requires full_visible");
        TORCH_CHECK(current->dim() == 3 && current->size(0) >= totalQueryTokens && current->size(1) == 1 &&
                        current->size(2) == kTokenStride && current->scalar_type() == query.scalar_type(),
                    "current must match query dtype and have shape [T>=query T,1,576]");
        TORCH_CHECK(positions->dim() == 1 && positions->numel() >= totalQueryTokens &&
                        positions->scalar_type() == at::kLong && valid->dim() == 1 &&
                        valid->numel() >= totalQueryTokens && valid->scalar_type() == at::kBool,
                    "positions and valid must be int64 and bool [T>=query T]");
        TORCH_CHECK(request_slots->dim() == 1 && request_slots->numel() == batchSize &&
                        request_slots->scalar_type() == at::kInt,
                    "request_slots must be int32 [B]");
        TORCH_CHECK(req_to_token->dim() == 2 && req_to_token->size(0) > 0 && req_to_token->size(1) > 0 &&
                        req_to_token->scalar_type() == at::kInt,
                    "req_to_token must be int32 [rows,context]");
        TORCH_CHECK(resident_end->dim() == 1 && resident_end->numel() == req_to_token->size(0) &&
                        resident_end->scalar_type() == at::kInt && host_token_capacity > 0 &&
                        host_token_capacity <= UINT32_MAX,
                    "invalid resident extent or host capacity");
        for (const auto *tensor : {&*current, &*positions, &*valid, &*request_slots, &*req_to_token, &*resident_end}) {
            CheckNpuTensor(*tensor, query, "resident input");
            tensor->record_stream(c10_npu::getCurrentNPUStream());
        }
    }
    TORCH_CHECK(totalQueryTokens >= batchSize && totalQueryTokens <= batchSize * kMaxQueriesPerRequest,
                "query T must satisfy B <= T <= 16 * B");
    TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == totalQueryTokens &&
                    query_rope.size(1) == query.size(1) && query_rope.size(2) == kRopeDim,
                "query_rope must be [T, H, 64]");
    TORCH_CHECK(hbm_ckv.scalar_type() == query.scalar_type() && hbm_ckv.dim() == 4 && hbm_ckv.size(1) == kPageSize &&
                    hbm_ckv.size(2) == 1 && hbm_ckv.size(3) == kKvDim && hbm_ckv.is_contiguous(),
                "hbm_ckv must be [pages, 128, 1, 512] and match query dtype");
    TORCH_CHECK(dram_rope_dev_ptr > 0, "split requires split registered host pointers");
    TORCH_CHECK(hbm_rope.scalar_type() == query.scalar_type() && hbm_rope.dim() == 4 &&
                    hbm_rope.size(0) == hbm_ckv.size(0) && hbm_rope.size(1) == kPageSize && hbm_rope.size(2) == 1 &&
                    hbm_rope.size(3) == kRopeDim,
                "hbm_rope must be [pages, 128, 1, 64] and match query dtype");
    CheckNpuTensor(hbm_rope, query, "hbm_rope");
    TORCH_CHECK(hbm_ckv.size(0) > 0, "hbm_ckv must contain at least one page");
    TORCH_CHECK(attention_out.sizes() == query.sizes() && attention_out.scalar_type() == query.scalar_type(),
                "attention_out must match query shape and dtype");
    TORCH_CHECK(dram_kv_dev_ptr > 0, "dram_kv_dev_ptr must be a registered host pointer");
    TORCH_CHECK(hbm_block_table.dim() == 2 && hbm_block_table.size(0) == batchSize && hbm_block_table.size(1) > 0,
                "hbm_block_table must be [B, hbm_pages_per_request]");
    TORCH_CHECK(dram_block_table.dim() == 2 && dram_block_table.size(0) == batchSize && dram_block_table.size(1) > 0,
                "dram_block_table must be [B, dram_pages_per_request]");
    TORCH_CHECK(topk_src_ids.sizes() == topk_dst_slots.sizes() && topk_src_ids.dim() == 2 &&
                    topk_src_ids.size(0) == totalQueryTokens && topk_src_ids.size(1) == kTopK,
                "top-k source and destination arrays must be [T, 2048]");
    TORCH_CHECK(actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_query.numel() == batchSize,
                "actual_seq_lengths_query must be cumulative query lengths [B]");
    TORCH_CHECK(actual_seq_lengths_kv.dim() == 1 && actual_seq_lengths_kv.numel() == batchSize,
                "actual_seq_lengths_kv must contain one request-local KV length per request [B]");
    TORCH_CHECK(miss_src_ids.sizes() == miss_dst_slots.sizes() && miss_src_ids.dim() == 2 &&
                    miss_src_ids.size(0) == batchSize && miss_src_ids.size(1) > 0,
                "request miss arrays must be [B, M] with M > 0");
    TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.numel() == totalQueryTokens && miss_counts.dim() == 1 &&
                    miss_counts.numel() == batchSize,
                "topk_miss_counts must be [T] and miss_counts must be [B]");

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
    CheckNpuTensor(hbm_ckv, query, "hbm_ckv");
    CheckNpuTensor(attention_out, query, "attention_out");
    TORCH_CHECK(query_rope.scalar_type() == query.scalar_type(), "query_rope dtype must match query");

    LaunchFusedCopySfa(query, query_rope, hbm_ckv, hbm_rope, actual_seq_lengths_query, actual_seq_lengths_kv,
                       num_cache_tokens, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
                       miss_counts, hbm_block_table, dram_block_table, dram_kv_dev_ptr, attention_out, scale_value,
                       full_visible, resident_refresh_tail, current, positions, valid, request_slots, req_to_token,
                       resident_end, host_token_capacity, dram_rope_dev_ptr);
}

void fused_scatter_copy_sparse_flash_attention_split(
    const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &hbm_ckv, at::Tensor &hbm_rope,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_kv,
    const at::Tensor &num_cache_tokens, const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids, const at::Tensor &miss_dst_slots,
    const at::Tensor &miss_counts, const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table,
    int64_t dram_kv_dev_ptr, at::Tensor &attention_out, double scale_value, bool full_visible,
    bool resident_refresh_tail, const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
    const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
    const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
    int64_t host_token_capacity, int64_t dram_rope_dev_ptr)
{
    FusedCopySfaImpl(query, query_rope, hbm_ckv, hbm_rope, actual_seq_lengths_query, actual_seq_lengths_kv,
                     num_cache_tokens, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
                     miss_counts, hbm_block_table, dram_block_table, dram_kv_dev_ptr, attention_out, scale_value,
                     full_visible, resident_refresh_tail, current, positions, valid, request_slots, req_to_token,
                     resident_end, host_token_capacity, dram_rope_dev_ptr);
}

}  // namespace sglang::npu_kernel
