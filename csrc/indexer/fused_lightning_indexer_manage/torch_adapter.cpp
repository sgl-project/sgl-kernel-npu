#include "aclnn_runtime.h"
#include "aclnn_sgl_fused_lightning_indexer_manage.h"
#include "op_kernel/upstream/fused_lightning_indexer_manage_constants.h"
#include <array>
#include <limits>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
namespace {
void fused_lightning_indexer_manage(const at::Tensor &index_weights, const at::Tensor &query,
                                    const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
                                    const at::Tensor &actual_seq_lengths_query,
                                    const at::Tensor &actual_seq_lengths_key, const at::Tensor &offload_seq_lengths_key,
                                    const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
                                    const at::Tensor &req_pool_entries, at::Tensor cache_slots_pool,
                                    at::Tensor topk_src_ids, at::Tensor topk_dst_slots, at::Tensor topk_miss_counts,
                                    at::Tensor miss_src_ids, at::Tensor miss_dst_slots, at::Tensor miss_counts,
                                    bool full_visible)
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
    TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 && index_key_cache.size(1) == kBlockSize &&
                    index_key_cache.size(2) == LIMConfig::KEY_HEADS && index_key_cache.size(3) == LIMConfig::HEAD_DIM,
                "index_key_cache must be [blocks, 128, 1, 128].");
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
    TORCH_CHECK(topk_src_ids.dim() == 2 && topk_src_ids.size(0) == total_queries && topk_src_ids.size(1) == kTopK &&
                    topk_dst_slots.sizes() == topk_src_ids.sizes(),
                "topk outputs must be [T, 2048].");
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
                                      query,
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

    TORCH_CHECK(batch_size <= UINT32_MAX && total_queries <= UINT32_MAX && cache_slots_pool.size(0) <= UINT32_MAX,
                "LIM dimensions exceed uint32");
    const c10::DeviceGuard guard(query.device());
    auto stream = c10_npu::getCurrentNPUStream();
    std::array<at::Tensor, 18> tensors = {index_weights,
                                          query,
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
                                          miss_counts,
                                          cache_slots_pool};
    std::array<std::shared_ptr<aclTensor>, 18> d;
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensors[i].record_stream(stream);
        aclDataType dtype = i < 3 ? (query.scalar_type() == at::kHalf ? ACL_FLOAT16 : ACL_BF16) : ACL_INT32;
        d[i] = sgl_cann::tensor_descriptor(tensors[i], dtype);
    }
    static auto get = reinterpret_cast<decltype(&aclnnSglFusedLightningIndexerManageGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglFusedLightningIndexerManageGetWorkspaceSize"));
    static auto exec = reinterpret_cast<decltype(&aclnnSglFusedLightningIndexerManage)>(
        sgl_cann::op_api_symbol("aclnnSglFusedLightningIndexerManage"));
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    auto status = get(d[0].get(), d[1].get(), d[2].get(), d[3].get(), d[4].get(), d[5].get(), d[6].get(), d[7].get(),
                      d[8].get(), d[9].get(), d[10].get(), full_visible, d[11].get(), d[12].get(), d[13].get(),
                      d[14].get(), d[15].get(), d[16].get(), d[17].get(), &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "LIM workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()), "LIM workspace too large");
    at::Tensor workspace;
    if (bytes) workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    auto aclStream = stream.stream(false);
    auto launch = [tensors, d, workspace, bytes, executor, aclStream]() -> int {
        auto result = exec(bytes ? workspace.data_ptr() : nullptr, bytes, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "LIM execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglFusedLightningIndexerManage", launch);
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "fused_lightning_indexer_manage(Tensor index_weights, Tensor query, Tensor index_key_cache, Tensor "
        "index_block_table, Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, Tensor "
        "offload_seq_lengths_key, Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, Tensor(a!) "
        "cache_slots_pool, Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, Tensor(e!) "
        "miss_src_ids, Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts, bool full_visible=False) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_lightning_indexer_manage", &fused_lightning_indexer_manage);
}
