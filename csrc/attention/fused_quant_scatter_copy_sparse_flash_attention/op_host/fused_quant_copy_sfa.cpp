// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. Host contract adapted from vllm-ascend PR #18490.
#include <array>
#include <cmath>
#include <optional>
#include "defines.h"
#include "batch_kernel_launch.h"
#include "ge_helper.h"
#include "owned_kernel_launch.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "../op_kernel/fused_quant_copy_sfa_tiling_data.h"
#include "../op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"
#include "aclrtlaunch_fused_quant_copy_sfa_bf16.h"
#include "aclrtlaunch_fused_quant_copy_sfa_bf16_split.h"
#include "aclrtlaunch_fused_quant_copy_sfa_fp16.h"
#include "aclrtlaunch_fused_quant_copy_sfa_fp16_split.h"
#include "aclrtlaunch_fused_quant_copy_sfa_v2_bf16.h"
#include "aclrtlaunch_fused_quant_copy_sfa_v2_bf16_split.h"
#include "aclrtlaunch_fused_quant_copy_sfa_v2_fp16.h"
#include "aclrtlaunch_fused_quant_copy_sfa_v2_fp16_split.h"

// Only this operator's generated host stub is redirected here. Other native
// kernels retain their existing launch policy. Match the original OPP's
// SetScheduleMode(1), including captured graphs and concurrent streams.
extern "C" uint32_t LaunchCopySfaBatchKernel(void *binary, uint64_t entry, uint32_t blocks, void **args,
                                             uint32_t args_size, const void *stream)
{
    return sglang::npu_kernel::LaunchKernelWithBatchScheduling(binary, entry, blocks, args, args_size, stream);
}

namespace sglang::npu_kernel {
inline bool is_fused_quant_sfa_head_count_supported(int64_t heads)
{
    return heads >= fused_quant_scatter_copy_sfa::MIN_QUERY_HEAD_COUNT &&
           heads <= fused_quant_scatter_copy_sfa::MAX_QUERY_HEAD_COUNT &&
           fused_quant_scatter_copy_sfa::IsSupportedHeadCount(static_cast<uint32_t>(heads));
}

HOST_API void fused_quant_scatter_copy_sparse_flash_attention(
    const at::Tensor &query, at::Tensor hbm_kv_bytes, const at::Tensor &dram_kv_bytes, int64_t dram_kv_dev_ptr,
    const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_src_ids, const at::Tensor &topk_miss_counts, const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts, double scale_value, at::Tensor attention_out)
{
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == fused_quant_scatter_copy_sfa::QUERY_DIM &&
                    is_fused_quant_sfa_head_count_supported(query.size(1)),
                "query must be [T, H, 576], H in {8, 16, 32, 64, 128}.");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
                "query must be bfloat16 or float16.");
    TORCH_CHECK((query.device().is_privateuseone() || query.is_meta()),
                "fused quant scatter-copy SFA tensors must be on NPU.");

    const int64_t batch_size = num_cache_tokens.numel();
    const int64_t total_queries = query.size(0);
    TORCH_CHECK(num_cache_tokens.dim() == 1 && batch_size > 0 && total_queries >= batch_size &&
                    total_queries <= fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT * batch_size,
                "num_cache_tokens must be [B] and B <= T <= 16B (MTP0..MTP15).");
    for (const auto *cache : std::array<const at::Tensor *, 2>{&hbm_kv_bytes, &dram_kv_bytes}) {
        TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 &&
                        cache->size(1) == fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE &&
                        cache->size(2) == fused_quant_scatter_copy_sfa::KV_HEAD_COUNT &&
                        cache->size(3) == fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES &&
                        cache->scalar_type() == at::kChar,
                    "KV caches must be int8 [blocks, 128, 1, 656] C8 byte views.");
    }
    for (const auto *table : {&hbm_block_table, &dram_block_table}) {
        TORCH_CHECK(table->dim() == 2 && table->size(0) == batch_size && table->size(1) > 0 &&
                        table->size(1) <= fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST,
                    "block tables must be [B, 1..2048].");
    }
    for (const auto *ids : {&topk_dst_slots, &topk_src_ids}) {
        TORCH_CHECK(ids->dim() == 3 && ids->size(0) == total_queries && ids->size(1) == 1 &&
                        ids->size(2) == fused_quant_scatter_copy_sfa::TOPK_CAPACITY,
                    "topk indices must be [T, 1, 2048].");
    }
    for (const auto *ids : {&miss_src_ids, &miss_dst_slots}) {
        TORCH_CHECK(ids->dim() == 3 && ids->size(0) == batch_size && ids->size(1) == 1 &&
                        ids->size(2) == fused_quant_scatter_copy_sfa::COPY_CAPACITY,
                    "miss indices must be [B, 1, 32768].");
    }
    TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries &&
                    actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_query.size(0) == batch_size &&
                    actual_seq_lengths_kv.dim() == 1 && actual_seq_lengths_kv.size(0) == batch_size &&
                    miss_counts.dim() == 1 && miss_counts.size(0) == batch_size,
                "topk_miss_counts must be [T], and query ends, actual KV "
                "lengths, num_cache_tokens and miss_counts must be [B].");

    for (const auto *metadata :
         std::array<const at::Tensor *, 11>{&hbm_block_table, &dram_block_table, &topk_dst_slots, &topk_src_ids,
                                            &topk_miss_counts, &actual_seq_lengths_query, &actual_seq_lengths_kv,
                                            &num_cache_tokens, &miss_src_ids, &miss_dst_slots, &miss_counts}) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt, "all metadata tensors must be int32.");
    }
    TORCH_CHECK(attention_out.dim() == 3 && attention_out.size(0) == total_queries &&
                    attention_out.size(1) == query.size(1) &&
                    attention_out.size(2) == fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM &&
                    attention_out.scalar_type() == query.scalar_type(),
                "attention_out must be [T, H, 512] with the query dtype.");
    TORCH_CHECK(std::isfinite(static_cast<float>(scale_value)), "scale_value must be finite in FP32.");

    const auto device = query.device();
    for (const auto *tensor : std::array<const at::Tensor *, 15>{
             &query, &hbm_kv_bytes, &dram_kv_bytes, &hbm_block_table, &dram_block_table, &topk_dst_slots, &topk_src_ids,
             &topk_miss_counts, &actual_seq_lengths_query, &actual_seq_lengths_kv, &num_cache_tokens, &miss_src_ids,
             &miss_dst_slots, &miss_counts, &attention_out}) {
        TORCH_CHECK((tensor == &dram_kv_bytes ? (tensor->device().is_cpu() || tensor->device() == device)
                                              : tensor->device() == device) &&
                        tensor->is_contiguous(),
                    "all tensors must be contiguous on the same NPU.");
    }

    if (query.is_meta()) return;
    TORCH_CHECK(dram_kv_dev_ptr > 0, "DRAM must have a registered device-visible address");
    c10_npu::OptionalNPUGuard guard(query.device());
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND950,
                "quant Copy-SFA requires Ascend950 (A5)");
    const uint32_t cores = platform->GetCoreNumAic();
    TORCH_CHECK(cores > 0 && cores % 2U == 0 && platform->GetCoreNumAiv() >= cores * 2U,
                "quant Copy-SFA requires paired mix cores");
    TORCH_CHECK(hbm_kv_bytes.size(0) <= UINT32_MAX && dram_kv_bytes.size(0) <= UINT32_MAX,
                "quant Copy-SFA physical block count exceeds uint32");
    FusedQuantScatterCopySparseFlashAttentionTilingData t{};
    t.baseParams.batchSize = num_cache_tokens.numel();
    t.baseParams.seqSize = hbm_block_table.size(1) * 128U;
    t.baseParams.qSeqSize = query.size(0);
    t.baseParams.nNumOfQInOneGroup = query.size(1);
    t.baseParams.sparseBlockCount = 2048;
    t.baseParams.sparseMode = 3;
    t.baseParams.outputLayout = 1;
    t.baseParams.blockSize = 128;
    t.baseParams.maxBlockNumPerBatch = hbm_block_table.size(1);
    t.baseParams.dSizeVInput = 656;
    t.baseParams.scaleValue = scale_value;
    t.singleCoreParams.usedCoreNum = cores;
    t.dramMaxBlocks = dram_block_table.size(1);
    t.hbmPhysicalBlocks = hbm_kv_bytes.size(0);
    t.dramPhysicalBlocks = dram_kv_bytes.size(0);
    auto tiling =
        ge_helper::TilingTensorCache<FusedQuantScatterCopySparseFlashAttentionTilingData>::Get(t, "quant_copy_sfa");
    const uint64_t bytes =
        platform->GetLibApiWorkSpaceSize() + fused_quant_scatter_copy_sfa::UserWorkspaceBytes(cores, query.size(0));
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    void *dram = reinterpret_cast<void *>(static_cast<uintptr_t>(dram_kv_dev_ptr));
    const std::vector<at::Tensor> owners = {query,
                                            hbm_kv_bytes,
                                            dram_kv_bytes,
                                            hbm_block_table,
                                            dram_block_table,
                                            topk_dst_slots,
                                            topk_src_ids,
                                            topk_miss_counts,
                                            actual_seq_lengths_query,
                                            actual_seq_lengths_kv,
                                            num_cache_tokens,
                                            miss_src_ids,
                                            miss_dst_slots,
                                            miss_counts,
                                            attention_out,
                                            workspace,
                                            tiling};
    if (query.scalar_type() == at::kBFloat16) {
        if (query.size(1) == 128) {
            EXEC_OWNED_KERNEL_CMD(fused_quant_copy_sfa_bf16_split, cores, owners, query, hbm_kv_bytes, dram,
                                  hbm_block_table, dram_block_table, topk_dst_slots, topk_src_ids, topk_miss_counts,
                                  actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens, miss_src_ids,
                                  miss_dst_slots, miss_counts, hbm_kv_bytes, attention_out, workspace, tiling);
        } else {
            EXEC_OWNED_KERNEL_CMD(fused_quant_copy_sfa_bf16, cores, owners, query, hbm_kv_bytes, dram, hbm_block_table,
                                  dram_block_table, topk_dst_slots, topk_src_ids, topk_miss_counts,
                                  actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens, miss_src_ids,
                                  miss_dst_slots, miss_counts, hbm_kv_bytes, attention_out, workspace, tiling);
        }
    } else {
        if (query.size(1) == 128) {
            EXEC_OWNED_KERNEL_CMD(fused_quant_copy_sfa_fp16_split, cores, owners, query, hbm_kv_bytes, dram,
                                  hbm_block_table, dram_block_table, topk_dst_slots, topk_src_ids, topk_miss_counts,
                                  actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens, miss_src_ids,
                                  miss_dst_slots, miss_counts, hbm_kv_bytes, attention_out, workspace, tiling);
        } else {
            EXEC_OWNED_KERNEL_CMD(fused_quant_copy_sfa_fp16, cores, owners, query, hbm_kv_bytes, dram, hbm_block_table,
                                  dram_block_table, topk_dst_slots, topk_src_ids, topk_miss_counts,
                                  actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens, miss_src_ids,
                                  miss_dst_slots, miss_counts, hbm_kv_bytes, attention_out, workspace, tiling);
        }
    }
}

HOST_API void fused_quant_scatter_copy_sparse_flash_attention_v2(
    const at::Tensor &query_nope, const at::Tensor &query_rope, at::Tensor hbm_kv_bytes,
    const at::Tensor &dram_kv_bytes, int64_t dram_kv_dev_ptr, const at::Tensor &hbm_block_table,
    const at::Tensor &dram_block_table, const at::Tensor &topk_dst_slots, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_miss_counts, const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts, double scale_value, at::Tensor attention_out,
    int64_t cache_capacity, const std::optional<at::Tensor> &current, const std::optional<at::Tensor> &positions,
    const std::optional<at::Tensor> &current_valid, const std::optional<at::Tensor> &request_slots,
    const std::optional<at::Tensor> &starts, const std::optional<at::Tensor> &ends)
{
    TORCH_CHECK(query_nope.dim() == 3 && query_nope.size(0) > 0 &&
                    query_nope.size(2) == fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM &&
                    is_fused_quant_sfa_head_count_supported(query_nope.size(1)),
                "query_nope must be [T, H, 512], H in {8, 16, 32, 64, 128}.");
    TORCH_CHECK(query_rope.dim() == 3 && query_rope.size(0) == query_nope.size(0) &&
                    query_rope.size(1) == query_nope.size(1) &&
                    query_rope.size(2) == fused_quant_scatter_copy_sfa::ROPE_DIM,
                "query_rope must be [T, H, 64].");
    TORCH_CHECK((query_nope.scalar_type() == at::kBFloat16 || query_nope.scalar_type() == at::kHalf) &&
                    query_rope.scalar_type() == query_nope.scalar_type(),
                "split queries must share the BF16 or FP16 dtype.");
    TORCH_CHECK(query_nope.device().is_privateuseone() || query_nope.is_meta(), "C8 v2 tensors must be on NPU.");
    const int64_t total_queries = query_nope.size(0);
    const int64_t batch_size = num_cache_tokens.numel();
    TORCH_CHECK(num_cache_tokens.dim() == 1 && batch_size > 0 && total_queries >= batch_size &&
                    total_queries <= fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT * batch_size,
                "num_cache_tokens must be [B] and B <= T <= 16B.");
    TORCH_CHECK(cache_capacity >= 2048 && cache_capacity <= 32384 &&
                    cache_capacity % fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE == 0,
                "cache_capacity must be 128-aligned in [2048, 32384].");
    for (const auto *cache : std::array<const at::Tensor *, 2>{&hbm_kv_bytes, &dram_kv_bytes}) {
        TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 &&
                        cache->size(1) == fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE &&
                        cache->size(2) == fused_quant_scatter_copy_sfa::KV_HEAD_COUNT &&
                        cache->size(3) == fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES &&
                        cache->scalar_type() == at::kChar,
                    "KV caches must be int8 [blocks, 128, 1, 656].");
    }
    TORCH_CHECK(hbm_block_table.dim() == 2 && hbm_block_table.size(0) == batch_size &&
                    hbm_block_table.size(1) >= cache_capacity / 128 + 2 &&
                    hbm_block_table.size(1) <= fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST &&
                    dram_block_table.dim() == 2 && dram_block_table.size(0) == batch_size &&
                    dram_block_table.size(1) > 0 &&
                    dram_block_table.size(1) <= fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST,
                "v2 block tables must be [B, pages] and HBM must include the two-page ring.");
    for (const auto *ids : {&topk_dst_slots, &topk_src_ids}) {
        TORCH_CHECK(ids->sizes() == at::IntArrayRef({total_queries, 1, fused_quant_scatter_copy_sfa::TOPK_CAPACITY}),
                    "topk indices must be [T, 1, 2048].");
    }
    for (const auto *ids : {&miss_src_ids, &miss_dst_slots}) {
        TORCH_CHECK(ids->sizes() == at::IntArrayRef({batch_size, 1, fused_quant_scatter_copy_sfa::COPY_CAPACITY}),
                    "miss indices must be [B, 1, 32768].");
    }
    TORCH_CHECK(topk_miss_counts.sizes() == at::IntArrayRef({total_queries}) &&
                    actual_seq_lengths_query.sizes() == at::IntArrayRef({batch_size}) &&
                    actual_seq_lengths_kv.sizes() == at::IntArrayRef({batch_size}) &&
                    miss_counts.sizes() == at::IntArrayRef({batch_size}),
                "v2 count and sequence metadata shapes do not match T/B.");
    for (const auto *metadata :
         std::array<const at::Tensor *, 11>{&hbm_block_table, &dram_block_table, &topk_dst_slots, &topk_src_ids,
                                            &topk_miss_counts, &actual_seq_lengths_query, &actual_seq_lengths_kv,
                                            &num_cache_tokens, &miss_src_ids, &miss_dst_slots, &miss_counts}) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt, "all v2 metadata tensors must be int32.");
    }
    TORCH_CHECK(attention_out.sizes() == at::IntArrayRef({total_queries, query_nope.size(1),
                                                          fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM}) &&
                    attention_out.scalar_type() == query_nope.scalar_type(),
                "attention_out must be [T, H, 512] with the query dtype.");
    TORCH_CHECK(std::isfinite(static_cast<float>(scale_value)), "scale_value must be finite in FP32.");

    const bool has_ring = current.has_value();
    TORCH_CHECK(positions.has_value() == has_ring && current_valid.has_value() == has_ring &&
                    request_slots.has_value() == has_ring && starts.has_value() == has_ring &&
                    ends.has_value() == has_ring,
                "current, positions, valid, request_slots, starts and ends must be all present or all absent.");
    if (has_ring) {
        TORCH_CHECK(current->scalar_type() == at::kChar &&
                        current->numel() == total_queries * fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES,
                    "current must contain exactly T opaque 656-byte rows.");
        TORCH_CHECK(positions->scalar_type() == at::kLong && positions->numel() == total_queries,
                    "positions must be int64[T].");
        TORCH_CHECK(current_valid->scalar_type() == at::kBool && current_valid->numel() == total_queries,
                    "current_valid must be bool[T].");
        TORCH_CHECK(request_slots->scalar_type() == at::kInt && request_slots->sizes() == at::IntArrayRef({batch_size}),
                    "request_slots must be int32[B].");
        TORCH_CHECK(starts->scalar_type() == at::kInt && ends->scalar_type() == at::kInt && starts->dim() == 1 &&
                        ends->sizes() == starts->sizes() && starts->numel() > 1 && starts->numel() <= UINT32_MAX,
                    "starts and ends must be equal one-dimensional int32 state arrays.");
    }

    const auto device = query_nope.device();
    std::vector<const at::Tensor *> tensors{&query_nope,
                                            &query_rope,
                                            &hbm_kv_bytes,
                                            &dram_kv_bytes,
                                            &hbm_block_table,
                                            &dram_block_table,
                                            &topk_dst_slots,
                                            &topk_src_ids,
                                            &topk_miss_counts,
                                            &actual_seq_lengths_query,
                                            &actual_seq_lengths_kv,
                                            &num_cache_tokens,
                                            &miss_src_ids,
                                            &miss_dst_slots,
                                            &miss_counts,
                                            &attention_out};
    if (has_ring) {
        tensors.insert(tensors.end(), {&*current, &*positions, &*current_valid, &*request_slots, &*starts, &*ends});
    }
    for (const auto *tensor : tensors) {
        const bool device_ok = tensor == &dram_kv_bytes ? (tensor->device().is_cpu() || tensor->device() == device)
                                                        : tensor->device() == device;
        TORCH_CHECK(device_ok && tensor->is_contiguous(), "all v2 tensors must be contiguous on the expected device.");
    }
    if (query_nope.is_meta()) return;

    TORCH_CHECK(dram_kv_dev_ptr > 0, "DRAM must have a registered device-visible address");
    c10_npu::OptionalNPUGuard guard(device);
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND950,
                "quant Copy-SFA v2 requires Ascend950 (A5)");
    const uint32_t cores = platform->GetCoreNumAic();
    TORCH_CHECK(cores > 0 && cores % 2U == 0 && platform->GetCoreNumAiv() >= cores * 2U,
                "quant Copy-SFA v2 requires paired mix cores");
    TORCH_CHECK(hbm_kv_bytes.size(0) <= UINT32_MAX && dram_kv_bytes.size(0) <= UINT32_MAX,
                "quant Copy-SFA v2 physical block count exceeds uint32");
    FusedQuantScatterCopySparseFlashAttentionTilingData t{};
    t.baseParams.batchSize = static_cast<uint32_t>(batch_size);
    t.baseParams.seqSize = static_cast<uint32_t>(hbm_block_table.size(1)) * 128U;
    t.baseParams.qSeqSize = static_cast<uint32_t>(total_queries);
    t.baseParams.nNumOfQInOneGroup = static_cast<uint32_t>(query_nope.size(1));
    t.baseParams.sparseBlockCount = fused_quant_scatter_copy_sfa::TOPK_CAPACITY;
    t.baseParams.sparseMode = 3;
    t.baseParams.outputLayout = 1;
    t.baseParams.blockSize = fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
    t.baseParams.maxBlockNumPerBatch = static_cast<uint32_t>(hbm_block_table.size(1));
    t.baseParams.dSizeVInput = fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES;
    t.baseParams.scaleValue = static_cast<float>(scale_value);
    t.singleCoreParams.usedCoreNum = cores;
    t.dramMaxBlocks = static_cast<uint32_t>(dram_block_table.size(1));
    t.hbmPhysicalBlocks = static_cast<uint32_t>(hbm_kv_bytes.size(0));
    t.dramPhysicalBlocks = static_cast<uint32_t>(dram_kv_bytes.size(0));
    t.reserved[0] = has_ring ? static_cast<uint32_t>(starts->numel()) : 0U;
    auto tiling =
        ge_helper::TilingTensorCache<FusedQuantScatterCopySparseFlashAttentionTilingData>::Get(t, "quant_copy_sfa_v2");
    const uint64_t bytes =
        platform->GetLibApiWorkSpaceSize() +
        fused_quant_scatter_copy_sfa::RingMetadataWorkspaceBytes(static_cast<uint32_t>(batch_size)) +
        fused_quant_scatter_copy_sfa::UserWorkspaceBytes(cores, static_cast<uint32_t>(total_queries));
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query_nope.options().dtype(at::kByte));
    void *dram = reinterpret_cast<void *>(static_cast<uintptr_t>(dram_kv_dev_ptr));
    void *current_ptr = has_ring ? current->data_ptr() : nullptr;
    void *positions_ptr = has_ring ? positions->data_ptr() : nullptr;
    void *valid_ptr = has_ring ? current_valid->data_ptr() : nullptr;
    void *requests_ptr = has_ring ? request_slots->data_ptr() : nullptr;
    void *starts_ptr = has_ring ? starts->data_ptr() : nullptr;
    void *ends_ptr = has_ring ? ends->data_ptr() : nullptr;
    std::vector<at::Tensor> owners{query_nope,
                                   query_rope,
                                   hbm_kv_bytes,
                                   dram_kv_bytes,
                                   hbm_block_table,
                                   dram_block_table,
                                   topk_dst_slots,
                                   topk_src_ids,
                                   topk_miss_counts,
                                   actual_seq_lengths_query,
                                   actual_seq_lengths_kv,
                                   num_cache_tokens,
                                   miss_src_ids,
                                   miss_dst_slots,
                                   miss_counts,
                                   attention_out,
                                   workspace,
                                   tiling};
    if (has_ring) owners.insert(owners.end(), {*current, *positions, *current_valid, *request_slots, *starts, *ends});
    const uint32_t capacity = static_cast<uint32_t>(cache_capacity);
#define LAUNCH_COPY_SFA_V2(kernel)                                                                                    \
    EXEC_OWNED_KERNEL_CMD(kernel, cores, owners, query_nope, query_rope, hbm_kv_bytes, dram, hbm_block_table,         \
                          dram_block_table, topk_dst_slots, topk_src_ids, topk_miss_counts, actual_seq_lengths_query, \
                          actual_seq_lengths_kv, num_cache_tokens, miss_src_ids, miss_dst_slots, miss_counts,         \
                          hbm_kv_bytes, attention_out, capacity, current_ptr, positions_ptr, valid_ptr, requests_ptr, \
                          starts_ptr, ends_ptr, workspace, tiling)
    if (query_nope.scalar_type() == at::kBFloat16) {
        if (query_nope.size(1) == 128)
            LAUNCH_COPY_SFA_V2(fused_quant_copy_sfa_v2_bf16_split);
        else
            LAUNCH_COPY_SFA_V2(fused_quant_copy_sfa_v2_bf16);
    } else {
        if (query_nope.size(1) == 128)
            LAUNCH_COPY_SFA_V2(fused_quant_copy_sfa_v2_fp16_split);
        else
            LAUNCH_COPY_SFA_V2(fused_quant_copy_sfa_v2_fp16);
    }
#undef LAUNCH_COPY_SFA_V2
}
}  // namespace sglang::npu_kernel
