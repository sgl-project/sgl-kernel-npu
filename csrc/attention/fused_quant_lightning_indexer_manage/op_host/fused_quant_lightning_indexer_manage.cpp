// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include "batch_kernel_launch.h"
#include "ge_helper.h"
#include "owned_kernel_launch.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_tiling.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_workspace.h"
#include "../op_kernel/arch35/lim_payload_layout.h"
#include "../op_kernel/arch35/lim_simt_config.h"
#include "aclrtlaunch_fused_quant_lightning_indexer_manage_full_visible.h"

extern "C" uint32_t LaunchFullVisibleLimKernel(void *binary, uint64_t entry, uint32_t blocks, void **args,
                                               uint32_t args_size, const void *stream)
{
    return sglang::npu_kernel::LaunchKernelWithBatchScheduling(binary, entry, blocks, args, args_size, stream,
                                                               LIM_SIMT_USER_UB_BYTES);
}

namespace {

void fused_quant_lightning_indexer_manage_full_visible(
    const at::Tensor &query, const at::Tensor &weights, const at::Tensor &key, const at::Tensor &query_scale,
    const at::Tensor &key_scale, const at::Tensor &block_table, const at::Tensor &query_ends, const at::Tensor &lengths,
    const at::Tensor &cache_tokens, const at::Tensor &request_state, const at::Tensor &pool_entries,
    const at::Tensor &slot_pool, const at::Tensor &topk_sources, const at::Tensor &topk_slots,
    const at::Tensor &topk_miss_counts, const at::Tensor &miss_sources, const at::Tensor &miss_slots,
    const at::Tensor &miss_counts)
{
    using namespace lim_request_limits;
    constexpr int64_t head_dim = 128;
    constexpr auto max_u32 = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == head_dim &&
                    (query.size(1) == 32 || query.size(1) == 64),
                "A5 LIM query must be [T, H, 128], H=32 or 64.");
    TORCH_CHECK(query.device().is_privateuseone() || query.is_meta(), "A5 LIM requires NPU tensors.");
    const int64_t rows = query.size(0);
    const int64_t heads = query.size(1);
    const int64_t batch = query_ends.numel();
    TORCH_CHECK(batch > 0 && rows >= batch && rows <= batch * MAX_ROUTES && rows <= max_u32,
                "A5 LIM requires B <= T <= 14B.");
    TORCH_CHECK(weights.sizes() == at::IntArrayRef({rows, heads}) && query_scale.sizes() == weights.sizes(),
                "A5 LIM weights and query scale must be [T, H].");
    TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(1) == BLOCK_SIZE && key.size(2) == 1 &&
                    key.size(3) == head_dim && key_scale.sizes() == at::IntArrayRef({key.size(0), BLOCK_SIZE, 1}),
                "A5 LIM key and key scale must be [blocks, 128, 1, 128] and [blocks, 128, 1].");
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn && key.scalar_type() == query.scalar_type() &&
                    weights.scalar_type() == at::kBFloat16 && query_scale.scalar_type() == at::kFloat &&
                    key_scale.scalar_type() == at::kFloat,
                "A5 LIM requires FP8 query/key, BF16 weights and FP32 scales.");
    TORCH_CHECK(query.is_contiguous() && query_scale.is_contiguous() && weights.stride(1) == 1 &&
                    weights.stride(0) >= heads && weights.stride(0) <= max_u32 && key.stride(3) == 1 &&
                    key.stride(1) == head_dim && key.stride(0) >= BLOCK_SIZE * head_dim && key.stride(0) <= max_u32 &&
                    key_scale.stride(1) == 1 && key_scale.stride(0) >= BLOCK_SIZE && key_scale.stride(0) <= max_u32,
                "A5 LIM supports striding along the weight row and key/scale block axes.");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch && block_table.size(1) > 0 &&
                    block_table.size(1) <= lim_payload_layout::MAX_SOURCE_CAPACITY / BLOCK_SIZE,
                "A5 LIM block table must be non-empty [B, pages].");
    const int64_t source_capacity = block_table.size(1) * BLOCK_SIZE;
    TORCH_CHECK(slot_pool.dim() == 2 && slot_pool.size(0) >= 3 && slot_pool.size(0) <= max_u32 &&
                    slot_pool.size(1) == source_capacity,
                "A5 LIM source map needs a live row between row zero and the padding row, with width pages * 128.");
    for (const auto *tensor : {&query_ends, &lengths, &cache_tokens, &request_state, &pool_entries, &miss_counts}) {
        TORCH_CHECK(tensor->sizes() == at::IntArrayRef({batch}), "A5 LIM request vectors must be [B].");
    }
    TORCH_CHECK(topk_sources.sizes() == at::IntArrayRef({rows, TOPK}) && topk_slots.sizes() == topk_sources.sizes() &&
                    topk_miss_counts.sizes() == at::IntArrayRef({rows}),
                "A5 LIM TopK outputs must be [T, 2048], [T].");
    TORCH_CHECK(
        miss_sources.sizes() == at::IntArrayRef({batch, MISS_CAPACITY}) && miss_slots.sizes() == miss_sources.sizes(),
        "A5 LIM miss outputs must be [B, 32768].");

    const auto device = query.device();
    const std::vector<at::Tensor> metadata = {block_table,  query_ends, lengths,      cache_tokens, request_state,
                                              pool_entries, slot_pool,  topk_sources, topk_slots,   topk_miss_counts,
                                              miss_sources, miss_slots, miss_counts};
    for (const auto &tensor : metadata) {
        TORCH_CHECK(tensor.device() == device && tensor.scalar_type() == at::kInt && tensor.is_contiguous(),
                    "A5 LIM metadata and outputs must be contiguous int32 on the query device.");
    }
    for (const auto *tensor : {&weights, &key, &query_scale, &key_scale}) {
        TORCH_CHECK(tensor->device() == device, "A5 LIM inputs must share one device.");
    }
    if (query.is_meta()) return;

    c10_npu::OptionalNPUGuard guard(device);
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "A5 LIM: Ascend platform is unavailable.");
    const uint32_t cores = std::min<uint32_t>(platform->GetCoreNumAic(), platform->GetCoreNumAiv() / 2);
    TORCH_CHECK(cores > 0, "A5 LIM requires paired mix cores.");
    const uint64_t route_rows = std::max<uint64_t>(4ULL * batch, rows);
    const uint64_t segments = fused_quant_li_manage_workspace::CandidateSegmentCapacity(source_capacity, cores);
    const uint64_t qli_bytes = fused_quant_li_manage_workspace::TotalBytes(
        source_capacity, route_rows, segments, source_capacity > lim_payload_layout::PACKED_SOURCE_CAPACITY);
    TORCH_CHECK(route_rows <= max_u32 && qli_bytes <= max_u32, "A5 LIM workspace exceeds uint32.");

    FusedQuantLightningIndexerManageTilingData t{};
    t.usedCoreNum = cores;
    t.batchSize = batch;
    t.totalQueryRows = rows;
    t.poolSize = slot_pool.size(0);
    t.sourceCapacity = source_capacity;
    t.indexHeads = heads;
    t.maxBlockNumPerBatch = block_table.size(1);
    t.weightStride = weights.stride(0);
    t.keyStride = key.stride(0);
    t.scaleStride = key_scale.stride(0);
    t.managementWorkspaceOffset = qli_bytes;
    t.scoreWorkspaceStride = 4 * source_capacity * sizeof(uint16_t);
    t.scoreRowStride = source_capacity;
    t.queryTileSize = heads == 32 && (rows > 64 || batch * 2 > cores) ? 4 : 2;
    t.routeStorageRows = route_rows;
    const uint32_t blocks = platform->CalcTschBlockDim(cores * 2, cores, cores * 2);
    auto tiling = sglang::ge_helper::TilingTensorCache<FusedQuantLightningIndexerManageTilingData>::Get(t, "a5_lim");
    const uint64_t bytes = platform->GetLibApiWorkSpaceSize() + qli_bytes + (batch + rows) * 32;
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    auto owners = metadata;
    owners.insert(owners.end(), {query, weights, key, query_scale, key_scale, workspace, tiling});
    EXEC_OWNED_KERNEL_CMD(fused_quant_lightning_indexer_manage_full_visible, blocks, owners, weights, query_scale,
                          query, key_scale, key, block_table, query_ends, lengths, cache_tokens, request_state,
                          pool_entries, slot_pool, topk_sources, topk_slots, topk_miss_counts, miss_sources, miss_slots,
                          miss_counts, workspace, tiling);
}

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "fused_quant_lightning_indexer_manage_full_visible("
        "Tensor query, Tensor weights, Tensor key, Tensor query_scale, Tensor key_scale, "
        "Tensor block_table, Tensor query_ends, Tensor lengths, Tensor cache_tokens, "
        "Tensor request_state, Tensor pool_entries, Tensor(a!) slot_pool, "
        "Tensor(b!) topk_sources, Tensor(c!) topk_slots, Tensor(d!) topk_miss_counts, "
        "Tensor(e!) miss_sources, Tensor(f!) miss_slots, Tensor(g!) miss_counts) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_quant_lightning_indexer_manage_full_visible",
           TORCH_FN(fused_quant_lightning_indexer_manage_full_visible));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("fused_quant_lightning_indexer_manage_full_visible",
           TORCH_FN(fused_quant_lightning_indexer_manage_full_visible));
}

}  // namespace
