// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <limits>
#include <vector>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include "torch_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "quant_lightning_indexer_packed_tiling.h"
#include "aclrtlaunch_sgl_quant_lightning_indexer_packed.h"

namespace sglang::npu_kernel {
// Keep tiling owned by the queued callback. The generated host launcher accepts
// a pointer but synchronously copies the struct into its by-value kernel args.
static uint32_t aclrtlaunch_sgl_quant_lightning_indexer_packed(
    uint32_t blockDim, aclrtStream stream, void* query, void* packed_cache,
    void* weights, void* query_scale, void* seq_q, void* seq_k,
    void* block_table, void* output, void* workspace, QLITilingData tiling)
{
    return ::aclrtlaunch_sgl_quant_lightning_indexer_packed(
        blockDim, stream, query, packed_cache, weights, query_scale, seq_q, seq_k,
        block_table, output, workspace, &tiling);
}

at::Tensor quant_lightning_indexer_packed(const at::Tensor& query, const at::Tensor& packed_cache,
    const at::Tensor& weights, const at::Tensor& query_scale, const at::Tensor& seq_q,
    const at::Tensor& seq_k, const at::Tensor& block_table, int64_t topk, int64_t sparse_mode)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on NPU");
    const c10::OptionalDeviceGuard guard(query.device());
    std::vector<at::Tensor> keep_alive{query, packed_cache, weights, query_scale, seq_q, seq_k, block_table};
    for (const auto& tensor : keep_alive) {
        TORCH_CHECK(tensor.device() == query.device(), "all inputs must be on the same NPU device");
        TORCH_CHECK(tensor.is_contiguous(), "all inputs must be contiguous");
        TORCH_CHECK(at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND, "all inputs must use ND format");
    }
    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn, "query must be float8_e4m3fn");
    TORCH_CHECK(query.dim() == 3 && query.size(2) == QLIPacked::HEAD_DIM, "query must have TND shape [T, heads, 128]");
    TORCH_CHECK(packed_cache.scalar_type() == at::kByte && packed_cache.dim() == 2,
        "packed_cache must be uint8 [pages, P*132]");
    TORCH_CHECK(packed_cache.size(0) > 0 && packed_cache.size(1) % QLIPacked::TOKEN_BYTES == 0,
        "packed_cache must have positive pages and width P*132");
    auto page = packed_cache.size(1) / QLIPacked::TOKEN_BYTES;
    TORCH_CHECK(page >= 16 && page <= 1024 && page % 16 == 0,
        "page size must be a multiple of 16 in [16, 1024]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed_cache.data_ptr()) % 4 == 0,
        "packed_cache must be FP32 aligned");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be float32");
    TORCH_CHECK(weights.dim() == 2 && query_scale.dim() == 2 &&
        weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
        weights.sizes() == query_scale.sizes(), "weights/query_scale must have shape [T, heads]");
    TORCH_CHECK(seq_q.scalar_type() == at::kInt && seq_k.scalar_type() == at::kInt &&
        block_table.scalar_type() == at::kInt, "sequence lengths and block_table must be int32");
    TORCH_CHECK(seq_q.dim() == 1 && seq_k.dim() == 1 && seq_q.numel() == seq_k.numel(),
        "seq_q/seq_k must have shape [batch]");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == seq_k.numel(),
        "block_table must have shape [batch, max_pages]");
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in [1, 2048]");
    TORCH_CHECK(sparse_mode == 0 || sparse_mode == 3, "sparse_mode must be 0 or 3");
    const auto heads = query.size(1);
    TORCH_CHECK(heads == 8 || heads == 16 || heads == 24 || heads == 32 || heads == 64,
                "query head count must be 8, 16, 24, 32 or 64");
    TORCH_CHECK(seq_q.numel() > 0, "batch must be positive");
    TORCH_CHECK(block_table.size(1) > 0, "block_table must have positive width");
    constexpr auto maxDim = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(query.size(0) <= maxDim && seq_q.numel() <= maxDim &&
                block_table.size(1) <= maxDim / page, "shape exceeds uint32 tiling limits");
    auto output = at::empty({query.size(0), 1, topk}, query.options().dtype(at::kInt));
    if (query.size(0) == 0) return output;


    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "AscendC platform is unavailable");
    const uint32_t aic = platform->GetCoreNumAic();
    const uint32_t aiv = platform->GetCoreNumAiv();
    TORCH_CHECK(aic > 0 && aiv > 0, "Ascend950 compute cores are unavailable");
    const uint32_t blockDim = platform->CalcTschBlockDim(aiv, aic, aiv);
    uint32_t batch = seq_q.numel();
    uint32_t tokens = query.size(0);
    uint32_t groups = heads;
    uint32_t pageSize = page;
    uint32_t maxPages = block_table.size(1);
    uint32_t sparseCount = topk;
    uint32_t sparseMode = sparse_mode;
    const QLITilingData tiling = QLIPacked::MakeTilingData(
        batch, tokens, groups, pageSize, maxPages, sparseCount, sparseMode, blockDim);
    // The arch35 kernel uses four query rows, 128-key tiles and uint16 scores.
    // Direct launch passes user workspace directly, without an ACLNN system prefix.
    const uint64_t maxKeys = static_cast<uint64_t>(maxPages) * pageSize;
    const uint64_t workspaceSize = 4 * ((maxKeys + 127) / 128) * 128 * sizeof(uint16_t) * aic;
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));

    // EXEC_KERNEL_CMD captures raw pointers only. Record every allocation before
    // queueing so temporary inputs and workspace survive asynchronous execution.
    auto stream = c10_npu::getCurrentNPUStream(query.device().index());
    keep_alive.push_back(output);
    keep_alive.push_back(workspace);
    for (const auto& tensor : keep_alive) {
        if (tensor.numel()) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    // The host-computed tiling travels by value, including during graph capture.
    EXEC_KERNEL_CMD(sgl_quant_lightning_indexer_packed, blockDim,
                    query, packed_cache, weights, query_scale, seq_q, seq_k,
                    block_table, output, workspace, tiling);
    return output;
}
} // namespace sglang::npu_kernel
