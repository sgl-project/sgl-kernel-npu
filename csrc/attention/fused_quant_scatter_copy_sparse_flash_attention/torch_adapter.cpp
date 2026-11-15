// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "a5_copy_sfa_ops.h"
#include "aclnn_runtime.h"
#include "aclnn_sgl_fused_quant_scatter_copy_sparse_flash_attention.h"
#include "op_kernel/fused_quant_scatter_copy_sparse_flash_attention_contract.h"

#include <array>
#include <ATen/core/Dict.h>
#include <cmath>
#include <limits>
#include <string>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

namespace {

using fused_quant_scatter_copy_sfa::ATTENTION_OUTPUT_DIM;
using fused_quant_scatter_copy_sfa::CACHE_BLOCK_SIZE;
using fused_quant_scatter_copy_sfa::COPY_CAPACITY;
using fused_quant_scatter_copy_sfa::KV_HEAD_COUNT;
using fused_quant_scatter_copy_sfa::MAX_BLOCKS_PER_REQUEST;
using fused_quant_scatter_copy_sfa::MAX_QUERY_COUNT;
using fused_quant_scatter_copy_sfa::PACKED_KV_ROW_BYTES;
using fused_quant_scatter_copy_sfa::QUERY_DIM;
using fused_quant_scatter_copy_sfa::TOPK_CAPACITY;

c10::Dict<std::string, int64_t> A5CopySfaCapabilities()
{
    c10::Dict<std::string, int64_t> capabilities;
    capabilities.insert("split_query", 0);
    capabilities.insert("ring_tail", 0);
    capabilities.insert("schema_version", 1);
    capabilities.insert("packed_row_bytes", PACKED_KV_ROW_BYTES);
    capabilities.insert("legacy_args", 17);
    return capabilities;
}

aclDataType AclQueryType(const at::Tensor &query)
{
    return query.scalar_type() == at::kBFloat16 ? ACL_BF16 : ACL_FLOAT16;
}

std::shared_ptr<aclTensor> TensorAtAddress(const at::Tensor &tensor, aclDataType dtype, void *address)
{
    auto descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, tensor.strides().data(), 0,
                                      ACL_FORMAT_ND, tensor.sizes().data(), tensor.dim(), address);
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

void Validate(const at::Tensor &query, const at::Tensor &hbmKvBytes, const at::Tensor &dramKvBytes,
              int64_t dramKvDevPtr, const at::Tensor &hbmBlockTable, const at::Tensor &dramBlockTable,
              const at::Tensor &topkDstSlots, const at::Tensor &topkSrcIds, const at::Tensor &topkMissCounts,
              const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv,
              const at::Tensor &numCacheTokens, const at::Tensor &missSrcIds, const at::Tensor &missDstSlots,
              const at::Tensor &missCounts, double scaleValue, const at::Tensor &attentionOut)
{
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == QUERY_DIM &&
                    fused_quant_scatter_copy_sfa::IsSupportedHeadCount(static_cast<uint32_t>(query.size(1))),
                "query must be [T, H, 576], H in {8, 16, 32, 64, 128}.");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kHalf,
                "query must be bfloat16 or float16.");
    TORCH_CHECK(query.device().is_privateuseone() || query.is_meta(), "C8 CopySFA tensors must be on NPU.");
    const int64_t batchSize = numCacheTokens.numel();
    const int64_t totalQueries = query.size(0);
    TORCH_CHECK(numCacheTokens.dim() == 1 && batchSize > 0 && totalQueries >= batchSize &&
                    totalQueries <= MAX_QUERY_COUNT * batchSize,
                "num_cache_tokens must be [B] and B <= T <= 16B.");
    for (const auto *cache : std::array<const at::Tensor *, 2>{&hbmKvBytes, &dramKvBytes}) {
        TORCH_CHECK(cache->dim() == 4 && cache->size(0) > 0 && cache->size(1) == CACHE_BLOCK_SIZE &&
                        cache->size(2) == KV_HEAD_COUNT && cache->size(3) == PACKED_KV_ROW_BYTES &&
                        cache->scalar_type() == at::kChar,
                    "KV caches must be int8 [blocks, 128, 1, 656].");
    }
    for (const auto *table : {&hbmBlockTable, &dramBlockTable}) {
        TORCH_CHECK(table->dim() == 2 && table->size(0) == batchSize && table->size(1) > 0 &&
                        table->size(1) <= MAX_BLOCKS_PER_REQUEST,
                    "block tables must be [B, 1..2048].");
    }
    for (const auto *ids : {&topkDstSlots, &topkSrcIds}) {
        TORCH_CHECK(
            ids->dim() == 3 && ids->size(0) == totalQueries && ids->size(1) == 1 && ids->size(2) == TOPK_CAPACITY,
            "topk indices must be [T, 1, 2048].");
    }
    for (const auto *ids : {&missSrcIds, &missDstSlots}) {
        TORCH_CHECK(ids->dim() == 3 && ids->size(0) == batchSize && ids->size(1) == 1 && ids->size(2) == COPY_CAPACITY,
                    "miss indices must be [B, 1, 32768].");
    }
    TORCH_CHECK(topkMissCounts.sizes() == at::IntArrayRef({totalQueries}) &&
                    actualSeqLengthsQuery.sizes() == at::IntArrayRef({batchSize}) &&
                    actualSeqLengthsKv.sizes() == at::IntArrayRef({batchSize}) &&
                    missCounts.sizes() == at::IntArrayRef({batchSize}),
                "count and sequence metadata shapes do not match T/B.");
    for (const auto *metadata : std::array<const at::Tensor *, 11>{
             &hbmBlockTable, &dramBlockTable, &topkDstSlots, &topkSrcIds, &topkMissCounts, &actualSeqLengthsQuery,
             &actualSeqLengthsKv, &numCacheTokens, &missSrcIds, &missDstSlots, &missCounts}) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt, "all metadata tensors must be int32.");
    }
    TORCH_CHECK(attentionOut.dim() == 3 && attentionOut.size(0) == totalQueries &&
                    attentionOut.size(1) == query.size(1) && attentionOut.size(2) == ATTENTION_OUTPUT_DIM &&
                    attentionOut.scalar_type() == query.scalar_type(),
                "attention_out must be [T, H, 512] with the query dtype.");
    TORCH_CHECK(std::isfinite(static_cast<float>(scaleValue)), "scale_value must be finite in FP32.");
    const auto device = query.device();
    for (const auto *tensor : std::array<const at::Tensor *, 15>{
             &query, &hbmKvBytes, &dramKvBytes, &hbmBlockTable, &dramBlockTable, &topkDstSlots, &topkSrcIds,
             &topkMissCounts, &actualSeqLengthsQuery, &actualSeqLengthsKv, &numCacheTokens, &missSrcIds, &missDstSlots,
             &missCounts, &attentionOut}) {
        const bool validDevice = tensor == &dramKvBytes ? (tensor->device().is_cpu() || tensor->device() == device)
                                                        : tensor->device() == device;
        TORCH_CHECK(validDevice && tensor->is_contiguous(), "all tensors must be contiguous on the expected device.");
    }
    if (!query.is_meta()) TORCH_CHECK(dramKvDevPtr > 0, "DRAM must have a registered device-visible address.");
}

}  // namespace

namespace sglang::npu_kernel {

void fused_quant_scatter_copy_sparse_flash_attention(
    const at::Tensor &query, at::Tensor hbmKvBytes, const at::Tensor &dramKvBytes, int64_t dramKvDevPtr,
    const at::Tensor &hbmBlockTable, const at::Tensor &dramBlockTable, const at::Tensor &topkDstSlots,
    const at::Tensor &topkSrcIds, const at::Tensor &topkMissCounts, const at::Tensor &actualSeqLengthsQuery,
    const at::Tensor &actualSeqLengthsKv, const at::Tensor &numCacheTokens, const at::Tensor &missSrcIds,
    const at::Tensor &missDstSlots, const at::Tensor &missCounts, double scaleValue, at::Tensor attentionOut)
{
    Validate(query, hbmKvBytes, dramKvBytes, dramKvDevPtr, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds,
             topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots,
             missCounts, scaleValue, attentionOut);
    if (query.is_meta()) return;
    c10_npu::OptionalNPUGuard guard(query.device());
    const auto queryType = AclQueryType(query);
    std::array<std::shared_ptr<aclTensor>, 15> descriptors{
        sgl_cann::tensor_descriptor(query, queryType),
        sgl_cann::tensor_descriptor(hbmKvBytes, ACL_INT8),
        TensorAtAddress(dramKvBytes, ACL_INT8, reinterpret_cast<void *>(static_cast<uintptr_t>(dramKvDevPtr))),
        sgl_cann::tensor_descriptor(hbmBlockTable, ACL_INT32),
        sgl_cann::tensor_descriptor(dramBlockTable, ACL_INT32),
        sgl_cann::tensor_descriptor(topkDstSlots, ACL_INT32),
        sgl_cann::tensor_descriptor(topkSrcIds, ACL_INT32),
        sgl_cann::tensor_descriptor(topkMissCounts, ACL_INT32),
        sgl_cann::tensor_descriptor(actualSeqLengthsQuery, ACL_INT32),
        sgl_cann::tensor_descriptor(actualSeqLengthsKv, ACL_INT32),
        sgl_cann::tensor_descriptor(numCacheTokens, ACL_INT32),
        sgl_cann::tensor_descriptor(missSrcIds, ACL_INT32),
        sgl_cann::tensor_descriptor(missDstSlots, ACL_INT32),
        sgl_cann::tensor_descriptor(missCounts, ACL_INT32),
        sgl_cann::tensor_descriptor(attentionOut, queryType),
    };
    static auto queryWorkspace =
        reinterpret_cast<decltype(&aclnnSglFusedQuantScatterCopySparseFlashAttentionGetWorkspaceSize)>(
            sgl_cann::op_api_symbol("aclnnSglFusedQuantScatterCopySparseFlashAttentionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglFusedQuantScatterCopySparseFlashAttention)>(
        sgl_cann::op_api_symbol("aclnnSglFusedQuantScatterCopySparseFlashAttention"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    TORCH_CHECK(queryWorkspace(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                               descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), descriptors[7].get(),
                               descriptors[8].get(), descriptors[9].get(), descriptors[10].get(), descriptors[11].get(),
                               descriptors[12].get(), descriptors[13].get(), static_cast<float>(scaleValue),
                               descriptors[14].get(), &workspaceSize, &executor) == ACL_SUCCESS,
                "C8 CopySFA workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "C8 CopySFA workspace exceeds Tensor allocation limits.");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::array<at::Tensor, 15> owners{
        query,        hbmKvBytes,     dramKvBytes,           hbmBlockTable,      dramBlockTable, topkDstSlots,
        topkSrcIds,   topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds,
        missDstSlots, missCounts,     attentionOut};
    for (const auto &tensor : owners) {
        if (tensor.device().is_privateuseone()) tensor.record_stream(stream);
    }
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [owners, descriptors, workspace, workspaceSize, executor, aclStream, execute]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "C8 CopySFA execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglFusedQuantScatterCopySparseFlashAttention", launch);
}

}  // namespace sglang::npu_kernel

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "fused_quant_scatter_copy_sparse_flash_attention("
        "Tensor query, Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, int dram_kv_dev_ptr, "
        "Tensor hbm_block_table, Tensor dram_block_table, Tensor topk_dst_slots, Tensor topk_src_ids, "
        "Tensor topk_miss_counts, Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
        "Tensor num_cache_tokens, Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, "
        "float scale_value, Tensor(b!) attention_out) -> ()");
    m.def("a5_copy_sfa_capabilities() -> Dict(str, int)", TORCH_FN(A5CopySfaCapabilities));
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
