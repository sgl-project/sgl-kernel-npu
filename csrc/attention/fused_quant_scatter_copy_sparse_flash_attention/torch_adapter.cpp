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
#include <vector>
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
    capabilities.insert("split_query", 1);
    capabilities.insert("full_visible", 1);
    capabilities.insert("resident_stage", 1);
    capabilities.insert("resident_source_gather", 1);
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

void ValidateV2(const at::Tensor &queryNope, const at::Tensor &queryRope, const at::Tensor &hbmKvBytes,
                const at::Tensor &dramKvBytes, int64_t dramKvDevPtr, const at::Tensor &hbmBlockTable,
                const at::Tensor &dramBlockTable, const at::Tensor &topkDstSlots, const at::Tensor &topkSrcIds,
                const at::Tensor &topkMissCounts, const at::Tensor &actualSeqLengthsQuery,
                const at::Tensor &actualSeqLengthsKv, const at::Tensor &numCacheTokens, const at::Tensor &missSrcIds,
                const at::Tensor &missDstSlots, const at::Tensor &missCounts, double scaleValue,
                const at::Tensor &attentionOut, const std::optional<at::Tensor> &current,
                const std::optional<at::Tensor> &positions, const std::optional<at::Tensor> &currentValid,
                const std::optional<at::Tensor> &requestSlots, const std::optional<at::Tensor> &reqToToken,
                const std::optional<at::Tensor> &residentEnd)
{
    TORCH_CHECK(queryNope.dim() == 3 && queryNope.size(0) > 0 && queryNope.size(2) == ATTENTION_OUTPUT_DIM &&
                    fused_quant_scatter_copy_sfa::IsSupportedHeadCount(static_cast<uint32_t>(queryNope.size(1))),
                "query_nope must be [T, H, 512], H in {8, 16, 32, 64, 128}.");
    TORCH_CHECK(queryRope.dim() == 3 && queryRope.size(0) == queryNope.size(0) &&
                    queryRope.size(1) == queryNope.size(1) &&
                    queryRope.size(2) == fused_quant_scatter_copy_sfa::ROPE_DIM,
                "query_rope must be [T, H, 64].");
    TORCH_CHECK((queryNope.scalar_type() == at::kBFloat16 || queryNope.scalar_type() == at::kHalf) &&
                    queryRope.scalar_type() == queryNope.scalar_type(),
                "split queries must share the bfloat16 or float16 dtype.");
    TORCH_CHECK(queryNope.device().is_privateuseone() || queryNope.is_meta(), "C8 CopySFA v2 tensors must be on NPU.");
    const int64_t batchSize = numCacheTokens.numel();
    const int64_t totalQueries = queryNope.size(0);
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
                    "v2 block tables must be [B, 1..2048].");
    }
    for (const auto *ids : {&topkDstSlots, &topkSrcIds}) {
        TORCH_CHECK(ids->sizes() == at::IntArrayRef({totalQueries, 1, TOPK_CAPACITY}),
                    "v2 topk indices must be [T, 1, 2048].");
    }
    for (const auto *ids : {&missSrcIds, &missDstSlots}) {
        TORCH_CHECK(ids->sizes() == at::IntArrayRef({batchSize, COPY_CAPACITY}), "v2 miss indices must be [B, 32768].");
    }
    TORCH_CHECK(topkMissCounts.sizes() == at::IntArrayRef({totalQueries}) &&
                    actualSeqLengthsQuery.sizes() == at::IntArrayRef({batchSize}) &&
                    actualSeqLengthsKv.sizes() == at::IntArrayRef({batchSize}) &&
                    missCounts.sizes() == at::IntArrayRef({batchSize}),
                "v2 count and sequence metadata shapes do not match T/B.");
    for (const auto *metadata : std::array<const at::Tensor *, 11>{
             &hbmBlockTable, &dramBlockTable, &topkDstSlots, &topkSrcIds, &topkMissCounts, &actualSeqLengthsQuery,
             &actualSeqLengthsKv, &numCacheTokens, &missSrcIds, &missDstSlots, &missCounts}) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt, "all v2 metadata tensors must be int32.");
    }
    TORCH_CHECK(attentionOut.sizes() == at::IntArrayRef({totalQueries, queryNope.size(1), ATTENTION_OUTPUT_DIM}) &&
                    attentionOut.scalar_type() == queryNope.scalar_type(),
                "attention_out must be [T, H, 512] with the query dtype.");
    TORCH_CHECK(std::isfinite(static_cast<float>(scaleValue)), "scale_value must be finite in FP32.");

    const auto device = queryNope.device();
    for (const auto *tensor : std::array<const at::Tensor *, 16>{
             &queryNope, &queryRope, &hbmKvBytes, &dramKvBytes, &hbmBlockTable, &dramBlockTable, &topkDstSlots,
             &topkSrcIds, &topkMissCounts, &actualSeqLengthsQuery, &actualSeqLengthsKv, &numCacheTokens, &missSrcIds,
             &missDstSlots, &missCounts, &attentionOut}) {
        const bool validDevice = tensor == &dramKvBytes ? (tensor->device().is_cpu() || tensor->device() == device)
                                                        : tensor->device() == device;
        TORCH_CHECK(validDevice && tensor->is_contiguous(),
                    "all v2 tensors must be contiguous on the expected device.");
    }
    const bool stageResident = current.has_value();
    for (const auto *tensor : {&current, &positions, &currentValid, &requestSlots, &reqToToken, &residentEnd}) {
        TORCH_CHECK(tensor->has_value() == stageResident, "v2 resident staging requires all six tensors.");
        if (stageResident) {
            TORCH_CHECK((*tensor)->device() == device && (*tensor)->is_contiguous(),
                        "v2 resident tensors must be contiguous on the query device.");
        }
    }
    if (stageResident) {
        TORCH_CHECK(current->scalar_type() == at::kChar &&
                        current->sizes() == at::IntArrayRef({totalQueries, PACKED_KV_ROW_BYTES}),
                    "current must be int8 [T, 656].");
        TORCH_CHECK(positions->scalar_type() == at::kLong && positions->sizes() == at::IntArrayRef({totalQueries}) &&
                        currentValid->scalar_type() == at::kBool && currentValid->sizes() == positions->sizes(),
                    "positions and current_valid must be int64 [T] and bool [T].");
        TORCH_CHECK(requestSlots->scalar_type() == at::kInt && requestSlots->sizes() == at::IntArrayRef({batchSize}) &&
                        reqToToken->scalar_type() == at::kInt && reqToToken->dim() == 2 && reqToToken->size(0) > 1 &&
                        reqToToken->size(0) <= UINT32_MAX && reqToToken->size(1) > 0 &&
                        reqToToken->size(1) <= UINT32_MAX && residentEnd->scalar_type() == at::kInt &&
                        residentEnd->sizes() == at::IntArrayRef({reqToToken->size(0)}),
                    "resident staging requires int32 request_slots [B], req_to_token [R, S], resident_end [R].");
    }
    if (!queryNope.is_meta()) TORCH_CHECK(dramKvDevPtr > 0, "DRAM must have a registered device-visible address.");
}

void RunAclnn(const at::Tensor &query, at::Tensor hbmKvBytes, const at::Tensor &dramKvBytes, int64_t dramKvDevPtr,
              const at::Tensor &hbmBlockTable, const at::Tensor &dramBlockTable, const at::Tensor &topkDstSlots,
              const at::Tensor &topkSrcIds, const at::Tensor &topkMissCounts, const at::Tensor &actualSeqLengthsQuery,
              const at::Tensor &actualSeqLengthsKv, const at::Tensor &numCacheTokens, const at::Tensor &missSrcIds,
              const at::Tensor &missDstSlots, const at::Tensor &missCounts, double scaleValue, at::Tensor attentionOut,
              const std::optional<at::Tensor> &queryRope, const std::optional<at::Tensor> &current,
              const std::optional<at::Tensor> &positions, const std::optional<at::Tensor> &currentValid,
              const std::optional<at::Tensor> &requestSlots, const std::optional<at::Tensor> &reqToToken,
              const std::optional<at::Tensor> &residentEnd, bool nativeLimPlan)
{
    c10_npu::OptionalNPUGuard guard(query.device());
    const auto queryType = AclQueryType(query);
    auto optionalDescriptor = [](const std::optional<at::Tensor> &tensor,
                                 aclDataType dtype) -> std::shared_ptr<aclTensor> {
        return tensor.has_value() ? sgl_cann::tensor_descriptor(*tensor, dtype) : std::shared_ptr<aclTensor>{};
    };
    std::array<std::shared_ptr<aclTensor>, 22> descriptors{
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
        optionalDescriptor(queryRope, queryType),
        optionalDescriptor(current, ACL_INT8),
        optionalDescriptor(positions, ACL_INT64),
        optionalDescriptor(currentValid, ACL_BOOL),
        optionalDescriptor(requestSlots, ACL_INT32),
        optionalDescriptor(reqToToken, ACL_INT32),
        optionalDescriptor(residentEnd, ACL_INT32),
        sgl_cann::tensor_descriptor(attentionOut, queryType),
    };
    static auto queryWorkspace =
        reinterpret_cast<decltype(&aclnnSglFusedQuantScatterCopySparseFlashAttentionGetWorkspaceSize)>(
            sgl_cann::op_api_symbol("aclnnSglFusedQuantScatterCopySparseFlashAttentionGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglFusedQuantScatterCopySparseFlashAttention)>(
        sgl_cann::op_api_symbol("aclnnSglFusedQuantScatterCopySparseFlashAttention"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = queryWorkspace(
        descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(), descriptors[4].get(),
        descriptors[5].get(), descriptors[6].get(), descriptors[7].get(), descriptors[8].get(), descriptors[9].get(),
        descriptors[10].get(), descriptors[11].get(), descriptors[12].get(), descriptors[13].get(),
        descriptors[14].get(), descriptors[15].get(), descriptors[16].get(), descriptors[17].get(),
        descriptors[18].get(), descriptors[19].get(), descriptors[20].get(), scaleValue, nativeLimPlan,
        descriptors[21].get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "C8 CopySFA workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "C8 CopySFA workspace exceeds Tensor allocation limits.");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> owners{
        query,        hbmKvBytes,     dramKvBytes,           hbmBlockTable,      dramBlockTable, topkDstSlots,
        topkSrcIds,   topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds,
        missDstSlots, missCounts,     attentionOut,          workspace};
    for (const auto *tensor :
         {&queryRope, &current, &positions, &currentValid, &requestSlots, &reqToToken, &residentEnd}) {
        if (tensor->has_value()) owners.push_back(**tensor);
    }
    for (const auto &tensor : owners) {
        if (tensor.device().is_privateuseone()) tensor.record_stream(stream);
    }
    auto aclStream = stream.stream(false);
    auto launch = [owners, descriptors, workspace, workspaceSize, executor, aclStream, execute]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "C8 CopySFA execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglFusedQuantScatterCopySparseFlashAttention", launch);
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
    RunAclnn(query, hbmKvBytes, dramKvBytes, dramKvDevPtr, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds,
             topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots,
             missCounts, scaleValue, attentionOut, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
             std::nullopt, std::nullopt, false);
}

void fused_quant_scatter_copy_sparse_flash_attention_v2(
    const at::Tensor &queryNope, const at::Tensor &queryRope, at::Tensor hbmKvBytes, const at::Tensor &dramKvBytes,
    int64_t dramKvDevPtr, const at::Tensor &hbmBlockTable, const at::Tensor &dramBlockTable,
    const at::Tensor &topkDstSlots, const at::Tensor &topkSrcIds, const at::Tensor &topkMissCounts,
    const at::Tensor &actualSeqLengthsQuery, const at::Tensor &actualSeqLengthsKv, const at::Tensor &numCacheTokens,
    const at::Tensor &missSrcIds, const at::Tensor &missDstSlots, const at::Tensor &missCounts, double scaleValue,
    at::Tensor attentionOut, const std::optional<at::Tensor> &current, const std::optional<at::Tensor> &positions,
    const std::optional<at::Tensor> &currentValid, const std::optional<at::Tensor> &requestSlots,
    const std::optional<at::Tensor> &reqToToken, const std::optional<at::Tensor> &residentEnd, bool nativeLimPlan)
{
    ValidateV2(queryNope, queryRope, hbmKvBytes, dramKvBytes, dramKvDevPtr, hbmBlockTable, dramBlockTable, topkDstSlots,
               topkSrcIds, topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds,
               missDstSlots, missCounts, scaleValue, attentionOut, current, positions, currentValid, requestSlots,
               reqToToken, residentEnd);
    if (queryNope.is_meta()) return;
    RunAclnn(queryNope, hbmKvBytes, dramKvBytes, dramKvDevPtr, hbmBlockTable, dramBlockTable, topkDstSlots, topkSrcIds,
             topkMissCounts, actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, missSrcIds, missDstSlots,
             missCounts, scaleValue, attentionOut, queryRope, current, positions, currentValid, requestSlots,
             reqToToken, residentEnd, nativeLimPlan);
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
    m.def(
        "fused_quant_scatter_copy_sparse_flash_attention_v2("
        "Tensor query_nope, Tensor query_rope, Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
        "int dram_kv_dev_ptr, Tensor hbm_block_table, Tensor dram_block_table, "
        "Tensor topk_dst_slots, Tensor topk_src_ids, Tensor topk_miss_counts, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, Tensor num_cache_tokens, "
        "Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, float scale_value, "
        "Tensor(b!) attention_out, Tensor? current=None, Tensor? positions=None, Tensor? current_valid=None, "
        "Tensor? request_slots=None, Tensor? req_to_token=None, Tensor(c!)? resident_end=None, "
        "*, bool native_lim_plan=False) -> ()");
    m.def("a5_copy_sfa_capabilities() -> Dict(str, int)", TORCH_FN(A5CopySfaCapabilities));
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
    m.impl("fused_quant_scatter_copy_sparse_flash_attention_v2",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention_v2));
}
TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
    m.impl("fused_quant_scatter_copy_sparse_flash_attention_v2",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention_v2));
}
}  // namespace
