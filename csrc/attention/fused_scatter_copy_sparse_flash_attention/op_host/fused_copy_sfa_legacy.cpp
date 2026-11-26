// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the BSD 3-Clause License.

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>

#include "aclrtlaunch_fused_scatter_copy_sparse_flash_attention_combined.h"
#include "aclrtlaunch_fused_scatter_copy_sparse_flash_attention_split.h"
#include "fused_copy_sfa_tiling_data.h"
#include "fused_copy_sfa.h"
#include "ge_helper.h"
#include "sparse_flash_attention_def.h"
#include "sparse_flash_attention_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

constexpr int64_t kPageSize = 128;
constexpr int64_t kKvDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kTokenStride = 576;

void LaunchCopySfa(bool split, uint32_t blockDim, const at::Tensor &query, const at::Tensor &key,
                   const at::Tensor &value, const at::Tensor &sparseIndices, const at::Tensor &cacheTokens,
                   const at::Tensor &hbmBlockTable, const at::Tensor &actualSeqLengthsQuery,
                   const at::Tensor &actualSeqLengthsKv, const at::Tensor &queryRope, void *hbmKeyRope,
                   void *dramKeyRope, void *dramKvCache, const at::Tensor &dramBlockTable,
                   const at::Tensor &topkSourceIds, const at::Tensor &topkMissCounts, const at::Tensor &missSourceIds,
                   const at::Tensor &missDstSlots, const at::Tensor &missCounts, const at::Tensor &attentionOut,
                   const at::Tensor &workspace, const at::Tensor &tiling, void *current, void *positions, void *valid,
                   void *requestSlots, void *reqToToken, void *residentEnd)
{
    if (split) {
        EXEC_KERNEL_CMD(fused_scatter_copy_sparse_flash_attention_split, blockDim, query, key, value, sparseIndices,
                        cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope,
                        dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts, missSourceIds,
                        missDstSlots, missCounts, attentionOut, current, positions, valid, requestSlots, reqToToken,
                        residentEnd, workspace, tiling);
    } else {
        EXEC_KERNEL_CMD(fused_scatter_copy_sparse_flash_attention_combined, blockDim, query, key, value, sparseIndices,
                        cacheTokens, hbmBlockTable, actualSeqLengthsQuery, actualSeqLengthsKv, queryRope, hbmKeyRope,
                        dramKeyRope, dramKvCache, dramBlockTable, topkSourceIds, topkMissCounts, missSourceIds,
                        missDstSlots, missCounts, attentionOut, current, positions, valid, requestSlots, reqToToken,
                        residentEnd, workspace, tiling);
    }
}

}  // namespace

void LaunchFusedCopySfa(const at::Tensor &query, const at::Tensor &query_rope, at::Tensor &hbm_ckv,
                        at::Tensor *hbm_rope, const at::Tensor &actual_seq_lengths_query,
                        const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens,
                        const at::Tensor &topk_src_ids, const at::Tensor &topk_dst_slots,
                        const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
                        const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts,
                        const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table, int64_t dram_kv_dev_ptr,
                        at::Tensor &attention_out, double scale_value, bool full_visible, bool resident_refresh_tail,
                        const c10::optional<at::Tensor> &current, const c10::optional<at::Tensor> &positions,
                        const c10::optional<at::Tensor> &valid, const c10::optional<at::Tensor> &request_slots,
                        const c10::optional<at::Tensor> &req_to_token, const c10::optional<at::Tensor> &resident_end,
                        int64_t host_token_capacity, int64_t dram_rope_dev_ptr)
{
    const bool resident = current.has_value();
    const int64_t totalQueryTokens = query.size(0);
    const int64_t hbmBlockCount = hbm_ckv.size(0);
    const int64_t rowStride = kTokenStride;
    auto cacheKey = hbm_rope ? hbm_ckv
                             : at::as_strided(hbm_ckv, {hbmBlockCount, kPageSize, 1, kKvDim},
                                              {kPageSize * rowStride, rowStride, rowStride, 1}, 0);
    auto cacheRope = hbm_rope ? *hbm_rope
                              : at::as_strided(hbm_ckv, {hbmBlockCount, kPageSize, 1, kRopeDim},
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
    auto sfa_indices = topk_dst_slots.unsqueeze(1);
    context->RegisterTensor(sfa_indices, true);
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
    fusedTilingData.splitHost = dram_rope_dev_ptr != 0;
    fusedTilingData.fullVisible = full_visible ? 1U : 0U;
    fusedTilingData.residentRefreshTail = resident_refresh_tail ? 1U : 0U;
    fusedTilingData.residentEnabled = resident ? 1U : 0U;
    if (resident) {
        fusedTilingData.residentRows = static_cast<uint32_t>(req_to_token->size(0));
        fusedTilingData.residentContext = static_cast<uint32_t>(req_to_token->stride(0));
        fusedTilingData.residentHostRows = static_cast<uint32_t>(host_token_capacity);
        fusedTilingData.residentCurrentRows = static_cast<uint32_t>(totalQueryTokens);
        fusedTilingData.residentHbmPages = static_cast<uint32_t>(hbm_ckv.size(0));
    }
    fusedTilingData.copyCap = static_cast<uint32_t>(topk_src_ids.size(1));
    fusedTilingData.missCap = static_cast<uint32_t>(miss_src_ids.size(1));
    fusedTilingData.hbmMaxBlockNum = static_cast<uint32_t>(hbm_block_table.size(1));
    fusedTilingData.dramMaxBlockNum = static_cast<uint32_t>(dram_block_table.size(1));
    std::array<uint8_t, sizeof(FusedCopySfaTilingData)> tilingBytes{};
    std::memcpy(tilingBytes.data(), &fusedTilingData, sizeof(fusedTilingData));
    auto tilingTensor = context->GetTilingTensor(tilingBytes);
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, query.options().dtype(at::kByte));
    auto *dramKv = reinterpret_cast<void *>(static_cast<uintptr_t>(dram_kv_dev_ptr));
    auto *dramRope = reinterpret_cast<void *>(
        static_cast<uintptr_t>(dram_rope_dev_ptr ? dram_rope_dev_ptr : dram_kv_dev_ptr + kKvDim * 2));
    auto *hbmRope = cacheRope.data_ptr();

    LaunchCopySfa(hbm_rope != nullptr, tiling.GetBlockDim(), query, cacheKey, cacheKey, topk_dst_slots,
                  num_cache_tokens, hbm_block_table, actual_seq_lengths_query, actual_seq_lengths_kv, query_rope,
                  hbmRope, dramRope, dramKv, dram_block_table, topk_src_ids, topk_miss_counts, miss_src_ids,
                  miss_dst_slots, miss_counts, attention_out, workspace, tilingTensor,
                  resident ? current->data_ptr() : nullptr, resident ? positions->data_ptr() : nullptr,
                  resident ? valid->data_ptr() : nullptr, resident ? request_slots->data_ptr() : nullptr,
                  resident ? req_to_token->data_ptr() : nullptr, resident ? resident_end->data_ptr() : nullptr);
}

}  // namespace sglang::npu_kernel
