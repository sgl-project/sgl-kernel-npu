/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <memory>
#include <string>
#include <tuple>

#include "acl/acl.h"
#include "ge_helper.h"
#include "torch_helper.h"
#include "sparse_attn_sharedkv_tree_def.h"
#include "sparse_attn_sharedkv_tree_tiling.h"
#include "aclrtlaunch_sparse_attn_sharedkv_tree_fp16.h"
#include "aclrtlaunch_sparse_attn_sharedkv_tree_bf16.h"

namespace sglang {
namespace npu_kernel {

namespace {

using ge_helper::TilingContext;
using optiling::SparseAttnSharedkvTreeTiling;
using SparseAttnSharedkvTreeHost::SparseAttnSharedkvTree;

at::Tensor EmptyLike(const at::Tensor &q, at::ScalarType dtype)
{
    return at::empty({0}, q.options().dtype(dtype));
}

c10::optional<at::Tensor> NonEmpty(const c10::optional<at::Tensor> &t)
{
    if (!t.has_value() || t->numel() == 0) {
        return c10::nullopt;
    }
    return t;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv_tree(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &cmp_tree_kv, const c10::optional<at::Tensor> &tree_ancestors,
    const c10::optional<at::Tensor> &tree_prefix, double softmax_scale, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    TORCH_CHECK(q.device().type() == DEVICE_TYPE, "sparse_attn_sharedkv_tree: q must be an NPU tensor");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "sparse_attn_sharedkv_tree: q must be fp16 or bf16");

    SparseAttnSharedkvTree op("SparseAttnSharedkvTree");
    auto context = std::make_shared<TilingContext>("SparseAttnSharedkvTree");
    TORCH_CHECK(context != nullptr, "TilingContext is null");

    at::Tensor attn_out = at::empty(q.sizes(), q.options());
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        auto lseSizes = q.sizes().vec();
        lseSizes.back() = 1;
        softmax_lse = at::empty(lseSizes, q.options().dtype(at::kFloat));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(at::kFloat));
    }

    c10::optional<at::Tensor> sinks_eff = sinks;
    if (sinks_eff.has_value() && sinks_eff->dim() == 2 && sinks_eff->size(0) == 1) {
        sinks_eff = sinks_eff->squeeze(0);
    }

    c10::optional<at::Tensor> seqused_q_eff = NonEmpty(seqused_q);

    op.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    op.SetAttrAny("cmp_ratio", static_cast<uint32_t>(cmp_ratio));
    op.SetAttrAny("ori_mask_mode", static_cast<uint32_t>(ori_mask_mode));
    op.SetAttrAny("cmp_mask_mode", static_cast<uint32_t>(cmp_mask_mode));
    op.SetAttrAny("ori_kv_stride", static_cast<uint32_t>(ori_kv.has_value() ? ori_kv->stride(0) : 0));
    op.SetAttrAny("cmp_kv_stride", static_cast<uint32_t>(cmp_kv.has_value() ? cmp_kv->stride(0) : 0));
    op.SetAttrAny("ori_win_left", static_cast<uint32_t>(ori_win_left));
    op.SetAttrAny("ori_win_right", static_cast<uint32_t>(ori_win_right));
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);
    op.SetAttrStr("layout_q", std::string(layout_q));
    op.SetAttrStr("layout_kv", std::string(layout_kv));

    auto qType = q.scalar_type();
    op.SetToContext(context, qType);
    context->RegisterTensor(q, true);
    context->RegisterTensor(ori_kv, true);
    context->RegisterTensor(cmp_kv, true);
    context->RegisterTensor(ori_sparse_indices, true);
    context->RegisterTensor(cmp_sparse_indices, true);
    context->RegisterTensor(ori_block_table, true);
    context->RegisterTensor(cmp_block_table, true);
    context->RegisterTensor(cu_seqlens_q, true);
    context->RegisterTensor(cu_seqlens_ori_kv, true);
    context->RegisterTensor(cu_seqlens_cmp_kv, true);
    context->RegisterTensor(seqused_q_eff, true);
    context->RegisterTensor(seqused_kv, true);
    context->RegisterTensor(sinks_eff, true);
    context->RegisterTensor(metadata, true);
    context->RegisterTensor(cmp_tree_kv, true);
    context->RegisterTensor(tree_ancestors, true);
    context->RegisterTensor(tree_prefix, true);
    context->RegisterTensor(attn_out, false);
    context->RegisterTensor(softmax_lse, false);

    SparseAttnSharedkvTreeTiling tiling(context.get());
    TORCH_CHECK(optiling::RunSparseAttnSharedkvTreeTiling(*context, tiling) == ge::GRAPH_SUCCESS,
                "sparse_attn_sharedkv_tree tiling failed");

    uint32_t blockDim = tiling.GetBlockDim();
    const auto &td = tiling.GetTilingData();
    const auto &bp = td.baseParams;
    const uint32_t kvBytes = static_cast<uint32_t>(q.element_size());
    const uint64_t coreCount = bp.usedCoreNum;
    const uint64_t mmBytes = static_cast<uint64_t>(bp.mmResUbSize) * coreCount * 2;
    const uint64_t bmmBytes = static_cast<uint64_t>(bp.bmm2ResUbSize) * coreCount * 2;
    const uint64_t userWorkspace =
        mmBytes * sizeof(float) + mmBytes * kvBytes + bmmBytes * sizeof(float) + bmmBytes * sizeof(float) +
        coreCount * 512ULL * 512ULL * 4ULL * kvBytes + coreCount * 2ULL * 128ULL * 4ULL * sizeof(int32_t);
    context->SetWorkspaceSizes(static_cast<size_t>(userWorkspace));
    size_t workspaceSize = context->GetWorkspaceSize();
    at::Tensor workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(q.options().device()));
    at::Tensor tilingTensor = context->GetTilingTensor(tiling.GetTilingData());

    at::Tensor oriKvT = ori_kv.has_value() ? *ori_kv : EmptyLike(q, q.scalar_type());
    at::Tensor cmpKvT = cmp_kv.has_value() ? *cmp_kv : EmptyLike(q, q.scalar_type());
    at::Tensor oriSparseT = ori_sparse_indices.has_value() ? *ori_sparse_indices : EmptyLike(q, at::kInt);
    at::Tensor cmpSparseT = cmp_sparse_indices.has_value() ? *cmp_sparse_indices : EmptyLike(q, at::kInt);
    at::Tensor oriBlockT = ori_block_table.has_value() ? *ori_block_table : EmptyLike(q, at::kInt);
    at::Tensor cmpBlockT = cmp_block_table.has_value() ? *cmp_block_table : EmptyLike(q, at::kInt);
    at::Tensor cuQ = cu_seqlens_q.has_value() ? *cu_seqlens_q : EmptyLike(q, at::kInt);
    at::Tensor cuOriKv = cu_seqlens_ori_kv.has_value() ? *cu_seqlens_ori_kv : EmptyLike(q, at::kInt);
    at::Tensor cuCmpKv = cu_seqlens_cmp_kv.has_value() ? *cu_seqlens_cmp_kv : EmptyLike(q, at::kInt);
    at::Tensor seqQ = seqused_q_eff.has_value() ? *seqused_q_eff : EmptyLike(q, at::kInt);
    at::Tensor seqKv = seqused_kv.has_value() ? *seqused_kv : EmptyLike(q, at::kInt);
    at::Tensor sinksT = sinks_eff.has_value() ? *sinks_eff : EmptyLike(q, at::kFloat);
    at::Tensor metadataT = metadata.has_value() ? *metadata : EmptyLike(q, at::kInt);
    at::Tensor treeKvT = cmp_tree_kv.has_value() ? *cmp_tree_kv : EmptyLike(q, q.scalar_type());
    at::Tensor ancestorsT = tree_ancestors.has_value() ? *tree_ancestors : EmptyLike(q, at::kLong);
    at::Tensor prefixT = tree_prefix.has_value() ? *tree_prefix : EmptyLike(q, at::kInt);

    if (q.scalar_type() == at::kHalf) {
        EXEC_KERNEL_CMD(sparse_attn_sharedkv_tree_fp16, blockDim, q, oriKvT, cmpKvT, oriSparseT, cmpSparseT,
                        oriBlockT, cmpBlockT, cuQ, cuOriKv, cuCmpKv, seqQ, seqKv, sinksT, metadataT,
                        treeKvT, ancestorsT, prefixT, attn_out, softmax_lse, workspace, tilingTensor);
    } else {
        EXEC_KERNEL_CMD(sparse_attn_sharedkv_tree_bf16, blockDim, q, oriKvT, cmpKvT, oriSparseT, cmpSparseT,
                        oriBlockT, cmpBlockT, cuQ, cuOriKv, cuCmpKv, seqQ, seqKv, sinksT, metadataT,
                        treeKvT, ancestorsT, prefixT, attn_out, softmax_lse, workspace, tilingTensor);
    }
    return std::make_tuple(attn_out, softmax_lse);
}

}  // namespace npu_kernel
}  // namespace sglang
