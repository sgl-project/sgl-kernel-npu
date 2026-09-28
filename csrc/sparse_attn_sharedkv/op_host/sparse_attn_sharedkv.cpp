/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "aclrtlaunch_sparse_attn_sharedkv.h"
#include "ge_helper.h"
#include "sparse_attn_sharedkv_def.h"
#include "sparse_attn_sharedkv_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

using ge_helper::TilingContext;
using optiling::SASInfoParser;
using optiling::SASTilingCheck;
using optiling::SASTilingInfo;
using optiling::SparseAttnSharedkvTiling;
using optiling::SparseAttnSharedkvTilingData;

void CheckTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.device().type() == q.device().type() && tensor.device().index() == q.device().index(), name,
                " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &q, const char *name)
{
    if (tensor.has_value()) {
        CheckTensor(*tensor, q, name);
    }
}

at::Tensor Placeholder(const at::Tensor &q, at::ScalarType dtype)
{
    return at::empty({1}, q.options().dtype(dtype));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &ori_sparse_indices, const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table, const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode,
    int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q,
    c10::string_view layout_kv, bool return_softmax_lse)
{
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "sparse_attn_sharedkv: q must be float16 or bfloat16");
    TORCH_CHECK(q.device().type() == DEVICE_TYPE, "sparse_attn_sharedkv: q must be an NPU tensor");
    TORCH_CHECK(q.is_contiguous(), "sparse_attn_sharedkv: q must be contiguous");
    TORCH_CHECK((layout_q == "BSND" && q.dim() == 4) || (layout_q == "TND" && q.dim() == 3),
                "sparse_attn_sharedkv: q rank does not match layout_q");
    TORCH_CHECK(ori_kv.has_value(), "sparse_attn_sharedkv: ori_kv is required");
    TORCH_CHECK(sinks.has_value(), "sparse_attn_sharedkv: sinks is required");
    TORCH_CHECK(metadata.has_value(), "sparse_attn_sharedkv: metadata is required");
    TORCH_CHECK(layout_kv == "PA_ND", "sparse_attn_sharedkv: this port currently supports layout_kv='PA_ND' only");
    TORCH_CHECK(layout_q == "BSND" || layout_q == "TND", "sparse_attn_sharedkv: layout_q must be 'BSND' or 'TND'");
    TORCH_CHECK(!return_softmax_lse, "sparse_attn_sharedkv: return_softmax_lse=true is not supported");
    TORCH_CHECK(ori_block_table.has_value(), "sparse_attn_sharedkv: ori_block_table is required for PA_ND");
    TORCH_CHECK(seqused_kv.has_value(), "sparse_attn_sharedkv: seqused_kv is required for PA_ND");
    TORCH_CHECK(!cmp_kv.has_value() || cmp_block_table.has_value(),
                "sparse_attn_sharedkv: cmp_block_table is required when cmp_kv is present");
    TORCH_CHECK(!cmp_sparse_indices.has_value() || cmp_kv.has_value(),
                "sparse_attn_sharedkv: cmp_kv is required when cmp_sparse_indices is present");
    TORCH_CHECK(layout_q != "TND" || cu_seqlens_q.has_value(),
                "sparse_attn_sharedkv: cu_seqlens_q is required for TND query layout");
    TORCH_CHECK(layout_q != "BSND" || !cu_seqlens_q.has_value(),
                "sparse_attn_sharedkv: cu_seqlens_q must be None for BSND query layout");
    TORCH_CHECK(ori_win_left >= 0 && ori_win_right >= 0, "sparse_attn_sharedkv: window sizes must be non-negative");
    TORCH_CHECK(softmax_scale >= 0.0, "sparse_attn_sharedkv: softmax_scale must be non-negative");
    TORCH_CHECK(ori_kv->scalar_type() == q.scalar_type(), "sparse_attn_sharedkv: ori_kv dtype must match q dtype");
    TORCH_CHECK(!cmp_kv.has_value() || cmp_kv->scalar_type() == q.scalar_type(),
                "sparse_attn_sharedkv: cmp_kv dtype must match q dtype");
    TORCH_CHECK(sinks->scalar_type() == at::kFloat, "sparse_attn_sharedkv: sinks must be float32");
    TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->numel() == 1024,
                "sparse_attn_sharedkv: metadata must be int32 with 1024 elements");

    const int64_t oriKvStride = ori_kv->stride(0);
    const int64_t cmpKvStride = cmp_kv.has_value() ? cmp_kv->stride(0) : 0;
    int64_t qHeads = layout_q == "BSND" ? q.size(2) : q.size(1);
    TORCH_CHECK(qHeads == 64, "sparse_attn_sharedkv: the initial port supports exactly 64 query heads");
    auto checkInt32 = [](const c10::optional<at::Tensor> &tensor, const char *name) {
        TORCH_CHECK(!tensor.has_value() || tensor->scalar_type() == at::kInt, name, " must be int32");
    };
    checkInt32(ori_sparse_indices, "ori_sparse_indices");
    checkInt32(cmp_sparse_indices, "cmp_sparse_indices");
    checkInt32(ori_block_table, "ori_block_table");
    checkInt32(cmp_block_table, "cmp_block_table");
    checkInt32(cu_seqlens_q, "cu_seqlens_q");
    checkInt32(cu_seqlens_ori_kv, "cu_seqlens_ori_kv");
    checkInt32(cu_seqlens_cmp_kv, "cu_seqlens_cmp_kv");
    checkInt32(seqused_q, "seqused_q");
    checkInt32(seqused_kv, "seqused_kv");
    TORCH_CHECK(cmp_ratio >= 0 && cmp_ratio <= std::numeric_limits<uint32_t>::max() && ori_mask_mode >= 0 &&
                    ori_mask_mode <= std::numeric_limits<uint32_t>::max() && cmp_mask_mode >= 0 &&
                    cmp_mask_mode <= std::numeric_limits<uint32_t>::max() && oriKvStride >= 0 &&
                    oriKvStride <= std::numeric_limits<uint32_t>::max() && cmpKvStride >= 0 &&
                    cmpKvStride <= std::numeric_limits<uint32_t>::max() &&
                    ori_win_left <= std::numeric_limits<uint32_t>::max() &&
                    ori_win_right <= std::numeric_limits<uint32_t>::max(),
                "sparse_attn_sharedkv: integer attributes exceed the supported uint32 range");

    CheckTensor(*ori_kv, q, "ori_kv");
    CheckOptionalTensor(cmp_kv, q, "cmp_kv");
    CheckOptionalTensor(ori_sparse_indices, q, "ori_sparse_indices");
    CheckOptionalTensor(cmp_sparse_indices, q, "cmp_sparse_indices");
    CheckOptionalTensor(ori_block_table, q, "ori_block_table");
    CheckOptionalTensor(cmp_block_table, q, "cmp_block_table");
    CheckOptionalTensor(cu_seqlens_q, q, "cu_seqlens_q");
    CheckOptionalTensor(cu_seqlens_ori_kv, q, "cu_seqlens_ori_kv");
    CheckOptionalTensor(cu_seqlens_cmp_kv, q, "cu_seqlens_cmp_kv");
    CheckOptionalTensor(seqused_q, q, "seqused_q");
    CheckOptionalTensor(seqused_kv, q, "seqused_kv");
    CheckTensor(*sinks, q, "sinks");
    CheckTensor(*metadata, q, "metadata");

    auto attention_out = at::empty_like(q);
    auto softmax_lse = at::empty({0}, q.options().dtype(at::kFloat));

    SASHost::SparseAttnSharedkv op("sparse_attn_sharedkv");
    op.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    op.SetAttrAny("cmp_ratio", static_cast<uint32_t>(cmp_ratio));
    op.SetAttrAny("ori_mask_mode", static_cast<uint32_t>(ori_mask_mode));
    op.SetAttrAny("cmp_mask_mode", static_cast<uint32_t>(cmp_mask_mode));
    op.SetAttrAny("ori_kv_stride", static_cast<uint32_t>(oriKvStride));
    op.SetAttrAny("cmp_kv_stride", static_cast<uint32_t>(cmpKvStride));
    op.SetAttrAny("ori_win_left", static_cast<uint32_t>(ori_win_left));
    op.SetAttrAny("ori_win_right", static_cast<uint32_t>(ori_win_right));
    op.SetAttrStr("layout_q", std::string(layout_q));
    op.SetAttrStr("layout_kv", std::string(layout_kv));
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>("sparse_attn_sharedkv");
    auto scalarType = q.scalar_type();
    op.SetToContext(context, scalarType);
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
    context->RegisterTensor(seqused_q, true);
    context->RegisterTensor(seqused_kv, true);
    context->RegisterTensor(sinks, true);
    context->RegisterTensor(metadata, true);
    context->RegisterTensor(attention_out, false);
    context->RegisterTensor(softmax_lse, false);

    SASTilingInfo info;
    SASInfoParser parser(context.get());
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS, "sparse_attn_sharedkv: parsing tiling inputs failed");
    SASTilingCheck checker(info);
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS, "sparse_attn_sharedkv: tiling validation failed");
    SparseAttnSharedkvTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, "sparse_attn_sharedkv: tiling failed");

    auto tilingTensor = context->GetTilingTensor(tiling.GetTilingData());
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, q.options().dtype(at::kByte));

    auto qPlaceholder = Placeholder(q, q.scalar_type());
    auto intPlaceholder = Placeholder(q, at::kInt);
    auto oriKvLaunch = *ori_kv;
    auto cmpKvLaunch = cmp_kv.value_or(qPlaceholder);
    auto oriSparseLaunch = ori_sparse_indices.value_or(intPlaceholder);
    auto cmpSparseLaunch = cmp_sparse_indices.value_or(intPlaceholder);
    auto oriBlockLaunch = ori_block_table.value_or(intPlaceholder);
    auto cmpBlockLaunch = cmp_block_table.value_or(intPlaceholder);
    auto cuQLaunch = cu_seqlens_q.value_or(intPlaceholder);
    auto cuOriKvLaunch = cu_seqlens_ori_kv.value_or(intPlaceholder);
    auto cuCmpKvLaunch = cu_seqlens_cmp_kv.value_or(intPlaceholder);
    auto seqQLaunch = seqused_q.value_or(intPlaceholder);
    auto seqKvLaunch = seqused_kv.value_or(intPlaceholder);
    auto sinksLaunch = *sinks;
    auto metadataLaunch = *metadata;

    const uint32_t blockDim = tiling.GetBlockDim();
    EXEC_KERNEL_CMD(sparse_attn_sharedkv, blockDim, q, oriKvLaunch, cmpKvLaunch, oriSparseLaunch, cmpSparseLaunch,
                    oriBlockLaunch, cmpBlockLaunch, cuQLaunch, cuOriKvLaunch, cuCmpKvLaunch, seqQLaunch, seqKvLaunch,
                    sinksLaunch, metadataLaunch, attention_out, softmax_lse, workspace, tilingTensor);
    return {attention_out, softmax_lse};
}

}  // namespace sglang::npu_kernel
