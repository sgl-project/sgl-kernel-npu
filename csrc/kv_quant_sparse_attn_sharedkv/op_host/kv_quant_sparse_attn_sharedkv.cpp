/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <limits>
#include <memory>
#include <string>
#include <tuple>

#include "acl/acl.h"
#include "aclrtlaunch_kv_quant_sparse_attn_sharedkv.h"
#include "ge_helper.h"
#include "kv_quant_sparse_attn_sharedkv_check.h"
#include "kv_quant_sparse_attn_sharedkv_def.h"
#include "kv_quant_sparse_attn_sharedkv_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {

using ge_helper::TilingContext;
using optiling::IsKvQuantSparseAttnSharedkvDispatchKeySupported;
using optiling::KvQuantSASInfoParser;
using optiling::KvQuantSASTilingCheck;
using optiling::KvQuantSASTilingInfo;
using optiling::KvQuantSparseAttnSharedkvTiling;

void CheckTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &q, const char *name)
{
    if (tensor.has_value()) {
        CheckTensor(*tensor, q, name);
    }
}

void *OptionalDataPtr(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() ? TorchNpuHelper::ConvertType(*tensor) : nullptr;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv, const c10::optional<at::Tensor> &cmp_kv,
    const c10::optional<at::Tensor> &ori_sparse_indices, const c10::optional<at::Tensor> &cmp_sparse_indices,
    const c10::optional<at::Tensor> &ori_block_table, const c10::optional<at::Tensor> &cmp_block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, int64_t kv_quant_mode, int64_t tile_size, int64_t rope_head_dim,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    constexpr const char *opName = "kv_quant_sparse_attn_sharedkv";
    TORCH_CHECK(q.device().type() == DEVICE_TYPE, opName, ": q must be an NPU tensor");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, opName, ": q must be bfloat16");
    TORCH_CHECK(q.is_contiguous(), opName, ": q must be contiguous");
    TORCH_CHECK((layout_q == "BSND" && q.dim() == 4) || (layout_q == "TND" && q.dim() == 3), opName,
                ": q rank does not match layout_q");
    TORCH_CHECK(layout_kv == "PA_ND", opName, ": only PA_ND KV layout is supported");
    TORCH_CHECK(ori_kv.has_value(), opName, ": ori_kv is required");
    TORCH_CHECK(ori_block_table.has_value(), opName, ": ori_block_table is required");
    TORCH_CHECK(seqused_kv.has_value(), opName, ": seqused_kv is required");
    TORCH_CHECK(sinks.has_value(), opName, ": sinks is required");
    TORCH_CHECK(metadata.has_value(), opName, ": metadata is required");
    TORCH_CHECK(layout_q != "TND" || cu_seqlens_q.has_value(), opName, ": TND requires cu_seqlens_q");
    TORCH_CHECK(!cmp_kv.has_value() || cmp_block_table.has_value(), opName, ": cmp_kv requires cmp_block_table");
    TORCH_CHECK(!cmp_sparse_indices.has_value() || cmp_kv.has_value(), opName, ": cmp_sparse_indices requires cmp_kv");
    TORCH_CHECK(!return_softmax_lse, opName, ": return_softmax_lse=true is unsupported");
    TORCH_CHECK(!cu_seqlens_ori_kv.has_value() && !cu_seqlens_cmp_kv.has_value(), opName,
                ": cu_seqlens_ori_kv and cu_seqlens_cmp_kv are unsupported");

    // The source commit's original sparse path is defined for per-query TND
    // slot rows in SWA mode. Its SCFA path shares the sparse-index reader and
    // cannot safely combine ori_sparse_indices with compressed KV.
    if (ori_sparse_indices.has_value()) {
        TORCH_CHECK(layout_q == "TND" && !cmp_kv.has_value(), opName,
                    ": ori_sparse_indices requires TND and SWA mode (cmp_kv=None)");
        TORCH_CHECK(ori_sparse_indices->dim() == 3 && ori_sparse_indices->size(0) == q.size(0) &&
                        ori_sparse_indices->size(1) == 1 && ori_sparse_indices->size(2) > 0,
                    opName, ": ori_sparse_indices must have shape [T, 1, K] with T=q.size(0)");
    }

    TORCH_CHECK(kv_quant_mode == 1 && tile_size == 64 && rope_head_dim == 64, opName,
                ": requires kv_quant_mode=1, tile_size=64, rope_head_dim=64");
    TORCH_CHECK(ori_mask_mode == 4 && cmp_mask_mode == 3, opName, ": requires mask modes 4 and 3");
    TORCH_CHECK(ori_win_left >= 0 && ori_win_left <= std::numeric_limits<int32_t>::max() && ori_win_right >= 0 &&
                    ori_win_right <= std::numeric_limits<int32_t>::max(),
                opName, ": window sizes must fit non-negative int32");
    TORCH_CHECK(cmp_ratio == 1 || cmp_ratio == 4 || cmp_ratio == 128, opName, ": cmp_ratio must be 1, 4 or 128");
    TORCH_CHECK(ori_kv->scalar_type() == at::kFloat8_e4m3fn, opName, ": ori_kv must be float8_e4m3fn packed KV");
    TORCH_CHECK(!cmp_kv.has_value() || cmp_kv->scalar_type() == at::kFloat8_e4m3fn, opName,
                ": cmp_kv must be float8_e4m3fn packed KV");
    TORCH_CHECK(sinks->scalar_type() == at::kFloat, opName, ": sinks must be float32");
    TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->numel() == 1024, opName,
                ": metadata must be int32 with 1024 elements");

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

    CheckTensor(*ori_kv, q, "ori_kv");
    CheckOptionalTensor(cmp_kv, q, "cmp_kv");
    CheckOptionalTensor(ori_sparse_indices, q, "ori_sparse_indices");
    CheckOptionalTensor(cmp_sparse_indices, q, "cmp_sparse_indices");
    CheckTensor(*ori_block_table, q, "ori_block_table");
    CheckOptionalTensor(cmp_block_table, q, "cmp_block_table");
    CheckOptionalTensor(cu_seqlens_q, q, "cu_seqlens_q");
    CheckOptionalTensor(cu_seqlens_ori_kv, q, "cu_seqlens_ori_kv");
    CheckOptionalTensor(cu_seqlens_cmp_kv, q, "cu_seqlens_cmp_kv");
    CheckOptionalTensor(seqused_q, q, "seqused_q");
    CheckTensor(*seqused_kv, q, "seqused_kv");
    CheckTensor(*sinks, q, "sinks");
    CheckTensor(*metadata, q, "metadata");

    const int64_t oriKvStride = ori_kv->stride(0);
    const int64_t cmpKvStride = cmp_kv.has_value() ? cmp_kv->stride(0) : 0;
    TORCH_CHECK(oriKvStride >= 0 && oriKvStride <= std::numeric_limits<int32_t>::max() && cmpKvStride >= 0 &&
                    cmpKvStride <= std::numeric_limits<int32_t>::max(),
                opName, ": KV stride(0) must fit non-negative int32");

    auto attentionOut = at::empty_like(q);
    auto softmaxLse = at::empty({0}, q.options().dtype(at::kFloat));

    KvQuantSASHost::KvQuantSparseAttnSharedkv op(opName);
    op.SetAttrAny("kv_quant_mode", kv_quant_mode);
    op.SetAttrAny("tile_size", tile_size);
    op.SetAttrAny("rope_head_dim", rope_head_dim);
    op.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    op.SetAttrAny("cmp_ratio", cmp_ratio);
    op.SetAttrAny("ori_mask_mode", static_cast<uint32_t>(ori_mask_mode));
    op.SetAttrAny("cmp_mask_mode", static_cast<uint32_t>(cmp_mask_mode));
    op.SetAttrAny("ori_win_left", ori_win_left);
    op.SetAttrAny("ori_win_right", ori_win_right);
    op.SetAttrStr("layout_q", std::string(layout_q));
    op.SetAttrStr("layout_kv", std::string(layout_kv));
    op.SetAttrAny("ori_kv_stride0", oriKvStride);
    op.SetAttrAny("cmp_kv_stride0", cmpKvStride);
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>(opName);
    auto qScalarType = q.scalar_type();
    op.SetToContext(context, qScalarType);
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
    context->RegisterTensor(attentionOut, false);
    context->RegisterTensor(softmaxLse, false);

    KvQuantSASTilingInfo info;
    KvQuantSASInfoParser parser(context.get());
    kv_quant_sas_log::ClearLastError();
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS, opName,
                ": parsing tiling inputs failed: ", kv_quant_sas_log::LastError());
    KvQuantSASTilingCheck checker(info);
    kv_quant_sas_log::ClearLastError();
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS, opName,
                ": tiling validation failed: ", kv_quant_sas_log::LastError());
    KvQuantSparseAttnSharedkvTiling tiling(context.get());
    kv_quant_sas_log::ClearLastError();
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, opName,
                ": tiling failed: ", kv_quant_sas_log::LastError());
    TORCH_CHECK(IsKvQuantSparseAttnSharedkvDispatchKeySupported(tiling.GetTilingKey()), opName,
                ": unsupported dispatch key ", tiling.GetTilingKey());

    auto tilingTensor = context->GetTilingTensor(tiling.GetTilingData());
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, q.options().dtype(at::kByte));
    // The kernel uses nullptr to distinguish absent optional inputs. A non-empty
    // placeholder would turn those branches on and could read past its bounds.
    void *cmpKvPtr = OptionalDataPtr(cmp_kv);
    void *oriSparseIndicesPtr = OptionalDataPtr(ori_sparse_indices);
    void *cmpSparseIndicesPtr = OptionalDataPtr(cmp_sparse_indices);
    void *cmpBlockTablePtr = OptionalDataPtr(cmp_block_table);
    void *cuSeqlensQPtr = OptionalDataPtr(cu_seqlens_q);
    void *cuSeqlensOriKvPtr = OptionalDataPtr(cu_seqlens_ori_kv);
    void *cuSeqlensCmpKvPtr = OptionalDataPtr(cu_seqlens_cmp_kv);
    void *sequsedQPtr = OptionalDataPtr(seqused_q);

    const uint32_t blockDim = tiling.GetBlockDim();
    EXEC_KERNEL_CMD(kv_quant_sparse_attn_sharedkv, blockDim, q, *ori_kv, cmpKvPtr, oriSparseIndicesPtr,
                    cmpSparseIndicesPtr, *ori_block_table, cmpBlockTablePtr, cuSeqlensQPtr, cuSeqlensOriKvPtr,
                    cuSeqlensCmpKvPtr, sequsedQPtr, *seqused_kv, *sinks, *metadata, attentionOut, softmaxLse, workspace,
                    tilingTensor);
    return {attentionOut, softmaxLse};
}

}  // namespace sglang::npu_kernel
