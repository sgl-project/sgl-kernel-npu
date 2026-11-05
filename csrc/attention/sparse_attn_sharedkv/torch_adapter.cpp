// BSD 3-Clause License
// Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

// Framework bridge for the migrated sparse attention with shared KV operator.
// Validation mirrors the legacy Host wrapper; all computation stays in the
// copied arch32 kernel.

#include "aclnn_runtime.h"
#include "aclnn_sgl_sparse_attn_sharedkv.h"

#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {

aclDataType ToAclType(const at::Tensor &tensor)
{
    switch (tensor.scalar_type()) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        default:
            TORCH_CHECK(false, "sparse_attn_sharedkv does not support dtype ", tensor.scalar_type());
            return ACL_FLOAT;
    }
}

void CheckTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &q, const char *name)
{
    if (tensor.has_value() && tensor->defined()) {
        CheckTensor(*tensor, q, name);
    }
}

bool IsPresent(const c10::optional<at::Tensor> &tensor)
{
    // The legacy Host treats any supplied tensor, including an empty one, as
    // present; keep that presence contract for the tiling parser.
    return tensor.has_value() && tensor->defined();
}

std::tuple<at::Tensor, at::Tensor> sgl_sparse_attn_sharedkv(
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
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "sparse_attn_sharedkv: q must be an NPU tensor");
    TORCH_CHECK(q.is_contiguous(), "sparse_attn_sharedkv: q must be contiguous");
    TORCH_CHECK((layout_q == "BSND" && q.dim() == 4) || (layout_q == "TND" && q.dim() == 3),
                "sparse_attn_sharedkv: q rank does not match layout_q");
    TORCH_CHECK(ori_kv.has_value() && ori_kv->defined(), "sparse_attn_sharedkv: ori_kv is required");
    TORCH_CHECK(sinks.has_value() && sinks->defined(), "sparse_attn_sharedkv: sinks is required");
    TORCH_CHECK(metadata.has_value() && metadata->defined(), "sparse_attn_sharedkv: metadata is required");
    TORCH_CHECK(layout_kv == "PA_ND", "sparse_attn_sharedkv: this port currently supports layout_kv='PA_ND' only");
    TORCH_CHECK(layout_q == "BSND" || layout_q == "TND", "sparse_attn_sharedkv: layout_q must be 'BSND' or 'TND'");
    TORCH_CHECK(!return_softmax_lse, "sparse_attn_sharedkv: return_softmax_lse=true is not supported");
    TORCH_CHECK(ori_block_table.has_value() && ori_block_table->defined(),
                "sparse_attn_sharedkv: ori_block_table is required for PA_ND");
    TORCH_CHECK(seqused_kv.has_value() && seqused_kv->defined(),
                "sparse_attn_sharedkv: seqused_kv is required for PA_ND");
    TORCH_CHECK(!IsPresent(cmp_kv) || IsPresent(cmp_block_table),
                "sparse_attn_sharedkv: cmp_block_table is required when cmp_kv is present");
    TORCH_CHECK(!IsPresent(cmp_sparse_indices) || IsPresent(cmp_kv),
                "sparse_attn_sharedkv: cmp_kv is required when cmp_sparse_indices is present");
    TORCH_CHECK(layout_q != "TND" || IsPresent(cu_seqlens_q),
                "sparse_attn_sharedkv: cu_seqlens_q is required for TND query layout");
    TORCH_CHECK(layout_q != "BSND" || !IsPresent(cu_seqlens_q),
                "sparse_attn_sharedkv: cu_seqlens_q must be None for BSND query layout");
    TORCH_CHECK(ori_win_left >= 0 && ori_win_right >= 0, "sparse_attn_sharedkv: window sizes must be non-negative");
    TORCH_CHECK(softmax_scale >= 0.0, "sparse_attn_sharedkv: softmax_scale must be non-negative");
    TORCH_CHECK(ori_kv->scalar_type() == q.scalar_type(), "sparse_attn_sharedkv: ori_kv dtype must match q dtype");
    TORCH_CHECK(!IsPresent(cmp_kv) || cmp_kv->scalar_type() == q.scalar_type(),
                "sparse_attn_sharedkv: cmp_kv dtype must match q dtype");
    TORCH_CHECK(sinks->scalar_type() == at::kFloat, "sparse_attn_sharedkv: sinks must be float32");
    TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->numel() == 1024,
                "sparse_attn_sharedkv: metadata must be int32 with 1024 elements");

    const int64_t oriKvStride = ori_kv->stride(0);
    const int64_t cmpKvStride = IsPresent(cmp_kv) ? cmp_kv->stride(0) : 0;
    const int64_t qHeads = layout_q == "BSND" ? q.size(2) : q.size(1);
    TORCH_CHECK(qHeads == 64, "sparse_attn_sharedkv: the initial port supports exactly 64 query heads");
    auto checkInt32 = [](const c10::optional<at::Tensor> &tensor, const char *name) {
        TORCH_CHECK(!IsPresent(tensor) || tensor->scalar_type() == at::kInt, name, " must be int32");
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
    constexpr int64_t u32Max = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(cmp_ratio >= 0 && cmp_ratio <= u32Max && ori_mask_mode >= 0 && ori_mask_mode <= u32Max &&
                    cmp_mask_mode >= 0 && cmp_mask_mode <= u32Max && oriKvStride >= 0 && oriKvStride <= u32Max &&
                    cmpKvStride >= 0 && cmpKvStride <= u32Max && ori_win_left <= u32Max && ori_win_right <= u32Max,
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

    const c10::DeviceGuard deviceGuard(q.device());

    auto attentionOut = at::empty_like(q);
    // Keep a valid data pointer for ACLNN while exposing the legacy empty output.
    auto softmaxLse = at::empty({1}, q.options().dtype(at::kFloat)).narrow(0, 0, 0);

    auto optionalDescriptor = [](const c10::optional<at::Tensor> &tensor) -> std::shared_ptr<aclTensor> {
        if (!IsPresent(tensor)) {
            return {};
        }
        return sgl_cann::tensor_descriptor(*tensor, ToAclType(*tensor));
    };

    auto qDesc = sgl_cann::tensor_descriptor(q, ToAclType(q));
    auto oriKvDesc = sgl_cann::tensor_descriptor(*ori_kv, ToAclType(*ori_kv));
    auto cmpKvDesc = optionalDescriptor(cmp_kv);
    auto oriSparseDesc = optionalDescriptor(ori_sparse_indices);
    auto cmpSparseDesc = optionalDescriptor(cmp_sparse_indices);
    auto oriBlockDesc = optionalDescriptor(ori_block_table);
    auto cmpBlockDesc = optionalDescriptor(cmp_block_table);
    auto cuQDesc = optionalDescriptor(cu_seqlens_q);
    auto cuOriKvDesc = optionalDescriptor(cu_seqlens_ori_kv);
    auto cuCmpKvDesc = optionalDescriptor(cu_seqlens_cmp_kv);
    auto seqQDesc = optionalDescriptor(seqused_q);
    auto seqKvDesc = optionalDescriptor(seqused_kv);
    auto sinksDesc = sgl_cann::tensor_descriptor(*sinks, ACL_FLOAT);
    auto metadataDesc = sgl_cann::tensor_descriptor(*metadata, ACL_INT32);
    auto outDesc = sgl_cann::tensor_descriptor(attentionOut, ToAclType(attentionOut));
    auto lseDesc = sgl_cann::tensor_descriptor(softmaxLse, ACL_FLOAT);

    static auto query = reinterpret_cast<decltype(&aclnnSglSparseAttnSharedkvGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseAttnSharedkvGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglSparseAttnSharedkv)>(sgl_cann::op_api_symbol("aclnnSglSparseAttnSharedkv"));
    // The generated ACLNN interface exposes string attributes as mutable
    // char*, so keep writable copies of the exact layout names.
    std::string layoutQString(layout_q);
    std::string layoutKvString(layout_kv);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = query(qDesc.get(), oriKvDesc.get(), cmpKvDesc.get(), oriSparseDesc.get(), cmpSparseDesc.get(),
                              oriBlockDesc.get(), cmpBlockDesc.get(), cuQDesc.get(), cuOriKvDesc.get(),
                              cuCmpKvDesc.get(), seqQDesc.get(), seqKvDesc.get(), sinksDesc.get(), metadataDesc.get(),
                              static_cast<float>(softmax_scale), cmp_ratio, ori_mask_mode, cmp_mask_mode, oriKvStride,
                              cmpKvStride, ori_win_left, ori_win_right, layoutQString.data(), layoutKvString.data(),
                              return_softmax_lse, outDesc.get(), lseDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "sparse_attn_sharedkv workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{q, *ori_kv, *sinks, *metadata, attentionOut, softmaxLse, workspace};
    for (const auto *tensor : {&cmp_kv, &ori_sparse_indices, &cmp_sparse_indices, &ori_block_table, &cmp_block_table,
                               &cu_seqlens_q, &cu_seqlens_ori_kv, &cu_seqlens_cmp_kv, &seqused_q, &seqused_kv}) {
        if (IsPresent(*tensor)) {
            tensors.push_back(**tensor);
        }
    }
    for (const auto &tensor : tensors) {
        tensor.record_stream(stream);
    }
    auto aclStream = stream.stream(false);

    std::vector<std::shared_ptr<aclTensor>> descriptors{
        qDesc,   oriKvDesc,   cmpKvDesc,   oriSparseDesc, cmpSparseDesc, oriBlockDesc, cmpBlockDesc,
        cuQDesc, cuOriKvDesc, cuCmpKvDesc, seqQDesc,      seqKvDesc,     sinksDesc,    metadataDesc};
    std::vector<std::shared_ptr<aclTensor>> outputs{outDesc, lseDesc};
    auto launch = [tensors, descriptors, outputs, layoutQString, layoutKvString, workspace, workspaceSize, executor,
                   aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "sparse_attn_sharedkv ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSparseAttnSharedkv", launch);
    return std::make_tuple(attentionOut, softmaxLse);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sparse_attn_sharedkv(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
        "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
        "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
        "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
        "Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=0, int cmp_ratio=0, "
        "int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=128, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', "
        "bool return_softmax_lse=False) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sparse_attn_sharedkv", &sgl_sparse_attn_sharedkv);
}
