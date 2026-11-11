/** Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#include "aclnn_runtime.h"
#include "aclnn_sgl_kv_quant_sparse_attn_sharedkv.h"
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
        case at::kBFloat16:
            return ACL_BF16;
        case at::kFloat8_e4m3fn:
            return ACL_FLOAT8_E4M3FN;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        default:
            TORCH_CHECK(false, "Unsupported quantized sparse-attention dtype: ", tensor.scalar_type());
    }
    return ACL_FLOAT;
}
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
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, opName, ": q must be an NPU tensor");
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
    auto softmaxLse = at::empty({1}, q.options().dtype(at::kFloat)).narrow(0, 0, 0);

    const c10::DeviceGuard guard(q.device());
    std::vector<std::shared_ptr<aclTensor>> inputs;
    inputs.push_back(sgl_cann::tensor_descriptor(q, ACL_BF16));
    for (const auto *tensor :
         {&ori_kv, &cmp_kv, &ori_sparse_indices, &cmp_sparse_indices, &ori_block_table, &cmp_block_table, &cu_seqlens_q,
          &cu_seqlens_ori_kv, &cu_seqlens_cmp_kv, &seqused_q, &seqused_kv, &sinks, &metadata}) {
        inputs.push_back(tensor->has_value() ? sgl_cann::tensor_descriptor(**tensor, ToAclType(**tensor)) : nullptr);
    }
    auto outDesc = sgl_cann::tensor_descriptor(attentionOut, ACL_BF16);
    auto lseDesc = sgl_cann::tensor_descriptor(softmaxLse, ACL_FLOAT);
    static auto query = reinterpret_cast<decltype(&aclnnSglKvQuantSparseAttnSharedkvGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglKvQuantSparseAttnSharedkvGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglKvQuantSparseAttnSharedkv)>(
        sgl_cann::op_api_symbol("aclnnSglKvQuantSparseAttnSharedkv"));
    std::string layoutQString(layout_q), layoutKvString(layout_kv);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status =
        query(inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get(), inputs[4].get(), inputs[5].get(),
              inputs[6].get(), inputs[7].get(), inputs[8].get(), inputs[9].get(), inputs[10].get(), inputs[11].get(),
              inputs[12].get(), inputs[13].get(), kv_quant_mode, tile_size, rope_head_dim,
              static_cast<float>(softmax_scale), cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
              layoutQString.data(), layoutKvString.data(), oriKvStride, cmpKvStride, return_softmax_lse, outDesc.get(),
              lseDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "Quantized sparse attention workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace exceeds Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, q.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    std::vector<at::Tensor> tensors{q, attentionOut, softmaxLse, workspace};
    for (const auto *tensor :
         {&ori_kv, &cmp_kv, &ori_sparse_indices, &cmp_sparse_indices, &ori_block_table, &cmp_block_table, &cu_seqlens_q,
          &cu_seqlens_ori_kv, &cu_seqlens_cmp_kv, &seqused_q, &seqused_kv, &sinks, &metadata}) {
        if (tensor->has_value()) tensors.push_back(**tensor);
    }
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, inputs, outDesc, lseDesc, layoutQString, layoutKvString, workspace, workspaceSize, executor,
                   aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "Quantized sparse attention execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglKvQuantSparseAttnSharedkv", launch);
    return {attentionOut, softmaxLse};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "kv_quant_sparse_attn_sharedkv(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
        "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
        "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
        "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
        "Tensor? sinks=None, Tensor? metadata=None, int kv_quant_mode=1, int tile_size=64, "
        "int rope_head_dim=64, float softmax_scale=1.0, int cmp_ratio=1, "
        "int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', "
        "bool return_softmax_lse=False) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("kv_quant_sparse_attn_sharedkv", &kv_quant_sparse_attn_sharedkv);
}
