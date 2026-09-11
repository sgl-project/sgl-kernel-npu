#include <memory>
#include <string>
#include <tuple>

#include "aclrtlaunch_kv_quant_sparse_attn_sharedkv.h"
#include "ge_helper.h"
#include "kv_quant_sparse_attn_sharedkv_def.h"
#include "kv_quant_sparse_attn_sharedkv_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {
using ge_helper::TilingContext;
using optiling::KvQuantSASInfoParser;
using optiling::KvQuantSASTilingCheck;
using optiling::KvQuantSASTilingInfo;
using optiling::KvQuantSparseAttnSharedkvTiling;

at::Tensor Placeholder(const at::Tensor &q, at::ScalarType dtype) {
    return at::empty({1}, q.options().dtype(dtype));
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t kv_quant_mode, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t tile_size,
    int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv,
    bool return_softmax_lse) {
    TORCH_CHECK(q.device().type() == DEVICE_TYPE && q.scalar_type() == at::kBFloat16,
                "kv_quant_sparse_attn_sharedkv: q must be BF16 on NPU");
    TORCH_CHECK(ori_kv.has_value() && metadata.has_value() && sinks.has_value(),
                "kv_quant_sparse_attn_sharedkv: ori_kv, sinks and metadata are required");
    TORCH_CHECK(ori_kv->scalar_type() == at::kFloat8_e4m3fn,
                "kv_quant_sparse_attn_sharedkv: ori_kv must be FP8_E4M3FN");
    TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->numel() == 1024,
                "kv_quant_sparse_attn_sharedkv: metadata must be int32[1024]");

    auto output = at::empty_like(q);
    auto lse_shape = q.sizes().vec();
    if (!lse_shape.empty()) {
        lse_shape.back() = 1;
    }
    auto lse = return_softmax_lse ? at::empty(lse_shape, q.options().dtype(at::kFloat))
                                  : at::empty({0}, q.options().dtype(at::kFloat));

    KvQuantSASHost::KvQuantSparseAttnSharedkv op("kv_quant_sparse_attn_sharedkv");
    op.SetAttrAny("kv_quant_mode", static_cast<int>(kv_quant_mode));
    op.SetAttrAny("tile_size", static_cast<int>(tile_size));
    op.SetAttrAny("rope_head_dim", static_cast<int>(rope_head_dim));
    op.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    op.SetAttrAny("cmp_ratio", static_cast<int>(cmp_ratio));
    op.SetAttrAny("ori_mask_mode", static_cast<int>(ori_mask_mode));
    op.SetAttrAny("cmp_mask_mode", static_cast<int>(cmp_mask_mode));
    op.SetAttrAny("ori_win_left", static_cast<int>(ori_win_left));
    op.SetAttrAny("ori_win_right", static_cast<int>(ori_win_right));
    op.SetAttrStr("layout_q", std::string(layout_q));
    op.SetAttrStr("layout_kv", std::string(layout_kv));
    op.SetAttrAny("ori_kv_stride0", static_cast<int>(ori_kv->stride(0)));
    op.SetAttrAny("cmp_kv_stride0", static_cast<int>(cmp_kv ? cmp_kv->stride(0) : 0));
    op.SetAttrAny("return_softmax_lse", return_softmax_lse);

    auto context = std::make_shared<TilingContext>("kv_quant_sparse_attn_sharedkv");
    at::ScalarType scalar = q.scalar_type();
    op.SetToContext(context, scalar);
    context->RegisterTensor(q, true);
    for (const auto &t : {ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices, ori_block_table,
                          cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                          seqused_q, seqused_kv, sinks, metadata})
        context->RegisterTensor(t, true);
    context->RegisterTensor(output, false);
    context->RegisterTensor(lse, false);

    KvQuantSASTilingInfo info;
    TORCH_CHECK(KvQuantSASInfoParser(context.get()).Parse(info) == ge::GRAPH_SUCCESS,
                "kv_quant_sparse_attn_sharedkv: parsing tiling inputs failed");
    TORCH_CHECK(KvQuantSASTilingCheck(info).Process() == ge::GRAPH_SUCCESS,
                "kv_quant_sparse_attn_sharedkv: tiling validation failed");
    KvQuantSparseAttnSharedkvTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS,
                "kv_quant_sparse_attn_sharedkv: tiling failed");
    auto tiling_tensor = context->GetTilingTensor(tiling.GetTilingData());
    auto workspace = at::empty({static_cast<int64_t>(tiling.GetWorkspaceSize())}, q.options().dtype(at::kByte));

    auto int_placeholder = Placeholder(q, at::kInt);
    auto q_placeholder = Placeholder(q, q.scalar_type());
    auto fp8_placeholder = Placeholder(q, at::kFloat8_e4m3fn);
    EXEC_KERNEL_CMD(kv_quant_sparse_attn_sharedkv, tiling.GetBlockDim(), q, *ori_kv,
                    cmp_kv.value_or(fp8_placeholder), ori_sparse_indices.value_or(int_placeholder),
                    cmp_sparse_indices.value_or(int_placeholder), ori_block_table.value_or(int_placeholder),
                    cmp_block_table.value_or(int_placeholder), cu_seqlens_q.value_or(int_placeholder),
                    cu_seqlens_ori_kv.value_or(int_placeholder), cu_seqlens_cmp_kv.value_or(int_placeholder),
                    seqused_q.value_or(int_placeholder), seqused_kv.value_or(int_placeholder), *sinks, *metadata,
                    output, lse, workspace, tiling_tensor);
    return {output, lse};
}
}  // namespace sglang::npu_kernel
