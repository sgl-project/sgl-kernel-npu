// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Framework adaptation of csrc/mega_chunk_gdn; PTO computation is unchanged.

#include "aclnn_runtime.h"
#include "aclnn_sgl_mega_chunk_gdn.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
namespace {
constexpr int64_t kHeadDim = 128;

bool is_supported_head_pair(int64_t value_heads, int64_t key_heads)
{
    return value_heads % key_heads == 0 &&
           (value_heads == 8 || value_heads == 12 || value_heads == 16 || value_heads == 24 || value_heads == 32 ||
            value_heads == 48 || value_heads == 64);
}

void check_shape(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                 const at::Tensor &beta, const at::Tensor &cu_seqlens, const at::Tensor &initial_state,
                 bool has_initial_state)
{
    const char *err = "Try running the alternative backend by setting GDN_ATTN_BACKEND_TRITON=1.";
    auto check = [&err](bool condition, const char *message) { TORCH_CHECK(condition, message, " ", err); };

    check(q.dim() == 4, "q must have shape [B, T, Hg, D]");
    check(k.dim() == 4, "k must have shape [B, T, Hg, D]");
    check(v.dim() == 4, "v must have shape [B, T, H, D]");
    check(g.dim() == 3, "g must have shape [B, T, H]");
    check(beta.dim() == 3, "beta must have shape [B, T, H]");

    check(q.size(0) == 1, "mega_chunk_gdn currently supports packed B=1 input");
    check(q.sizes() == k.sizes(), "q and k must have the same shape");
    check(q.size(1) == v.size(1), "q/k and v sequence lengths must match");
    check(is_supported_head_pair(v.size(2), q.size(2)), "unsupported mega_chunk_gdn (NumValueHeads, NumKeyHeads) pair");
    check(q.size(3) == kHeadDim && v.size(3) == kHeadDim, "mega_chunk_gdn supports head dimension 128");

    check(g.size(0) == 1 && beta.size(0) == 1, "g and beta must use packed B=1 layout");
    check(g.size(1) == q.size(1) && beta.size(1) == q.size(1), "g/beta sequence lengths must match q");
    check(g.size(2) == v.size(2) && beta.size(2) == v.size(2), "g/beta must use the same NumValueHeads as v");

    check(q.scalar_type() == at::kHalf, "q must be float16");
    check(k.scalar_type() == at::kHalf, "k must be float16");
    check(v.scalar_type() == at::kHalf, "v must be float16");
    check(beta.scalar_type() == at::kHalf, "beta must be float16");
    check(g.scalar_type() == at::kFloat, "g must be float32");
    check(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");

    check(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q, k, and v must be contiguous");
    check(g.is_contiguous() && beta.is_contiguous() && cu_seqlens.is_contiguous(),
          "g, beta, and cu_seqlens must be contiguous");

    if (has_initial_state) {
        check(initial_state.dim() == 4, "initial_state must have shape [N, H, D, D]");
        check(initial_state.size(0) == cu_seqlens.numel() - 1, "initial_state.size(0) must match cu_seqlens sequences");
        check(initial_state.size(1) == v.size(2), "initial_state.size(1) must match NumValueHeads");
        check(initial_state.size(2) == kHeadDim && initial_state.size(3) == kHeadDim,
              "initial_state must use head dimensions 128 x 128");
        check(initial_state.scalar_type() == at::kHalf, "initial_state must be float16");
        check(initial_state.is_contiguous(), "initial_state must be contiguous");
    }
}
}  // namespace

void mega_chunk_gdn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &g,
                    const at::Tensor &beta, const at::Tensor &mask_lower, const at::Tensor &mask_full,
                    const at::Tensor &minus_identity, const at::Tensor &cu_seqlens, at::Tensor &out, at::Tensor &g_sum,
                    at::Tensor &g_t, at::Tensor &beta_t, at::Tensor &a, at::Tensor &a_inv_f32, at::Tensor &a_inv,
                    at::Tensor &w, at::Tensor &u, at::Tensor &s, at::Tensor &v_new, at::Tensor &final_state,
                    const at::Tensor &initial_state, bool has_initial_state, at::Tensor &kkt_workspace,
                    at::Tensor &wy_workspace_a1, at::Tensor &wy_workspace_a2, at::Tensor &h_workspace,
                    at::Tensor &o_workspace_qk, at::Tensor &o_workspace_qs, at::Tensor &o_workspace_gated,
                    int64_t block_dim, int64_t batch_size, int64_t seq_len, int64_t total_tokens, int64_t num_matrices)
{
    check_shape(q, k, v, g, beta, cu_seqlens, initial_state, has_initial_state);
    TORCH_CHECK(block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(), "block_dim is out of uint32 range");
    TORCH_CHECK(batch_size == cu_seqlens.numel() - 1, "batch_size must match cu_seqlens");
    TORCH_CHECK(seq_len == q.size(1), "seq_len must match q.shape[1]");
    TORCH_CHECK(total_tokens == q.size(1), "total_tokens must match packed token count");
    TORCH_CHECK(num_matrices >= 0 && num_matrices <= std::numeric_limits<uint32_t>::max(),
                "num_matrices is out of uint32 range");
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "mega_chunk_gdn requires NPU tensors");
    const c10::DeviceGuard device_guard(q.device());
    std::vector<at::Tensor> tensors{q,
                                    k,
                                    v,
                                    g,
                                    beta,
                                    mask_lower,
                                    mask_full,
                                    minus_identity,
                                    cu_seqlens,
                                    out,
                                    g_sum,
                                    g_t,
                                    beta_t,
                                    a,
                                    a_inv_f32,
                                    a_inv,
                                    w,
                                    u,
                                    s,
                                    v_new,
                                    final_state,
                                    initial_state,
                                    kkt_workspace,
                                    wy_workspace_a1,
                                    wy_workspace_a2,
                                    h_workspace,
                                    o_workspace_qk,
                                    o_workspace_qs,
                                    o_workspace_gated};
    for (const auto &tensor : tensors) {
        TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                    "mega buffers must be contiguous on q's NPU");
    }
    std::vector<std::shared_ptr<aclTensor>> descriptors;
    descriptors.push_back(sgl_cann::tensor_descriptor(q, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(k, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(v, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(g, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(beta, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(mask_lower, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(mask_full, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(minus_identity, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(cu_seqlens, ACL_INT32));
    descriptors.push_back(sgl_cann::tensor_descriptor(out, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(g_sum, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(g_t, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(beta_t, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(a, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(a_inv_f32, ACL_FLOAT));
    descriptors.push_back(sgl_cann::tensor_descriptor(a_inv, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(w, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(u, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(s, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(v_new, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(final_state, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(initial_state, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(kkt_workspace, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(wy_workspace_a1, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(wy_workspace_a2, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(h_workspace, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(o_workspace_qk, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(o_workspace_qs, ACL_FLOAT16));
    descriptors.push_back(sgl_cann::tensor_descriptor(o_workspace_gated, ACL_FLOAT16));
    static auto query = reinterpret_cast<decltype(&aclnnSglMegaChunkGdnGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglMegaChunkGdnGetWorkspaceSize"));
    static auto execute =
        reinterpret_cast<decltype(&aclnnSglMegaChunkGdn)>(sgl_cann::op_api_symbol("aclnnSglMegaChunkGdn"));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    // Same-name CANN Ref ports appear once in the generated ACLNN prototype.
    const auto status = query(
        descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(), descriptors[4].get(),
        descriptors[5].get(), descriptors[6].get(), descriptors[7].get(), descriptors[8].get(), descriptors[9].get(),
        descriptors[10].get(), descriptors[11].get(), descriptors[12].get(), descriptors[13].get(),
        descriptors[14].get(), descriptors[15].get(), descriptors[16].get(), descriptors[17].get(),
        descriptors[18].get(), descriptors[19].get(), descriptors[20].get(), descriptors[21].get(),
        descriptors[22].get(), descriptors[23].get(), descriptors[24].get(), descriptors[25].get(),
        descriptors[26].get(), descriptors[27].get(), descriptors[28].get(), has_initial_state, block_dim, batch_size,
        seq_len, total_tokens, num_matrices, &workspace_size, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "mega workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "mega workspace is too large");
    auto workspace = at::empty({static_cast<int64_t>(workspace_size)}, q.options().dtype(at::kByte));
    tensors.push_back(workspace);
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto acl_stream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspace_size, executor, acl_stream]() -> int {
        const auto result =
            execute(workspace_size ? workspace.data_ptr() : nullptr, workspace_size, executor, acl_stream);
        TORCH_CHECK(result == ACL_SUCCESS, "mega ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglMegaChunkGdn", launch);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "mega_chunk_gdn(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
        "Tensor mask_lower, Tensor mask_full, Tensor minus_identity, Tensor cu_seqlens, "
        "Tensor(a!) out, Tensor(b!) g_sum, Tensor(c!) g_t, Tensor(d!) beta_t, "
        "Tensor(e!) A, Tensor(f!) A_inv_f32, Tensor(g!) A_inv, Tensor(h!) w, "
        "Tensor(i!) u, Tensor(j!) s, Tensor(k!) v_new, Tensor(l!) final_state, "
        "Tensor initial_state, bool has_initial_state, "
        "Tensor(m!) kkt_workspace, Tensor(n!) wy_workspace_a1, "
        "Tensor(o!) wy_workspace_a2, Tensor(p!) h_workspace, "
        "Tensor(q!) o_workspace_qk, Tensor(r!) o_workspace_qs, "
        "Tensor(s!) o_workspace_gated, int block_dim, int batch_size, "
        "int seq_len, int total_tokens, int num_matrices) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mega_chunk_gdn", &mega_chunk_gdn);
}
