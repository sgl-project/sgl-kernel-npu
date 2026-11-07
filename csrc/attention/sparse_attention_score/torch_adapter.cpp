// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Preserve the legacy Torch contract; only replace direct launch with ACLNN.

#include "aclnn_runtime.h"
#include "aclnn_sgl_sparse_attention_score.h"
#include <torch/library.h>
#include <vector>

namespace {
std::shared_ptr<aclTensor> raw_descriptor(const at::Tensor &tensor, aclDataType dtype)
{
    // The old launcher passes data_ptr(), ignoring Tensor strides and storage dtype.
    // Describe that same dense raw span; do not normalize or convert its contents.
    std::vector<int64_t> strides(tensor.dim());
    int64_t stride = 1;
    for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= tensor.size(i);
    }
    auto *descriptor = aclCreateTensor(tensor.sizes().data(), tensor.dim(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                       tensor.sizes().data(), tensor.dim(), tensor.data_ptr());
    TORCH_CHECK(descriptor, "aclCreateTensor failed: ", aclGetRecentErrMsg());
    return {descriptor, [](aclTensor *value) { aclDestroyTensor(value); }};
}

at::Tensor sparse_attention_score(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                  const at::Tensor &select_idx, const at::Tensor &block_table,
                                  const c10::optional<at::Tensor> &select_num_idx,
                                  const c10::optional<at::Tensor> &q_dequant_scale,
                                  const c10::optional<at::Tensor> &k_dequant_scale,
                                  const c10::optional<at::Tensor> &v_dequant_scale,
                                  const c10::optional<at::Tensor> &actual_seq_lengths,
                                  const c10::optional<at::Tensor> &actual_seq_lengths_kv, int64_t num_key_value_heads,
                                  double scale_value, int64_t block_size, int64_t top_k, int64_t inner_precise)
{
    const auto outDtype = query.scalar_type() == at::kFloat8_e4m3fn ? at::kHalf : query.scalar_type();
    auto out = at::zeros(query.sizes(), query.options().dtype(outDtype));
    const int64_t totalQ = query.size(0), numHeads = query.size(1);
    auto lse = at::zeros({totalQ, numHeads, 1}, at::TensorOptions().dtype(at::kFloat).device(query.options().device()));

    // Match legacy None handling exactly, including its uninitialized one-element
    // placeholders. In particular None count is not changed into a null GM pointer.
    const auto devOpts = at::TensorOptions().device(query.options().device());
    const auto selectNumIdx =
        select_num_idx.has_value() ? select_num_idx.value() : at::empty({1}, devOpts.dtype(at::kInt));
    const auto actualSeqLengths =
        actual_seq_lengths.has_value() ? actual_seq_lengths.value() : at::empty({1}, devOpts.dtype(at::kInt));
    const auto actualSeqLengthsKv =
        actual_seq_lengths_kv.has_value() ? actual_seq_lengths_kv.value() : at::empty({1}, devOpts.dtype(at::kInt));
    const auto qDequantScale =
        q_dequant_scale.has_value() ? q_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));
    const auto kDequantScale =
        k_dequant_scale.has_value() ? k_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));
    const auto vDequantScale =
        v_dequant_scale.has_value() ? v_dequant_scale.value() : at::empty({1}, devOpts.dtype(at::kFloat));

    // Preserve the old Host's size() accesses, including their order and the
    // conditional key access. GERT GetDim() does not reproduce their rank errors.
    (void)query.size(0);
    (void)query.size(1);
    const auto embeddingSize = static_cast<uint32_t>(query.size(2));
    if (static_cast<uint32_t>(num_key_value_heads) == 0) {
        (void)key.size(2);
    }
    (void)block_table.size(0);
    (void)block_table.size(1);
    (void)select_idx.size(0);
    (void)select_idx.size(1);
    (void)select_idx.size(2);

    // On arch22, the old dispatch selects BF16 only for this combination and
    // otherwise falls back to its FP16 pointer interpretation, without validation.
    // This is descriptor metadata only: no Tensor cast/copy or new dtype rejection.
    const bool bf16Kernel =
        query.scalar_type() == at::kBFloat16 && embeddingSize == 128 && static_cast<uint32_t>(block_size) == 128;
    const auto kernelDtype = bf16Kernel ? ACL_BF16 : ACL_FLOAT16;
    std::vector<at::Tensor> inputs{query,
                                   key,
                                   value,
                                   select_idx,
                                   block_table,
                                   selectNumIdx,
                                   actualSeqLengths,
                                   actualSeqLengthsKv,
                                   qDequantScale,
                                   kDequantScale,
                                   vDequantScale};
    std::vector<std::shared_ptr<aclTensor>> descriptors;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto dtype = i < 3 ? kernelDtype : (i < 8 ? ACL_INT32 : ACL_FLOAT);
        descriptors.push_back(raw_descriptor(inputs[i], dtype));
    }
    const auto outDesc = raw_descriptor(out, kernelDtype);
    const auto lseDesc = raw_descriptor(lse, ACL_FLOAT);
    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglSparseAttentionScoreGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglSparseAttentionScoreGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglSparseAttentionScore)>(
        sgl_cann::op_api_symbol("aclnnSglSparseAttentionScore"));
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = queryWorkspace(descriptors[0].get(), descriptors[1].get(), descriptors[2].get(), descriptors[3].get(),
                                 descriptors[4].get(), descriptors[5].get(), descriptors[6].get(), descriptors[7].get(),
                                 descriptors[8].get(), descriptors[9].get(), descriptors[10].get(), num_key_value_heads,
                                 scale_value, block_size, top_k, inner_precise, outDesc.get(), lseDesc.get(),
                                 &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "sparse_attention_score workspace query failed: ", aclGetRecentErrMsg());
    auto workspace = at::zeros({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
    inputs.insert(inputs.end(), {out, lse, workspace});
    descriptors.insert(descriptors.end(), {outDesc, lseDesc});
    const auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : inputs) {
        if (tensor.device().type() == c10::DeviceType::PrivateUse1) tensor.record_stream(stream);
    }
    const auto aclStream = stream.stream(false);
    auto launch = [inputs, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "sparse_attention_score execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglSparseAttentionScore", launch);
    return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "npu_sparse_attention_score(Tensor query, Tensor key, Tensor value, Tensor select_idx, "
        "Tensor block_table, Tensor? select_num_idx=None, Tensor? q_dequant_scale=None, "
        "Tensor? k_dequant_scale=None, Tensor? v_dequant_scale=None, "
        "Tensor? actual_seq_lengths=None, Tensor? actual_seq_lengths_kv=None, "
        "int num_key_value_heads=1, float scale_value=1.0, int block_size=128, "
        "int top_k=16, int inner_precise=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("npu_sparse_attention_score", &sparse_attention_score);
}
