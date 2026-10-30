// Framework bridge for the independent attention CANN operator.
// Validation and contiguous-buffer semantics follow the existing CGDR Host
// contract; computation remains in the separately copied native implementation.

#include "aclnn_runtime.h"
#include "aclnn_sgl_chunk_gated_delta_rule.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <limits>
#include <tuple>
#include <vector>

namespace {
std::tuple<at::Tensor, at::Tensor>
chunk_gated_delta_rule(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                       const c10::optional<at::Tensor> &beta, const c10::optional<at::Tensor> &initial_state,
                       const c10::optional<at::Tensor> &actual_seq_lengths, const c10::optional<double> &scale,
                       const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &chunk_state)
{
    // Validate required inputs before creating descriptors or touching the SDK.
    TORCH_CHECK(query.defined() && key.defined() && value.defined(), "query/key/value must be defined");
    TORCH_CHECK(beta.has_value() && beta->defined(), "beta must be provided");
    TORCH_CHECK(initial_state.has_value() && initial_state->defined(), "initial_state must be provided");
    TORCH_CHECK(actual_seq_lengths.has_value() && actual_seq_lengths->defined(), "actual_seq_lengths must be provided");
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "query must be on an NPU");
    const bool hasGamma = g.has_value() && g->defined();
    const bool outputChunkState = chunk_state.has_value() && chunk_state->defined();
    for (const auto &tensor : {key, value, *beta, *initial_state, *actual_seq_lengths}) {
        TORCH_CHECK(tensor.device() == query.device(), "all CGDR inputs must be on the same NPU");
    }
    if (hasGamma) TORCH_CHECK(g->device() == query.device(), "g must be on the same NPU as query");
    if (outputChunkState) {
        TORCH_CHECK(chunk_state->device() == query.device(), "chunk_state must be on the same NPU as query");
    }
    const c10::DeviceGuard deviceGuard(query.device());
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3, "query/key/value must be 3D");
    TORCH_CHECK(beta->dim() == 2, "beta must be 2D (T, Nv)");
    TORCH_CHECK(initial_state->dim() == 4, "initial_state must be 4D (B, Nv, Dv, Dk)");
    TORCH_CHECK(actual_seq_lengths->dim() == 1, "actual_seq_lengths must be 1D (B,)");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
                    value.scalar_type() == at::kBFloat16 && beta->scalar_type() == at::kBFloat16,
                "query/key/value/beta must be bfloat16");
    TORCH_CHECK(initial_state->scalar_type() == at::kBFloat16 || initial_state->scalar_type() == at::kFloat,
                "initial_state must be bfloat16 or float32");
    TORCH_CHECK(actual_seq_lengths->scalar_type() == at::kInt, "actual_seq_lengths must be int32");

    const int64_t t = query.size(0), nk = query.size(1), dk = query.size(2);
    const int64_t nv = value.size(1), dv = value.size(2), b = initial_state->size(0);
    TORCH_CHECK(t > 0 && nk > 0 && dk > 0 && nv > 0 && dv > 0 && b > 0, "dims must be positive");
    TORCH_CHECK(nk <= 64 && nv <= 64, "nk and nv must be <= 64");
    TORCH_CHECK(dk <= 128 && dv <= 128, "dk and dv must be <= 128");
    TORCH_CHECK(nv % nk == 0, "nv must be an integer multiple of nk");
    TORCH_CHECK(key.sizes() == query.sizes(), "key shape must match query");
    TORCH_CHECK(value.size(0) == t, "value shape must be (T, Nv, Dv)");
    TORCH_CHECK(beta->size(0) == t && beta->size(1) == nv, "beta shape must be (T, Nv)");
    TORCH_CHECK(initial_state->size(1) == nv && initial_state->size(2) == dv && initial_state->size(3) == dk,
                "initial_state shape must be (B, Nv, Dv, Dk)");
    TORCH_CHECK(actual_seq_lengths->size(0) == b, "actual_seq_lengths size must equal initial_state dim 0");
    if (hasGamma) {
        TORCH_CHECK(g->scalar_type() == at::kFloat && g->dim() == 2 && g->size(0) == t && g->size(1) == nv,
                    "g must be float32 (T, Nv)");
    }
    if (outputChunkState) {
        // Preserve the legacy trailing-dimension check, without adding a capacity rule.
        TORCH_CHECK(chunk_state->dim() == 4 && chunk_state->size(1) == nv && chunk_state->size(2) == dv &&
                        chunk_state->size(3) == dk,
                    "chunk_state shape must be (totalChunks, Nv, Dv, Dk)");
        TORCH_CHECK(chunk_state->scalar_type() == initial_state->scalar_type(),
                    "chunk_state dtype must match initial_state");
    }

    // Continuous work buffers preserve the old noncontiguous chunk no-copyback rule.
    auto q = query.contiguous(), k = key.contiguous(), v = value.contiguous();
    auto betaTensor = beta->contiguous(), state = initial_state->contiguous();
    auto lengths = actual_seq_lengths->contiguous();
    at::Tensor gamma, chunk;
    if (hasGamma) gamma = g->contiguous();
    if (outputChunkState) chunk = chunk_state->contiguous();
    auto out = at::empty({t, nv, dv}, query.options());
    auto finalState = at::empty({b, nv, dv, dk}, initial_state->options());
    const float scaleValue = scale.has_value() ? static_cast<float>(*scale) : 1.0f;

    // Generated prototypes are authoritative: an incompatible Ref signature must
    // fail SDK compilation. Optional same-name chunk Ref occupies one input slot.
    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglChunkGatedDeltaRuleGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglChunkGatedDeltaRuleGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglChunkGatedDeltaRule)>(
        sgl_cann::op_api_symbol("aclnnSglChunkGatedDeltaRule"));
    const auto stateDtype = state.scalar_type() == at::kFloat ? ACL_FLOAT : ACL_BF16;
    auto qDesc = sgl_cann::tensor_descriptor(q, ACL_BF16);
    auto kDesc = sgl_cann::tensor_descriptor(k, ACL_BF16);
    auto vDesc = sgl_cann::tensor_descriptor(v, ACL_BF16);
    auto betaDesc = sgl_cann::tensor_descriptor(betaTensor, ACL_BF16);
    auto stateDesc = sgl_cann::tensor_descriptor(state, stateDtype);
    auto lengthsDesc = sgl_cann::tensor_descriptor(lengths, ACL_INT32);
    auto gammaDesc = hasGamma ? sgl_cann::tensor_descriptor(gamma, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto chunkDesc = outputChunkState ? sgl_cann::tensor_descriptor(chunk, stateDtype) : std::shared_ptr<aclTensor>{};
    auto outDesc = sgl_cann::tensor_descriptor(out, ACL_BF16);
    auto finalDesc = sgl_cann::tensor_descriptor(finalState, stateDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = queryWorkspace(qDesc.get(), kDesc.get(), vDesc.get(), betaDesc.get(), stateDesc.get(),
                                       lengthsDesc.get(), gammaDesc.get(), chunkDesc.get(), scaleValue, outDesc.get(),
                                       finalDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "CGDR workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace size exceeds the Tensor allocation limit");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));

    // Captures cover the host queue; stream recording covers asynchronous use.
    std::vector<at::Tensor> tensors{q, k, v, betaTensor, state, lengths, out, finalState, workspace};
    if (hasGamma) tensors.push_back(gamma);
    if (outputChunkState) tensors.push_back(chunk);
    std::vector<std::shared_ptr<aclTensor>> descriptors{qDesc,       kDesc,     vDesc,     betaDesc, stateDesc,
                                                        lengthsDesc, gammaDesc, chunkDesc, outDesc,  finalDesc};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "CGDR ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglChunkGatedDeltaRule", launch);
    return std::make_tuple(out, finalState);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "sgl_chunk_gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor? beta=None, "
        "Tensor? initial_state=None, Tensor? actual_seq_lengths=None, float? scale=None, Tensor? g=None, "
        "Tensor(a!)? chunk_state=None) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("sgl_chunk_gated_delta_rule", &chunk_gated_delta_rule);
}
