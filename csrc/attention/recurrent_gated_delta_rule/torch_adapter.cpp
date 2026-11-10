// Torch bridge for the independent RGDR CANN operator. Legacy direct launch is retained.
#include "aclnn_runtime.h"
#include "aclnn_sgl_recurrent_gated_delta_rule.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <limits>
#include <vector>

namespace {
bool present(const c10::optional<at::Tensor> &value)
{
    return value.has_value() && value->defined();
}

// The legacy required tensors are read from data_ptr as dense physical spans.
// A view preserves that interpretation and storage offset without copying back
// a new logical-stride interpretation into the caller's mutable state.
at::Tensor raw_dense(const at::Tensor &tensor)
{
    std::vector<int64_t> strides(tensor.dim());
    int64_t stride = 1;
    for (int64_t i = tensor.dim(); i-- > 0;) {
        strides[i] = stride;
        TORCH_CHECK(tensor.size(i) == 0 || stride <= std::numeric_limits<int64_t>::max() / tensor.size(i),
                    "RGDR dense span is too large");
        stride *= tensor.size(i);
    }
    return tensor.as_strided(tensor.sizes(), strides);
}

at::Tensor recurrent_gated_delta_rule(const at::Tensor &mix_qkv, const at::Tensor &recurrent_state,
                                      const at::Tensor &beta, double scale, const at::Tensor &actual_seq_lengths,
                                      const at::Tensor &ssm_state_indices, int64_t nk, int64_t nv,
                                      const c10::optional<at::Tensor> &intermediate_state,
                                      const c10::optional<at::Tensor> &cache_indices,
                                      const c10::optional<at::Tensor> &num_accepted_tokens,
                                      const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk)
{
    TORCH_CHECK(mix_qkv.dim() == 3, "MixQKV must be 3-dimensional (B, S, D)");
    TORCH_CHECK(recurrent_state.dim() == 4, "State must be 4-dimensional (N, nv, dv, dk)");
    TORCH_CHECK(beta.dim() == 3, "Beta must be 3-dimensional (B, S, nv)");
    TORCH_CHECK(mix_qkv.device().type() == c10::DeviceType::PrivateUse1, "mix_qkv must be on an NPU");
    for (const auto &tensor : {mix_qkv, recurrent_state, beta}) {
        TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "RGDR mix_qkv, beta and recurrent_state must be bfloat16");
        TORCH_CHECK(tensor.device() == mix_qkv.device(), "RGDR tensors must be on the same NPU");
    }
    for (const auto &tensor : {actual_seq_lengths, ssm_state_indices}) {
        TORCH_CHECK(tensor.device() == mix_qkv.device(), "RGDR indices must be on the same NPU");
        // Unlike optional indices, the old Host never casts these required ports.
        TORCH_CHECK(tensor.scalar_type() == at::kInt, "actual_seq_lengths and ssm_state_indices must be int32");
    }
    for (const auto &tensor : {intermediate_state, cache_indices, num_accepted_tokens, g, gk}) {
        if (present(tensor))
            TORCH_CHECK(tensor->device() == mix_qkv.device(), "RGDR optional inputs must be on the same NPU");
    }
    const c10::DeviceGuard deviceGuard(mix_qkv.device());
    const int64_t b = mix_qkv.size(0), s = mix_qkv.size(1);
    const int64_t dv = recurrent_state.size(2), dk = recurrent_state.size(3);
    constexpr int64_t LIMIT = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(b > 0 && s > 0 && s <= 8 && nk > 0 && nv > 0 && dv > 0 && dk > 0 && nk <= LIMIT && nv <= LIMIT &&
                    dk <= LIMIT / 16 && dv <= LIMIT / 16 && b <= LIMIT / s && b <= LIMIT / nv && nv % nk == 0,
                "RGDR requires positive representable dimensions, S <= 8 and nv divisible by nk");
    // DataCopyPad's gate block lengths are uint16 byte counts, even though the
    // later vector operations consume the full S*nv elements.
    constexpr int64_t COPY_LIMIT = std::numeric_limits<uint16_t>::max();
    TORCH_CHECK(nv <= COPY_LIMIT / sizeof(c10::BFloat16) / s,
                "RGDR beta copy requires S*nv*sizeof(bfloat16) <= 65535 bytes");
    TORCH_CHECK(!present(g) || nv <= COPY_LIMIT / sizeof(float) / s,
                "RGDR g copy requires S*nv*sizeof(float) <= 65535 bytes");
    TORCH_CHECK(nk <= (LIMIT / 2) / (2 * dk) && nv <= (LIMIT / 2 - 2 * nk * dk) / dv,
                "RGDR packed width exceeds the kernel index range");
    TORCH_CHECK(mix_qkv.size(2) == 2 * nk * dk + nv * dv, "mix_qkv width must equal 2*nk*dk + nv*dv");
    TORCH_CHECK(recurrent_state.size(0) > 0 && recurrent_state.size(1) == nv, "recurrent_state must be (N,nv,dv,dk)");
    TORCH_CHECK(beta.size(0) == b && beta.size(1) == s && beta.size(2) == nv, "beta must be (B,S,nv)");
    TORCH_CHECK(actual_seq_lengths.dim() == 1 && actual_seq_lengths.size(0) == b, "actual_seq_lengths must be (B,)");
    TORCH_CHECK(ssm_state_indices.dim() == 2 && ssm_state_indices.size(0) == b && ssm_state_indices.size(1) == s,
                "ssm_state_indices must be (B,S)");

    auto mix = raw_dense(mix_qkv), recurrent = raw_dense(recurrent_state), betaTensor = raw_dense(beta);
    auto lengths = raw_dense(actual_seq_lengths), indices = raw_dense(ssm_state_indices);
    const bool intermediate = present(intermediate_state), hasGamma = present(g);
    at::Tensor state = recurrent, seed, cache, accepted, gamma, keyGate;
    if (intermediate) {
        TORCH_CHECK(intermediate_state->scalar_type() == at::kBFloat16, "intermediate_state must be bfloat16");
        TORCH_CHECK(intermediate_state->numel() > 0 && intermediate_state->numel() % (nv * dv * dk) == 0,
                    "intermediate_state must contain complete (nv,dv,dk) slots");
        // Preserve the old temporary-without-copyback behavior for noncontiguous views.
        state = intermediate_state->contiguous().view({-1, nv, dv, dk});
        seed = recurrent;
        TORCH_CHECK(present(cache_indices) && cache_indices->numel() >= b,
                    "cache_indices must provide one recurrent seed slot per sequence in MTP mode");
    }
    if (present(cache_indices)) cache = cache_indices->to(at::kInt).contiguous();
    if (present(num_accepted_tokens)) {
        TORCH_CHECK(num_accepted_tokens->numel() >= b, "num_accepted_tokens must contain B values");
        accepted = num_accepted_tokens->to(at::kInt).contiguous();
    }
    if (hasGamma) {
        TORCH_CHECK(g->numel() >= b * s * nv, "g must provide B*S*nv log-decay values");
        gamma = g->to(at::kFloat).contiguous();
    } else {
        // Keep the old absent-g allocation, but pass a null optional port so the
        // tiler does not enable exp(g). Missing g means decay 1, not exp(1).
        gamma = at::ones({b * s, nv}, mix_qkv.options().dtype(at::kFloat));
    }
    if (present(gk)) keyGate = gk->to(at::kFloat).contiguous();  // Compatibility port, unused by RGDR.
    auto out = at::empty({b, s, nv, dv}, mix_qkv.options());

    static auto queryWorkspace = reinterpret_cast<decltype(&aclnnSglRecurrentGatedDeltaRuleGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglRecurrentGatedDeltaRuleGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglRecurrentGatedDeltaRule)>(
        sgl_cann::op_api_symbol("aclnnSglRecurrentGatedDeltaRule"));
    auto mixDesc = sgl_cann::tensor_descriptor(mix, ACL_BF16);
    auto stateDesc = sgl_cann::tensor_descriptor(state, ACL_BF16);
    auto betaDesc = sgl_cann::tensor_descriptor(betaTensor, ACL_BF16);
    auto lengthsDesc = sgl_cann::tensor_descriptor(lengths, ACL_INT32);
    auto indicesDesc = sgl_cann::tensor_descriptor(indices, ACL_INT32);
    auto seedDesc = intermediate ? sgl_cann::tensor_descriptor(seed, ACL_BF16) : std::shared_ptr<aclTensor>{};
    auto cacheDesc = cache.defined() ? sgl_cann::tensor_descriptor(cache, ACL_INT32) : std::shared_ptr<aclTensor>{};
    auto acceptedDesc =
        accepted.defined() ? sgl_cann::tensor_descriptor(accepted, ACL_INT32) : std::shared_ptr<aclTensor>{};
    auto gammaDesc = hasGamma ? sgl_cann::tensor_descriptor(gamma, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto keyGateDesc =
        keyGate.defined() ? sgl_cann::tensor_descriptor(keyGate, ACL_FLOAT) : std::shared_ptr<aclTensor>{};
    auto outDesc = sgl_cann::tensor_descriptor(out, ACL_BF16);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // The required state Input/Output is one Ref argument in the generated API.
    const auto status =
        queryWorkspace(mixDesc.get(), stateDesc.get(), betaDesc.get(), lengthsDesc.get(), indicesDesc.get(),
                       seedDesc.get(), cacheDesc.get(), acceptedDesc.get(), gammaDesc.get(), keyGateDesc.get(),
                       static_cast<float>(scale), nk, nv, outDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "RGDR workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "RGDR workspace too large");
    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, mix_qkv.options().dtype(at::kByte));
    std::vector<at::Tensor> tensors{mix, state, recurrent, betaTensor, lengths, indices, gamma, out, workspace};
    for (const auto &tensor : {seed, cache, accepted, keyGate})
        if (tensor.defined()) tensors.push_back(tensor);
    std::vector<std::shared_ptr<aclTensor>> descriptors{mixDesc,     stateDesc,   betaDesc,  lengthsDesc,
                                                        indicesDesc, seedDesc,    cacheDesc, acceptedDesc,
                                                        gammaDesc,   keyGateDesc, outDesc};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : tensors)
        tensor.record_stream(stream);
    auto aclStream = stream.stream(false);
    auto launch = [tensors, descriptors, workspace, workspaceSize, executor, aclStream]() -> int {
        const auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "RGDR ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglRecurrentGatedDeltaRule", launch);
    return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "recurrent_gated_delta_rule(Tensor mix_qkv, Tensor(a!) recurrent_state, Tensor beta, "
        "float scale, Tensor actual_seq_lengths, Tensor ssm_state_indices, int nk, int nv, "
        "Tensor(b!)? intermediate_state=None, Tensor? cache_indices=None, Tensor? num_accepted_tokens=None, "
        "Tensor? g=None, Tensor? gk=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("recurrent_gated_delta_rule", &recurrent_gated_delta_rule);
}
