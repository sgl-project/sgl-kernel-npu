import torch
import triton
import triton.language as tl


@triton.jit
def experts_compute_identity_kernel(
    expert_indices_ptr,
    expert_scales_ptr,
    hidden_states_ptr,
    zero_result_ptr,
    num_experts: tl.constexpr,
    D: tl.constexpr,
    NBD: tl.constexpr,
    BD: tl.constexpr,
    K: tl.constexpr,
    identity_mask_value: tl.constexpr,
):
    i_s = tl.program_id(0)

    idx_ptr = expert_indices_ptr + i_s * K + tl.arange(0, K)
    sca_ptr = expert_scales_ptr + i_s * K + tl.arange(0, K)

    idx = tl.load(idx_ptr)
    mask = idx >= num_experts
    scales = tl.load(sca_ptr, mask=mask, other=0.0)
    sum_scales = tl.sum(scales, axis=0)
    zero_tensor = tl.zeros([K], dtype=scales.dtype)
    identity_tensor = tl.full([K], identity_mask_value, dtype=tl.int32)
    mask_sum = tl.sum(~mask, axis=None)
    if mask_sum == 0:
        is_first_mask = tl.arange(0, K) == 0
        identity_tensor = tl.where(is_first_mask, 0, identity_tensor)

    for i_d in range(NBD):
        hid_ptr = hidden_states_ptr + i_s * D + i_d * BD + tl.arange(0, BD)
        res_ptr = zero_result_ptr + i_s * D + i_d * BD + tl.arange(0, BD)

        hid = tl.load(hid_ptr)

        hid *= sum_scales

        tl.store(res_ptr, hid)

    tl.store(sca_ptr, zero_tensor, mask=mask)
    tl.store(idx_ptr, identity_tensor, mask=mask)


def zero_experts_compute_identity_triton(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,  # should be "identity"
    hidden_states,
    identity_mask_value=0,
):
    S, D = hidden_states.shape
    K = expert_indices.shape[1]
    BD = D
    NBD = D // BD

    zero_result = torch.empty_like(hidden_states)

    grid = [S]
    experts_compute_identity_kernel[grid](
        expert_indices,
        expert_scales,
        hidden_states,
        zero_result,
        num_experts,
        D,
        NBD,
        BD,
        K,
        identity_mask_value,
    )
    return zero_result
