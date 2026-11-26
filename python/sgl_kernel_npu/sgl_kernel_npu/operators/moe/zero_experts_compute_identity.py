"""Public zero experts compute identity."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("moe.zero_experts_compute_identity", "auto")
@register_impl("moe.zero_experts_compute_identity", "triton")
def _triton_impl(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.moe.zero_experts_compute_identity import (
            zero_experts_compute_identity,
        )

        _triton_fn = zero_experts_compute_identity
    return _triton_fn(
        expert_indices,
        expert_scales,
        num_experts,
        zero_expert_type,
        hidden_states,
        identity_mask_value,
    )


@register_impl("moe.zero_experts_compute_identity", "torch_native")
def _torch_native_impl(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.moe.zero_experts_compute_identity import (
            zero_experts_compute_identity,
        )

        _torch_native_fn = zero_experts_compute_identity
    return _torch_native_fn(
        expert_indices,
        expert_scales,
        num_experts,
        zero_expert_type,
        hidden_states,
        identity_mask_value,
    )


@register_operator("moe.zero_experts_compute_identity")
def zero_experts_compute_identity(
    expert_indices,
    expert_scales,
    num_experts,
    zero_expert_type,
    hidden_states,
    identity_mask_value=0,
    *,
    impl="auto"
):
    """Computes the weighted identity-expert contribution for LongCat MoE. Routes with expert ID >= num_experts contribute their weights times hidden_states, then their scales are zeroed and IDs replaced. If every route is identity, the first replacement ID is zero. Both routing tensors are modified in place.

    Args:
        expert_indices (Tensor): Writable contiguous integer [tokens,topk] routes. Default: Required.
        expert_scales (Tensor): Writable contiguous floating [tokens,topk] route weights. Default: Required.
        num_experts (int): IDs at or above this count are identity experts. Default: Required.
        zero_expert_type (str): Use identity; other strings do not select other computations. Default: Required.
        hidden_states (Tensor): Floating contiguous [tokens,hidden] states. Default: Required.
        identity_mask_value (int): Replacement route ID for identity experts. Default: 0.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [tokens,hidden] weighted identity output with hidden_states dtype; modifies routing IDs and scales.
    """
    return get_impl_by_name("moe.zero_experts_compute_identity", impl)(
        expert_indices,
        expert_scales,
        num_experts,
        zero_expert_type,
        hidden_states,
        identity_mask_value,
    )
