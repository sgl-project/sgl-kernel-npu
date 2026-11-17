"""Public move intermediate cache."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mamba.move_intermediate_cache", "auto")
@register_impl("mamba.move_intermediate_cache", "triton")
def _triton_impl(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mamba.mamba_state_update import (
            move_intermediate_cache,
        )

        _triton_fn = move_intermediate_cache
    return _triton_fn(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )


@register_impl("mamba.move_intermediate_cache", "torch_native")
def _torch_native_impl(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.mamba_state_update import (
            move_intermediate_cache,
        )

        _torch_native_fn = move_intermediate_cache
    return _torch_native_fn(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )


@register_operator("mamba.move_intermediate_cache")
def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=2,
    *,
    impl="auto"
):
    """Copies each selected speculative snapshot into the persistent SSM cache across all layers. Requests with a negative last step leave their destination untouched. Within each head this interface copies physical contiguous V*K storage; use move_intermediate_cache_kda for a logically transposed destination.

    Args:
        ssm_states (Tensor): Writable floating [layers,pool,H,V,K] persistent state. Default: Required.
        intermediate_state_cache (Tensor): Floating [layers,pool,steps,H,V,K], contiguous inside H/V/K. Default: Required.
        dst_indices_tensor (Tensor): Integer [requests] destination slots, distinct for concurrent requests. Default: Required.
        src_indices_tensor (Tensor): Integer [requests] source slots. Default: Required.
        last_steps_tensor (Tensor): Integer [requests] snapshot step; negative skips the request. Default: Required.
        h_block_size (int): Head tiling size; does not change mathematical results. Default: 2.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        The original ssm_states tensor, updated in place.
    """
    return get_impl_by_name("mamba.move_intermediate_cache", impl)(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )
