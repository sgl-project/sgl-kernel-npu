"""Public move intermediate cache kda."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mamba.move_intermediate_cache_kda", "auto")
@register_impl("mamba.move_intermediate_cache_kda", "triton")
def _triton_impl(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mamba.mamba_state_update import (
            move_intermediate_cache_kda,
        )

        _triton_fn = move_intermediate_cache_kda
    return _triton_fn(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )


@register_impl("mamba.move_intermediate_cache_kda", "torch_native")
def _torch_native_impl(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.mamba_state_update import (
            move_intermediate_cache_kda,
        )

        _torch_native_fn = move_intermediate_cache_kda
    return _torch_native_fn(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )


@register_operator("mamba.move_intermediate_cache_kda")
def move_intermediate_cache_kda(
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
    *,
    impl="auto"
):
    """Commits accepted KDA snapshots to persistent state across all layers, honoring the destination head, V and K strides. Negative last-step entries are skipped; an empty selection returns the original destination unchanged.

    Args:
        ssm_states (Tensor): Writable floating [layers,pool,H,V,K] persistent state. Default: Required.
        intermediate_state_cache (Tensor): Floating [layers,pool,steps,H,V,K], contiguous inside H/V/K. Default: Required.
        dst_indices_tensor (Tensor): Integer [requests] destination slots, distinct for concurrent requests. Default: Required.
        src_indices_tensor (Tensor): Integer [requests] source slots. Default: Required.
        last_steps_tensor (Tensor): Integer [requests] snapshot step; negative skips the request. Default: Required.
        h_block_size (int): Head tiling size; does not change mathematical results. Default: 1.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        The original ssm_states tensor, updated in place.
    """
    return get_impl_by_name("mamba.move_intermediate_cache_kda", impl)(
        ssm_states,
        intermediate_state_cache,
        dst_indices_tensor,
        src_indices_tensor,
        last_steps_tensor,
        h_block_size,
    )
