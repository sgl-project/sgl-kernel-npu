"""Public conv state rollback."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mamba.conv_state_rollback", "auto")
@register_impl("mamba.conv_state_rollback", "triton")
def _triton_impl(conv_states, state_indices, step_indices, draft_token_num):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mamba.mamba_state_update import (
            conv_state_rollback,
        )

        _triton_fn = conv_state_rollback
    return _triton_fn(conv_states, state_indices, step_indices, draft_token_num)


@register_impl("mamba.conv_state_rollback", "torch_native")
def _torch_native_impl(conv_states, state_indices, step_indices, draft_token_num):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.mamba_state_update import (
            conv_state_rollback,
        )

        _torch_native_fn = conv_state_rollback
    return _torch_native_fn(conv_states, state_indices, step_indices, draft_token_num)


@register_operator("mamba.conv_state_rollback")
def conv_state_rollback(
    conv_states, state_indices, step_indices, draft_token_num, *, impl="auto"
):
    """Rolls convolution history to the right by draft_token_num-1-step for each selected request across all layers. Only the shifted suffix is overwritten; the leading part stays unchanged. Non-contiguous state views retain their original storage and strides.

    Args:
        conv_states (Tensor): Prefill: [pool,channels,width-1]; rollback: [layers,pool,window,channels]. Default: Required.
        state_indices (Tensor): Integer [requests] rollback slots, converted to INT32. Default: Required.
        step_indices (Tensor): Integer [requests] accepted-step indices, converted to INT32. Default: Required.
        draft_token_num (int): Number of draft tokens. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        Original conv_states alias; None for an empty selection.
    """
    return get_impl_by_name("mamba.conv_state_rollback", impl)(
        conv_states, state_indices, step_indices, draft_token_num
    )
