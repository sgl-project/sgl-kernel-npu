"""Public masked cross-layer conv_state snapshot copy for speculative Radix tracking."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mamba.copy_tracked_conv_states", "auto")
@register_impl("mamba.copy_tracked_conv_states", "triton")
def _triton_impl(conv_states, src_slots, dst_slots, track_steps):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mamba.mamba_state_update import (
            copy_tracked_conv_states,
        )

        _triton_fn = copy_tracked_conv_states
    return _triton_fn(conv_states, src_slots, dst_slots, track_steps)


@register_impl("mamba.copy_tracked_conv_states", "torch_native")
def _torch_native_impl(conv_states, src_slots, dst_slots, track_steps):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.mamba_state_update import (
            copy_tracked_conv_states,
        )

        _torch_native_fn = copy_tracked_conv_states
    return _torch_native_fn(conv_states, src_slots, dst_slots, track_steps)


@register_operator("mamba.copy_tracked_conv_states")
def copy_tracked_conv_states(
    conv_states, src_slots, dst_slots, track_steps, *, impl="auto"
):
    """Copy ``conv_states[:, src_slots[b]]`` into ``conv_states[:, dst_slots[b]]`` where ``track_steps[b] >= 0``.

    auto selects Triton. Rows with ``track_steps[b] < 0``, negative slot indices, or
    ``src_slots[b] == dst_slots[b]`` are skipped in-kernel without self-copy overhead.

    Args:
        conv_states (Tensor): State pool of shape ``[layers, pool, ...]``, updated in place.
        src_slots (Tensor): 1-D integer source slot indices ``[B]``.
        dst_slots (Tensor): 1-D integer destination slot indices ``[B]``.
        track_steps (Tensor): 1-D integer tracking step indices ``[B]`` (``< 0`` skips row).
        impl (str): ``"auto"``, ``"triton"``, or ``"torch_native"``. Default: ``"auto"``.

    Returns:
        Tensor: Original ``conv_states`` alias updated in place.
    """
    return get_impl_by_name("mamba.copy_tracked_conv_states", impl)(
        conv_states, src_slots, dst_slots, track_steps
    )
