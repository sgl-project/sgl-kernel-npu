"""Public speculative convolution state scatter."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def _check_inputs(dst, src, dst_indices, src_indices, step_indices):
    if dst.ndim < 3 or src.ndim != dst.ndim + 1:
        raise ValueError(
            f"invalid state ranks: dst.ndim={dst.ndim}, src.ndim={src.ndim}"
        )
    if dst.shape[0] != src.shape[0] or tuple(dst.shape[2:]) != tuple(src.shape[3:]):
        raise ValueError(
            "persistent and speculative state shapes do not match: "
            f"dst={tuple(dst.shape)}, src={tuple(src.shape)}"
        )
    if dst.dtype != src.dtype or dst.device != src.device:
        raise ValueError("persistent and speculative states must share dtype/device")
    if any(index.ndim != 1 for index in (dst_indices, src_indices, step_indices)):
        raise ValueError("state index tensors must be one-dimensional")

    num_requests = dst_indices.numel()
    if src_indices.numel() != num_requests or step_indices.numel() != num_requests:
        raise ValueError("state index tensors must have equal lengths")
    if num_requests == 0:
        return 0
    if any(
        index.device != dst.device for index in (dst_indices, src_indices, step_indices)
    ):
        raise ValueError("state indices must be on the cache device")

    tail_shape = tuple(dst.shape[2:])
    if not 1 <= len(tail_shape) <= 3:
        raise ValueError(f"state tail rank must be in [1, 3], got {tail_shape}")
    return num_requests


@register_impl("mamba.speculative_state_scatter", "auto")
@register_impl("mamba.speculative_state_scatter", "triton")
def _triton_impl(dst, src, dst_indices, src_indices, step_indices):
    global _triton_fn
    if not _check_inputs(dst, src, dst_indices, src_indices, step_indices):
        return dst
    if _triton_fn is None:
        from ...implementation.triton.mamba.speculative_state_scatter import (
            speculative_state_scatter,
        )

        _triton_fn = speculative_state_scatter
    return _triton_fn(dst, src, dst_indices, src_indices, step_indices)


@register_impl("mamba.speculative_state_scatter", "torch_native")
def _torch_native_impl(dst, src, dst_indices, src_indices, step_indices):
    global _torch_native_fn
    if not _check_inputs(dst, src, dst_indices, src_indices, step_indices):
        return dst
    if _torch_native_fn is None:
        from ...implementation.torch_native.mamba.speculative_state_scatter import (
            speculative_state_scatter,
        )

        _torch_native_fn = speculative_state_scatter
    return _torch_native_fn(dst, src, dst_indices, src_indices, step_indices)


@register_operator("mamba.speculative_state_scatter")
def speculative_state_scatter(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    """Commit one speculative convolution state snapshot per request across layers.

    The destination is updated in place. Indices are converted to int32; a
    negative destination, source or step skips that request.

    Args:
        dst (Tensor): Writable persistent state of shape [layers, slots, *state].
        src (Tensor): Snapshots of shape [layers, scratch, steps, *state], with
            the same dtype, device and state shape as dst. State has 1 to 3 axes.
        dst_indices (Tensor): Integer [requests] destination slots, distinct
            among active requests.
        src_indices (Tensor): Integer [requests] snapshot slots.
        step_indices (Tensor): Integer [requests] steps within each snapshot slot.
        impl (str): Backend to use. Defaults to "auto", which selects "triton".

    Returns:
        Tensor: The original dst object, including when the selection is empty.
    """
    return get_impl_by_name("mamba.speculative_state_scatter", impl)(
        dst, src, dst_indices, src_indices, step_indices
    )
