"""Public SiTU activation with optional routed-MoE token counts."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


def _check_inputs(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor],
    group_list_type: Optional[int],
) -> None:
    """Apply shared legacy checks and reject empty expert metadata."""
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")
    if has_group_list and group_list.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"group_list dtype must be torch.int32 or torch.int64, "
            f"but got {group_list.dtype}"
        )
    if has_group_list and group_list.numel() == 0:
        raise ValueError("group_list must not be empty")


@register_impl("activation.situ_and_mul", "auto")
@register_impl("activation.situ_and_mul", "triton")
def _triton_impl(x, group_list, group_list_type, beta, linear_beta):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.activation.situ_and_mul import situ_and_mul

        _triton_fn = situ_and_mul
    return _triton_fn(x, group_list, group_list_type, beta, linear_beta)


@register_impl("activation.situ_and_mul", "torch_native")
def _torch_native_impl(x, group_list, group_list_type, beta, linear_beta):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.activation.situ_and_mul import situ_and_mul

        _torch_native_fn = situ_and_mul
    return _torch_native_fn(x, group_list, group_list_type, beta, linear_beta)


@register_operator("activation.situ_and_mul")
def situ_and_mul(
    x: torch.Tensor,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: Optional[int] = None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    *,
    impl="auto",
) -> torch.Tensor:
    """Apply SiTU to gate/up halves; auto selects Triton.

    Args:
        x: FP16, BF16 or FP32 tensor [..., 2*d], d > 0, with gate then up
            features. Triton requires an NPU for nonempty inputs and limits
            the flattened row count N to 2**31 - C, for C vector cores.
        group_list: Optional contiguous int32/int64 tensor [E], E > 0 experts,
            on x's device. Counts determine the valid flattened row prefix.
        group_list_type: 0 for cumulative counts, 1 for counts; ignored if absent.
        beta: Finite positive gate saturation bound; default 4.0.
        linear_beta: Finite positive up saturation bound; None disables it.
        impl: "auto", "triton" or "torch_native" (keyword-only).

    Returns:
        New tensor [..., d] on x's device with x's dtype. Grouped calls write
        only the valid flattened row prefix; remaining rows are unspecified.
        Intermediates use FP32; inputs are not modified.
    """
    _check_inputs(x, group_list, group_list_type)
    impl_fn = get_impl_by_name("activation.situ_and_mul", impl)
    return impl_fn(x, group_list, group_list_type, beta, linear_beta)
