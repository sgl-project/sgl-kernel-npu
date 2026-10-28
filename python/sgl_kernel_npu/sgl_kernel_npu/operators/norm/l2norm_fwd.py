"""Public l2norm_fwd operator with parameter preparation in registered adapters."""

from typing import Optional

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("norm.l2norm_fwd", "auto")
@register_impl("norm.l2norm_fwd", "triton")
def _triton_impl(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.norm.l2norm_fwd import l2norm_fwd

        _triton_fn = l2norm_fwd

    return _triton_fn(x, eps, output_dtype)


@register_impl("norm.l2norm_fwd", "torch_native")
def _torch_native_impl(
    x: torch.Tensor, eps: float = 1e-06, output_dtype: Optional[torch.dtype] = None
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.l2norm_fwd import l2norm_fwd

        _torch_native_fn = l2norm_fwd

    shape = x.shape
    flat = x.view(-1, x.shape[-1])
    output = torch.empty_like(
        flat, dtype=x.dtype if output_dtype is None else output_dtype
    )
    _torch_native_fn(flat, output, eps)
    return output.reshape(shape)


@register_operator("norm.l2norm_fwd")
def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-06,
    output_dtype: Optional[torch.dtype] = None,
    *,
    impl="auto"
):
    """Normalize the last dimension by sqrt(sum(x**2) + eps).

    Preserve the input shape. Output dtype defaults to x.dtype; output_dtype
    overrides it. impl selects auto (Triton), triton, or torch_native.
    The two original Triton kernels remain internal width-dependent variants."""
    return get_impl_by_name("norm.l2norm_fwd", impl)(x, eps, output_dtype)
