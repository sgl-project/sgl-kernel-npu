"""Public fused_variance with implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("norm.fused_variance", "auto")
@register_impl("norm.fused_variance", "triton")
def _triton_impl(x: torch.Tensor):
    from ...implementation.triton.norm.rmsnorm_split import fused_variance_kernel
    from ...utils.triton_utils import get_device_properties

    if x.ndim != 3 or x.shape[-1] == 0:
        raise ValueError("x must have shape [batch, length, positive hidden]")
    x = x.contiguous()
    _, kernel_num = get_device_properties()
    B, L, C = x.shape[0], x.shape[1], x.shape[2]
    grid = (kernel_num,)

    output = torch.empty((B, L, 1), device=x.device, dtype=x.dtype)

    fused_variance_kernel[grid](x, output, B * L, C, kernel_num=kernel_num)
    return output


@register_impl("norm.fused_variance", "torch_native")
def _torch_native_impl(x: torch.Tensor):
    from ...implementation.torch_native.norm.fused_variance import fused_variance

    if x.ndim != 3 or x.shape[-1] == 0:
        raise ValueError("x must have shape [batch, length, positive hidden]")
    return fused_variance(x)


@register_operator("norm.fused_variance")
def fused_variance(x: torch.Tensor, *, impl="auto"):
    """Use the original Triton implementation by default, or the PyTorch reference."""
    return get_impl_by_name("norm.fused_variance", impl)(x)
