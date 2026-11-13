"""Gemma RMS normalization with architecture-specific native bindings."""

import torch_npu

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("norm.gemma_rmsnorm", "auto", supported_generations={Generation.ARCH22})
@register_impl(
    "norm.gemma_rmsnorm", "torch_npu", supported_generations={Generation.ARCH22}
)
def _arch22_impl(input, weight, eps):
    return torch_npu.npu_gemma_rms_norm(input, weight, eps)


@register_impl("norm.gemma_rmsnorm", "auto", supported_generations={Generation.ARCH35})
@register_impl(
    "norm.gemma_rmsnorm", "torch_npu", supported_generations={Generation.ARCH35}
)
def _arch35_impl(input, weight, eps):
    return torch_npu.npu_rms_norm(input, 1.0 + weight, eps)


@register_impl("norm.gemma_rmsnorm", "torch_native")
def _torch_native_impl(input, weight, eps):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.norm.gemma_rmsnorm import gemma_rmsnorm

        _torch_native_fn = gemma_rmsnorm
    return _torch_native_fn(input, weight, eps)


@register_operator("norm.gemma_rmsnorm")
def gemma_rmsnorm(input, weight, eps=1e-6, *, impl="auto"):
    """Computes RMS normalization followed by the Gemma scale (1+weight), returning both normalized output and reciprocal RMS. ARCH22 uses the native Gemma operation; ARCH35 forms 1+weight in the stored weight dtype before RMSNorm, retaining that rounding boundary.

    Args:
        input (Tensor): Floating NPU [...,H] input. Default: Required.
        weight (Tensor): Convolution: [channels,width]. Gemma: [H], same dtype as input. Default: Required.
        eps (float): RMSNorm stabilization constant. Default: 1e-06.
        impl (str): auto selects torch_npu; explicit backend selection. Default: 'auto'.

    Returns:
        Output has input shape/dtype; FP32 reciprocal RMS has shape [...,1].
    """
    return get_impl_by_name("norm.gemma_rmsnorm", impl)(input, weight, eps)
