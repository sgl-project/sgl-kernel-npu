"""Public grouped Gemma RMSNorm with independent backend implementations."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_grouped_gemma_rmsnorm_triton_fn = None
_grouped_gemma_rmsnorm_torch_fn = None


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _grouped_gemma_rmsnorm_triton(x, weight, group_size, eps):
    global _grouped_gemma_rmsnorm_triton_fn
    if _grouped_gemma_rmsnorm_triton_fn is None:
        from ...implementation.triton.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _grouped_gemma_rmsnorm_triton_fn = grouped_gemma_rmsnorm
    return _grouped_gemma_rmsnorm_triton_fn(x, weight, group_size, eps)


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _grouped_gemma_rmsnorm_torch(x, weight, group_size, eps):
    global _grouped_gemma_rmsnorm_torch_fn
    if _grouped_gemma_rmsnorm_torch_fn is None:
        from ...implementation.torch_native.norm.grouped_gemma_rmsnorm import (
            grouped_gemma_rmsnorm,
        )

        _grouped_gemma_rmsnorm_torch_fn = grouped_gemma_rmsnorm
    return _grouped_gemma_rmsnorm_torch_fn(x, weight, group_size, eps)


@register_operator("norm.grouped_gemma_rmsnorm")
def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    """Normalize four independent groups and scale by FP32 (1 + weight).

    x is contiguous BF16 [R,10240] and weight is contiguous BF16 [10240],
    both on the same NPU. group_size is 2560; eps is a finite positive float.
    Return a new BF16 tensor shaped as x, without modifying either input.
    auto selects the Triton implementation, which reuses the existing norm kernel.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("norm.grouped_gemma_rmsnorm", impl)(
        x, weight, group_size, eps
    )
