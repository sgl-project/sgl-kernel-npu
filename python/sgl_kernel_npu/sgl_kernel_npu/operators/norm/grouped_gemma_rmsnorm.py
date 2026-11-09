"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_grouped_gemma_rmsnorm_triton_fn = None
_grouped_gemma_rmsnorm_torch_fn = None


@register_impl(
    "norm.grouped_gemma_rmsnorm", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "norm.grouped_gemma_rmsnorm", "triton", supported_generations={Generation.ARCH22}
)
def _grouped_gemma_rmsnorm_triton(x, weight, group_size, eps):
    global _grouped_gemma_rmsnorm_triton_fn
    if _grouped_gemma_rmsnorm_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import grouped_norm

        _grouped_gemma_rmsnorm_triton_fn = grouped_norm
    return _grouped_gemma_rmsnorm_triton_fn(x, weight, group_size, eps)


@register_impl(
    "norm.grouped_gemma_rmsnorm",
    "torch_native",
    supported_generations={Generation.ARCH22},
)
def _grouped_gemma_rmsnorm_torch(x, weight, group_size, eps):
    global _grouped_gemma_rmsnorm_torch_fn
    if _grouped_gemma_rmsnorm_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import grouped_norm

        _grouped_gemma_rmsnorm_torch_fn = grouped_norm
    return _grouped_gemma_rmsnorm_torch_fn(x, weight, group_size, eps)


@register_operator("norm.grouped_gemma_rmsnorm")
def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    """Grouped Gemma RMSNorm for contiguous BF16 x[R,10240], weight[10240], group_size=2560 and finite positive eps. Each group has independent statistics and weights. Returns BF16 shaped as x. auto composes public grouped RMSNorm with FP32 (1+weight).

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("norm.grouped_gemma_rmsnorm", impl)(
        x, weight, group_size, eps
    )
