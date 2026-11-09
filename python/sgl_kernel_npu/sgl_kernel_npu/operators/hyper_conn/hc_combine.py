"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_combine_triton_fn = None
_hc_combine_torch_fn = None


@register_impl(
    "hyper_conn.hc_combine", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "hyper_conn.hc_combine", "triton", supported_generations={Generation.ARCH22}
)
def _hc_combine_triton(block, residual, normed, weight, hc, hs):
    global _hc_combine_triton_fn
    if _hc_combine_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import combine

        _hc_combine_triton_fn = combine
    return _hc_combine_triton_fn(block, residual, normed, weight, hc, hs)


@register_impl(
    "hyper_conn.hc_combine", "torch_native", supported_generations={Generation.ARCH22}
)
def _hc_combine_torch(block, residual, normed, weight, hc, hs):
    global _hc_combine_torch_fn
    if _hc_combine_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import combine

        _hc_combine_torch_fn = combine
    return _hc_combine_torch_fn(block, residual, normed, weight, hc, hs)


@register_operator("hyper_conn.hc_combine")
def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    """HC gated residual injection: contiguous BF16 block[R,2560], residual/normed[R,10240], weight[4,10240], hc=4, hs=2560. Returns BF16 residual + block*2*sigmoid(normed@weight.T/4), without modifying inputs. auto selects the Triton/native hybrid.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("hyper_conn.hc_combine", impl)(
        block, residual, normed, weight, hc, hs
    )
