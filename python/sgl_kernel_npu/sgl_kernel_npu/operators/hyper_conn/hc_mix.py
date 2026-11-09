"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_mix_triton_fn = None
_hc_mix_torch_fn = None


@register_impl("hyper_conn.hc_mix", "auto", supported_generations={Generation.ARCH22})
@register_impl("hyper_conn.hc_mix", "triton", supported_generations={Generation.ARCH22})
def _hc_mix_triton(x, down, up, hc, hs):
    global _hc_mix_triton_fn
    if _hc_mix_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import mix

        _hc_mix_triton_fn = mix
    return _hc_mix_triton_fn(x, down, up, hc, hs)


@register_impl(
    "hyper_conn.hc_mix", "torch_native", supported_generations={Generation.ARCH22}
)
def _hc_mix_torch(x, down, up, hc, hs):
    global _hc_mix_torch_fn
    if _hc_mix_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import mix

        _hc_mix_torch_fn = mix
    return _hc_mix_torch_fn(x, down, up, hc, hs)


@register_operator("hyper_conn.hc_mix")
def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    """HC sigmoid-gated mixing. Contiguous BF16 x[R,10240], down[320,10240], up[10240,320], hc=4, hs=2560. Returns BF16[R,2560], using FP32 projections and SiLU(x@down.T/4). auto selects the Triton/native hybrid.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("hyper_conn.hc_mix", impl)(x, down, up, hc, hs)
