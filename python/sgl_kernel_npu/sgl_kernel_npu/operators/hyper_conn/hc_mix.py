"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_mix_triton_fn = None
_hc_mix_torch_fn = None


@register_impl(
    "hyper_conn.hc_mix",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "hyper_conn.hc_mix",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_mix_triton(x, down, up, hc, hs):
    global _hc_mix_triton_fn
    if _hc_mix_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import mix

        _hc_mix_triton_fn = mix
    return _hc_mix_triton_fn(x, down, up, hc, hs)


@register_impl(
    "hyper_conn.hc_mix",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_mix_torch(x, down, up, hc, hs):
    global _hc_mix_torch_fn
    if _hc_mix_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import mix

        _hc_mix_torch_fn = mix
    return _hc_mix_torch_fn(x, down, up, hc, hs)


@register_operator("hyper_conn.hc_mix")
def hc_mix(x, down, up, hc, hs, *, impl="auto"):
    """Mix four groups using learned sigmoid gates.

    x is BF16 [R,10240], down is BF16 [320,10240], and up is BF16
    [10240,320]. All tensors are contiguous on the same NPU. hc and hs
    are integers fixed at 4 and 2560. Compute h = SiLU(x @ down.T / 4),
    then average x * sigmoid(h @ up.T) across the four groups in FP32.
    Return a new contiguous BF16 [R,2560] tensor without modifying inputs.
    impl defaults to auto, selecting the Triton/native hybrid.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("hyper_conn.hc_mix", impl)(x, down, up, hc, hs)
