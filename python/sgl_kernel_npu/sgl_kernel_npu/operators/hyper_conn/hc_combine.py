"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_hc_combine_triton_fn = None
_hc_combine_torch_fn = None


@register_impl(
    "hyper_conn.hc_combine",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "hyper_conn.hc_combine",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_combine_triton(block, residual, normed, weight, hc, hs):
    global _hc_combine_triton_fn
    if _hc_combine_triton_fn is None:
        from ...implementation.triton.hyper_conn.hc import combine

        _hc_combine_triton_fn = combine
    return _hc_combine_triton_fn(block, residual, normed, weight, hc, hs)


@register_impl(
    "hyper_conn.hc_combine",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _hc_combine_torch(block, residual, normed, weight, hc, hs):
    global _hc_combine_torch_fn
    if _hc_combine_torch_fn is None:
        from ...implementation.torch_native.hyper_conn.hc import combine

        _hc_combine_torch_fn = combine
    return _hc_combine_torch_fn(block, residual, normed, weight, hc, hs)


@register_operator("hyper_conn.hc_combine")
def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    """Add a gated block output to four residual groups.

    block is BF16 [R,2560]; residual and normed are BF16 [R,10240]; weight
    is BF16 [4,10240]. All tensors are contiguous on the same NPU.
    hc and hs are fixed at 4 and 2560. Let gate = 2*sigmoid(normed@weight.T/4).
    Each residual group receives the same block with its own gate:
        out[r, h*2560+d] = residual[r, h*2560+d] + gate[r, h]*block[r, d]
    Return a new BF16 [R,10240] tensor. Inputs are unchanged.
    impl defaults to auto, selecting the Triton/native hybrid.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("hyper_conn.hc_combine", impl)(
        block, residual, normed, weight, hc, hs
    )
