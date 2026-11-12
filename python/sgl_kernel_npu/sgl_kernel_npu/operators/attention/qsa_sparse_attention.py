"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_sparse_attention_triton_fn = None
_qsa_sparse_attention_torch_fn = None


@register_impl(
    "attention.qsa_sparse_attention",
    "auto",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
@register_impl(
    "attention.qsa_sparse_attention",
    "triton",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_sparse_attention_triton(q, k, v, slots, softmax_scale=None):
    global _qsa_sparse_attention_triton_fn
    if _qsa_sparse_attention_triton_fn is None:
        from ...implementation.triton.attention.qsa_sparse_attention import (
            sparse_attention,
        )

        _qsa_sparse_attention_triton_fn = sparse_attention
    return _qsa_sparse_attention_triton_fn(q, k, v, slots, softmax_scale)


@register_impl(
    "attention.qsa_sparse_attention",
    "torch_native",
    supported_generations={Generation.ARCH22, Generation.ARCH35},
)
def _qsa_sparse_attention_torch(q, k, v, slots, softmax_scale=None):
    global _qsa_sparse_attention_torch_fn
    if _qsa_sparse_attention_torch_fn is None:
        from ...implementation.torch_native.attention.qsa_sparse_attention import (
            sparse_attention,
        )

        _qsa_sparse_attention_torch_fn = sparse_attention
    return _qsa_sparse_attention_torch_fn(q, k, v, slots, softmax_scale)


@register_operator("attention.qsa_sparse_attention")
def qsa_sparse_attention(q, k, v, slots, softmax_scale=None, *, impl="auto"):
    """Compute sparse grouped-query attention over physical slot indices.

    q is BF16 [R,Hq,256]; k and v are BF16 [N,Hkv,256] on the same NPU.
    Supported local head pairs are 24:2, 12:1, 6:1 and 3:1. slots is int32
    [R,S]; negative entries are padding. q/slots may have row gaps; k/v
    must be contiguous. softmax_scale is a float or None; None and zero
    select 1/16. Return contiguous BF16 shaped as q, with zero empty rows.
    impl defaults to auto, selecting the Triton/native hybrid.

    torch_native is an independent accuracy reference for use outside graph
    capture; it is never selected automatically after a failure.
    """
    return get_impl_by_name("attention.qsa_sparse_attention", impl)(
        q, k, v, slots, softmax_scale
    )
