"""Public operators for the QSA/PLE/HC model contracts."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_qsa_sparse_attention_triton_fn = None
_qsa_sparse_attention_torch_fn = None


@register_impl(
    "attention.qsa_sparse_attention", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "attention.qsa_sparse_attention",
    "triton",
    supported_generations={Generation.ARCH22},
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
    supported_generations={Generation.ARCH22},
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
    """Sparse GQA over physical slots[R,S]. BF16 Q[R,Hq,256], K/V[N,Hkv,256]; local heads 24:2,12:1,6:1,3:1. Returns contiguous BF16 shaped as Q; all-padding rows are zero. auto selects the Triton/native hybrid; zero scale retains the legacy default 1/16.

    torch_native is an independent reference; it may synchronize and is not
    a graph fallback. No implementation is selected by catching errors.
    """
    return get_impl_by_name("attention.qsa_sparse_attention", impl)(
        q, k, v, slots, softmax_scale
    )
