"""Public quant_lightning_indexer operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.quant_lightning_indexer", "auto")
@register_impl("attention.quant_lightning_indexer", "cann_ops_transformer")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.cann_ops_transformer.attention.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_impl("attention.quant_lightning_indexer", "torch_native")
def _torch_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.attention.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_operator("attention.quant_lightning_indexer")
def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl="auto", **kwargs
):
    """Return logical compressed-KV top-k indices and optional values.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.quant_lightning_indexer", impl)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
