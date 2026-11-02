"""DSV4 indexer operators with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("dsv4.quant_lightning_indexer_metadata", "auto")
@register_impl("dsv4.quant_lightning_indexer_metadata", "cann_ops_transformer")
def _cann_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.cann_ops_transformer.dsv4 import (
        quant_lightning_indexer_metadata,
    )

    return quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_impl("dsv4.quant_lightning_indexer_metadata", "torch_native")
def _torch_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.dsv4.indexer import (
        quant_lightning_indexer_metadata,
    )

    return quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_operator("dsv4.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl="auto", **kwargs
):
    """Build quantized C4 indexer metadata.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.quant_lightning_indexer_metadata", impl)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_impl("dsv4.quant_lightning_indexer", "auto")
@register_impl("dsv4.quant_lightning_indexer", "cann_ops_transformer")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.cann_ops_transformer.dsv4 import quant_lightning_indexer

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_impl("dsv4.quant_lightning_indexer", "torch_native")
def _torch_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.dsv4.indexer import quant_lightning_indexer

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_operator("dsv4.quant_lightning_indexer")
def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl="auto", **kwargs
):
    """Return logical compressed-KV top-k indices and optional values.

    impl: auto (released CANN), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("dsv4.quant_lightning_indexer", impl)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
