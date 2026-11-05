"""Public quant_lightning_indexer_metadata operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_indexer_ops,
)
from .attention_utils import _legacy_indexer_kwargs


@register_impl("attention.quant_lightning_indexer_metadata", "auto")
def _auto_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    return get_impl_by_name(
        "attention.quant_lightning_indexer_metadata", cann_backend()
    )(num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs)


@register_impl("attention.quant_lightning_indexer_metadata", "cann_ops_transformer")
def _cann_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    op = require_indexer_ops("cann_ops_transformer")[0]
    return op(num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs)


@register_impl("attention.quant_lightning_indexer_metadata", "cann_9_1")
def _legacy_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    op = require_indexer_ops("cann_9_1")[0]
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "Legacy indexer metadata supports the migrated FP8/INT8 modes only"
        )
    args = _legacy_indexer_kwargs(kwargs)
    args.update(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        sparse_count=topk,
        device=str(args["actual_seq_lengths_query"].device),
    )
    return op(**args)


@register_impl("attention.quant_lightning_indexer_metadata", "torch_native")
def _torch_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.attention.metadata import (
        quant_lightning_indexer_metadata,
    )

    return quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_operator("attention.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl="auto", **kwargs
):
    """Build quantized C4 indexer metadata.

    impl: auto (SDK-selected), cann_9_1, cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.quant_lightning_indexer_metadata", impl)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )
