"""Public quant_lightning_indexer_metadata operator with registered implementations."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import (
    OperatorUnavailableError,
    cann_backend,
    require_op,
)
from .indexer_utils import _cann_9_1_indexer_kwargs


@register_impl("indexer.quant_lightning_indexer_metadata", "auto")
def _auto_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    backend = cann_backend()
    if backend == "cann_9_1":
        backend = "ops_transformer_cann910"
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", backend)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_impl(
    "indexer.quant_lightning_indexer_metadata", "cann_ops_transformer_cann920"
)
def _cann_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    op = require_op(
        "cann_ops_transformer",
        "quant_lightning_indexer_metadata",
        backend="cann_ops_transformer_cann920",
    )
    return op(num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs)


@register_impl("indexer.quant_lightning_indexer_metadata", "ops_transformer_cann910")
def _cann_9_1_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    op = require_op(
        "custom", "npu_quant_lightning_indexer_metadata", backend="cann_9_1"
    )
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "cann_9_1 indexer metadata supports the migrated FP8/INT8 modes only"
        )
    args = _cann_9_1_indexer_kwargs(kwargs)
    args.update(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        sparse_count=topk,
        device=str(args["actual_seq_lengths_query"].device),
    )
    return op(**args)


@register_impl("indexer.quant_lightning_indexer_metadata", "torch_native")
def _torch_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.indexer.quant_lightning_indexer_metadata import (
        quant_lightning_indexer_metadata,
    )

    return quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )


@register_operator("indexer.quant_lightning_indexer_metadata")
def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, *, impl="auto", **kwargs
):
    """Build quantized C4 indexer metadata.

    impl: auto (SDK-selected), ops_transformer_cann910, cann_ops_transformer_cann920, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata", impl)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )
