"""Public quant_lightning_indexer_metadata operator with registered implementations."""

from ..utils._transformer_compat import OperatorUnavailableError, indexer_family
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._indexer_compat import legacy_indexer_kwargs


@register_impl("attention.quant_lightning_indexer_metadata", "auto")
@register_impl("attention.quant_lightning_indexer_metadata", "cann_ops_transformer")
def _cann_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    family = indexer_family()
    if not family.legacy:
        return family.ops[0](
            num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
        )
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "Legacy indexer metadata supports the migrated FP8/INT8 modes only"
        )
    args = legacy_indexer_kwargs(kwargs)
    args.update(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        sparse_count=topk,
        device=str(args["actual_seq_lengths_query"].device),
    )
    return family.ops[0](**args)


@register_impl("attention.quant_lightning_indexer_metadata", "torch_native")
def _torch_quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.attention.quant_lightning_indexer_metadata import (
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

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    return get_impl_by_name("attention.quant_lightning_indexer_metadata", impl)(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )
