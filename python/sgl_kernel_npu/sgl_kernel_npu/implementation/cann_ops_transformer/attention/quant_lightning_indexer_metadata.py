"""cann_ops_transformer adapter for quant_lightning_indexer_metadata."""

from .._ops import get_op


def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    return get_op("quant_lightning_indexer_metadata")(
        num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
    )
