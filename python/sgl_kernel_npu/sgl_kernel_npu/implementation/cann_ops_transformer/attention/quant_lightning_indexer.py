"""cann_ops_transformer adapter for quant_lightning_indexer."""

from .._ops import get_op


def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    return get_op("quant_lightning_indexer")(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
