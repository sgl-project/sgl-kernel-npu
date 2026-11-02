"""cann_ops_transformer adapter for indexer_quant_cache."""

from .._ops import get_op


def indexer_quant_cache(cache, scale_cache, x, slot_mapping, **kwargs):
    get_op("indexer_quant_cache")(cache, scale_cache, x, slot_mapping, **kwargs)
    return cache, scale_cache
