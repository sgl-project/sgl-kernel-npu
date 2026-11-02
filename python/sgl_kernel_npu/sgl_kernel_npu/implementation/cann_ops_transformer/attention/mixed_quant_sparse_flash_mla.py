"""cann_ops_transformer adapter for mixed_quant_sparse_flash_mla."""

from .._ops import get_op


def mixed_quant_sparse_flash_mla(q, **kwargs):
    return get_op("mixed_quant_sparse_flash_mla")(q, **kwargs)
