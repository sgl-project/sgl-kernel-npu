"""cann_ops_transformer adapter for sparse_flash_mla."""

from .._ops import get_op


def sparse_flash_mla(q, **kwargs):
    return get_op("sparse_flash_mla")(q, **kwargs)
