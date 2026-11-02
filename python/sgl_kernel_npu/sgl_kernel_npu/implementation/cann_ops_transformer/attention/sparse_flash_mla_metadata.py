"""cann_ops_transformer adapter for sparse_flash_mla_metadata."""

from .._ops import get_op


def sparse_flash_mla_metadata(**kwargs):
    return get_op("sparse_flash_mla_metadata")(**kwargs)
