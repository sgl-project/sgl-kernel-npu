"""Independent eager reference for mixed_quant_sparse_flash_mla."""

from ._sparse_flash_mla import _attention


def mixed_quant_sparse_flash_mla(q, **kwargs):
    return _attention(q, **kwargs)
