"""Independent eager reference for sparse_flash_mla."""

from ._sparse_flash_mla import _attention


def sparse_flash_mla(q, **kwargs):
    return _attention(q, **kwargs)
