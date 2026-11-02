"""Independent eager reference for sparse_flash_mla_metadata."""

from ._metadata import ReferenceMetadata


def sparse_flash_mla_metadata(**kwargs):
    return ReferenceMetadata(kwargs)
