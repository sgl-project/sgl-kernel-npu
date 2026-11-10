"""Reference-only metadata descriptors; not CANN tiling bytes."""

from typing import NamedTuple


class ReferenceMetadata(NamedTuple):
    parameters: dict

    def to(self, device):
        return self


def sparse_flash_mla_metadata(**kwargs):
    return ReferenceMetadata(kwargs)


def mixed_quant_sparse_flash_mla_metadata(**kwargs):
    return ReferenceMetadata(kwargs)
