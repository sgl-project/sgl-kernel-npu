"""DSV4 shared-KV sparse MLA reference descriptor, not CANN tiling bytes."""

from typing import NamedTuple


class DSV4SparseMLAMetadata(NamedTuple):
    parameters: dict

    def to(self, device):
        return self


def dsv4_sparse_mla_metadata(**kwargs):
    return DSV4SparseMLAMetadata(kwargs)
