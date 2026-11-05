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


def quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode, **kwargs
):
    return ReferenceMetadata(
        dict(
            num_heads_q=num_heads_q,
            num_heads_k=num_heads_k,
            head_dim=head_dim,
            topk=topk,
            quant_mode=quant_mode,
            **kwargs,
        )
    )
