"""Independent indexer reference metadata builder."""

from typing import NamedTuple


class ReferenceMetadata(NamedTuple):
    """Indexer-only reference descriptor; not vendor tiling bytes."""

    parameters: dict

    def to(self, device):
        return self


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
