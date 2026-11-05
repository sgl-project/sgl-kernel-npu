"""Independent indexer reference metadata builder."""

from ..attention.metadata import ReferenceMetadata


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
