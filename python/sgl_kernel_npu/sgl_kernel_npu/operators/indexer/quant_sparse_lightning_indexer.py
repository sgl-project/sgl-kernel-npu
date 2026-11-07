"""Public quant_sparse_lightning_indexer operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from typing import Optional

import torch

from ..utils.cannbot_dsl import ascend950_device, load_cannbot_dsl
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl(
    "indexer.quant_sparse_lightning_indexer",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_sparse_lightning_indexer",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    with ascend950_device(*args, **kwargs):
        operator = load_cannbot_dsl("quant_sparse_lightning_indexer")
        return operator(*args, **kwargs)


@register_operator("indexer.quant_sparse_lightning_indexer")
def quant_sparse_lightning_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    descale_q: torch.Tensor,
    candidate_block_indices: torch.Tensor,
    candidate_block_length: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    *,
    descale_k: Optional[torch.Tensor] = None,
    topk: int,
    candidate_block_size: int,
    quant_mode: int,
    max_seqlen_q: int = -1,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    layout_q: str = "TND",
    layout_k: str = "TND",
    return_value: bool = False,
    impl="auto",
):
    """Select top-k key indices from the supplied candidate blocks.

    ``auto`` and ``cannbot_dsl`` select the same lazy-loaded A5 implementation.
    """
    return get_impl_by_name("indexer.quant_sparse_lightning_indexer", impl)(
        q,
        k,
        w,
        descale_q,
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        descale_k=descale_k,
        topk=topk,
        candidate_block_size=candidate_block_size,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
    )
