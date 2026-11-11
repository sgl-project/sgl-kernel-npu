"""Public quant_lightning_indexer_metadata_dsl operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from importlib import import_module

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cannbot_dsl_fn = None


@register_impl(
    "indexer.quant_lightning_indexer_metadata_dsl",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_lightning_indexer_metadata_dsl",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(*args, **kwargs):
    global _cannbot_dsl_fn
    # Positional arguments are optional sequence tensors.
    device = next(
        (value.device for value in args if value is not None),
        torch.npu.current_device(),
    )
    with torch.npu.device(device):
        if _cannbot_dsl_fn is None:
            _cannbot_dsl_fn = import_module(
                "sgl_kernel_npu.implementation.cannbot-dsl.indexer.quant_lightning_indexer_metadata"
            ).quant_lightning_indexer_metadata
        return _cannbot_dsl_fn(*args, **kwargs)


@register_operator("indexer.quant_lightning_indexer_metadata_dsl")
def quant_lightning_indexer_metadata_dsl(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
    impl="auto",
):
    """Build scheduling metadata for ``quant_lightning_indexer_dsl``.

    ``auto`` selects the lazy A5 CANNBot-DSL backend.

    Args:
        cu_seqlens_q: Optional INT32 [B+1], cumulative query row boundaries.
        cu_seqlens_k: Optional INT32 [B+1], cumulative TND key boundaries.
        seqused_q: Optional INT32 [B], valid query rows per request.
        seqused_k: Optional INT32 [B], valid key tokens per request.
        cmp_residual_k: Optional INT32 [B], uncompressed residual tokens for causal compression.
        batch_size: Optional int B for PA_BBND; TND requires None and infers B from query boundaries.
        max_seqlen_q: int, query capacity per request; -1 derives it from sequence lengths.
        max_seqlen_k: int, key capacity per request; -1 means infer.
        num_heads_q: int, query heads; use 32.
        num_heads_k: int, shared index key heads; use 1.
        head_dim: int, logical head dimension; use 128.
        topk: int, maximum selected tokens per query, in [1,8192].
        mask_mode: int: 0 for visible keys, 3 for causal masking.
        cmp_ratio: int in [1,128], compression ratio used to calculate causal visibility.
        layout_q: str, use TND (flattened query tokens).
        layout_k: str, TND or PA_BBND key storage.
        candidate_topk_blocks: int, candidate-block output capacity; -1 disables the output.
        candidate_block_size: int, use 8 tokens per candidate block when enabled.
        impl: str, auto and cannbot_dsl select the same AICPU schedule implementation.
    """
    return get_impl_by_name("indexer.quant_lightning_indexer_metadata_dsl", impl)(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
