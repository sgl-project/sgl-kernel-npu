"""Public quant_sparse_lightning_indexer_dsl operator backed by CANNBot-DSL on arch35."""

from __future__ import annotations

from importlib import import_module
from typing import Optional

import torch

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cannbot_dsl_fn = None


@register_impl(
    "indexer.quant_sparse_lightning_indexer_dsl",
    "torch_native",
    supported_generations={Generation.ARCH35},
)
def _torch_native_impl(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.quant_sparse_lightning_indexer_dsl import (
            quant_sparse_lightning_indexer_dsl,
        )

        _torch_native_fn = quant_sparse_lightning_indexer_dsl
    return _torch_native_fn(*args, **kwargs)


@register_impl(
    "indexer.quant_sparse_lightning_indexer_dsl",
    "auto",
    supported_generations={Generation.ARCH35},
)
@register_impl(
    "indexer.quant_sparse_lightning_indexer_dsl",
    "cannbot_dsl",
    supported_generations={Generation.ARCH35},
)
def _cannbot_dsl_impl(q, *args, **kwargs):
    global _cannbot_dsl_fn
    with torch.npu.device(q.device):
        if _cannbot_dsl_fn is None:
            _cannbot_dsl_fn = import_module(
                "sgl_kernel_npu.implementation.cannbot-dsl.indexer.quant_sparse_lightning_indexer"
            ).quant_sparse_lightning_indexer
        return _cannbot_dsl_fn(q, *args, **kwargs)


@register_operator("indexer.quant_sparse_lightning_indexer_dsl")
def quant_sparse_lightning_indexer_dsl(
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

    ``auto`` selects the lazy A5 CANNBot-DSL backend.

    Args:
        q: UINT8 [T,32,64], packed MXFP4 queries with logical dimension 128.
        k: UINT8 K: TND [K,1,64] or PA_BBND [P,S,1,64]. QSLI PA uses combined [P,S/8,544] bytes.
        w: FP32 [T,32], query-head score weights.
        descale_q: UINT8/E8M0 [T,32,2,2], one scale per 32 logical features.
        candidate_block_indices: INT32 [T,1,2048], candidate logical block IDs; valid prefix given by lengths.
        candidate_block_length: INT32 [T,1], valid candidate block count per query.
        cu_seqlens_q: Optional INT32 [B+1], cumulative query row boundaries.
        cu_seqlens_k: Optional INT32 [B+1], cumulative TND key boundaries.
        seqused_q: Optional INT32 [B], valid query rows per request.
        seqused_k: Optional INT32 [B], valid key tokens per request.
        cmp_residual_k: Optional INT32 [B], uncompressed residual tokens for causal compression.
        block_table: Optional INT32 [B,max_blocks], logical-to-physical PA page map.
        output_idx_offset: Optional INT32 [T,1], added to each valid selected token index.
        metadata: INT32 [1024] device schedule required by cannbot_dsl; torch_native ignores it.
        descale_k: UINT8/E8M0 [K,1,2,2] or [P,S,1,2,2]; omit for QSLI combined PA storage.
        topk: int, maximum selected tokens per query, in [1,8192].
        candidate_block_size: int, use 8 tokens per candidate block when enabled.
        quant_mode: int, use 1 for the supported quantized profile.
        max_seqlen_q: int or None, query capacity per request; infer when unspecified.
        mask_mode: int: 0 for visible keys, 3 for causal masking.
        cmp_ratio: int in [1,128], compression ratio used to calculate causal visibility.
        layout_q: str, use TND (flattened query tokens).
        layout_k: str, TND or PA_BBND key storage.
        return_value: bool, also return BF16 selected scores when True.
        impl: str, auto selects cannbot_dsl; torch_native is an independent compute reference.
    """
    return get_impl_by_name("indexer.quant_sparse_lightning_indexer_dsl", impl)(
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
