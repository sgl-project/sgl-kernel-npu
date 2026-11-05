"""CANN 9.2.0 ops-transformer interface for arch22 (A3)."""

from functools import lru_cache

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._lightning_indexer import validate_candidates, validate_inputs


@lru_cache(maxsize=1)
def _binding():
    from ..._third_party.sgl_ops_transformer.ops import lightning_indexer

    return lightning_indexer


@register_impl(
    "indexer.lightning_indexer_v2", "auto", supported_generations={Generation.ARCH22}
)
@register_impl(
    "indexer.lightning_indexer_v2",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("lightning_indexer_v2 requires CANN 9.2.0")
    return _binding()(*args, **kwargs)


@register_impl("indexer.lightning_indexer_v2", "torch_native")
def _native_impl(*args, **kwargs):
    from ...implementation.torch_native.indexer.lightning_indexer import (
        lightning_indexer_v2,
    )

    return lightning_indexer_v2(*args, **kwargs)


@register_operator("indexer.lightning_indexer_v2")
def lightning_indexer_v2(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
    candidate_topk_blocks=None,
    candidate_block_size=None,
    impl="auto"
):
    """See docs/api/{en,zh}/operators/indexer/lightning_indexer_v2.md for the contract."""
    validate_inputs(
        q,
        k,
        w,
        topk,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        return_value,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        block_table,
        output_idx_offset,
        max_seqlen_q,
    )
    if candidate_topk_blocks is not None:
        validate_candidates(
            topk,
            8 if candidate_block_size is None else candidate_block_size,
            width=candidate_topk_blocks,
        )
    for tensor in (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        cmp_residual_k,
        block_table,
        metadata,
    ):
        if tensor is not None and (
            tensor.dtype != torch.int32 or tensor.device != q.device
        ):
            raise ValueError(
                "Optional index/metadata tensors must be int32 on q.device"
            )
    if cmp_ratio != 1 and mask_mode == 3 and cmp_residual_k is None:
        raise ValueError("Compressed causal attention requires cmp_residual_k")
    return get_impl_by_name("indexer.lightning_indexer_v2", impl)(
        q=q,
        k=k,
        w=w,
        topk=topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
