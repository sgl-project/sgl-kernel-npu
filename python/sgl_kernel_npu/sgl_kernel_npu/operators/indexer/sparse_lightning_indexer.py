"""CANN 9.2.0 ops-transformer interface for arch22 (A3)."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from .indexer_utils import (
    validate_lightning_indexer_candidates,
    validate_lightning_indexer_inputs,
)

_cann_fn = None
_native_fn = None


def _binding():
    global _cann_fn
    if _cann_fn is None:
        from ..._third_party.sgl_ops_transformer.ops import sparse_lightning_indexer

        _cann_fn = sparse_lightning_indexer
    return _cann_fn


@register_impl(
    "indexer.sparse_lightning_indexer",
    "auto",
    supported_generations={Generation.ARCH22},
)
@register_impl(
    "indexer.sparse_lightning_indexer",
    "cann_ops_transformers_cann920",
    supported_generations={Generation.ARCH22},
)
def _cann_impl(*args, **kwargs):
    if detect_cann_version() != "9.2.0":
        raise RuntimeError("sparse_lightning_indexer requires CANN 9.2.0")
    return _binding()(*args, **kwargs)


@register_impl("indexer.sparse_lightning_indexer", "torch_native")
def _native_impl(*args, **kwargs):
    global _native_fn
    if _native_fn is None:
        from ...implementation.torch_native.indexer.lightning_indexer import (
            sparse_lightning_indexer,
        )

        _native_fn = sparse_lightning_indexer
    return _native_fn(*args, **kwargs)


@register_operator("indexer.sparse_lightning_indexer")
def sparse_lightning_indexer(
    q,
    k,
    w,
    topk,
    candidate_block_indices,
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
    candidate_block_length=None,
    candidate_block_size=8,
    impl="auto"
):
    """Select weighted ReLU query/key TopK within candidate blocks.

    The default auto implementation uses bundled CANN 9.2.0 operators on A3.

    Args:
        q (Tensor): Contiguous fp16/bf16 [B,Q,H,128] or TND [Tq,H,128].
        k (Tensor): Same dtype; BSND [B,K,1,128], TND [Tk,1,128], or
            PA_BBND [P,S,1,128] paged keys.
        w (Tensor): Contiguous FP32 head weights with shape q.shape[:-1].
        topk (int): Number of selected positions, in 1..2048.
        candidate_block_indices (Tensor): Contiguous int32 [B,Q,1,C] or
            [Tq,1,C] block IDs; C=64..2048, divisible by 64.
        cu_seqlens_q (Tensor | None): Int32 [B+1]; TND prefix boundaries or
            BSND [0, valid_Q_length_0, ...]. Required for TND.
        cu_seqlens_k (Tensor | None): Int32 [B+1] TND key prefix boundaries.
        seqused_q (Tensor | None): Reserved; must be None on A3.
        seqused_k (Tensor | None): Int32 [B] valid key lengths.
        cmp_residual_k (Tensor | None): Int32 [B] compression remainders.
        block_table (Tensor | None): Int32 [B,max_pages], required for PA_BBND.
        output_idx_offset (Tensor | None): Reserved; must be None on A3.
        metadata (Tensor | None): Int32 [2048] scheduling data, ignored on A3.
        max_seqlen_q (int): Reserved; must remain -1 on A3.
        layout_q (str): BSND or TND; defaults to BSND.
        layout_k (str): BSND, TND or PA_BBND; defaults to BSND.
        mask_mode (int): 0 for no mask, 3 for right-aligned causal masking.
        cmp_ratio (int): Key compression ratio, a power of two in 1..128.
        return_value (int): 1 returns FP32 scores; 0 returns an empty tensor.
        candidate_block_length (Tensor | None): Reserved empty int32 tensor.
        candidate_block_size (int): Positions per block; defaults to 8.
        impl (str): auto, cann_ops_transformers_cann920, or torch_native.

    All tensors share q.device. Returns int32 indices and FP32 scores with shape
    [B,Q,1,topk] or [Tq,1,topk]. Invalid entries use -1 indices and zero scores;
    negative, duplicate and out-of-range block IDs add no extra positions.
    """
    validate_lightning_indexer_inputs(
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
    validate_lightning_indexer_candidates(
        topk,
        candidate_block_size,
        indices=candidate_block_indices,
        length=candidate_block_length,
    )
    for tensor in (
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        cmp_residual_k,
        block_table,
        metadata,
        candidate_block_length,
    ):
        if tensor is not None and (
            tensor.dtype != torch.int32 or tensor.device != q.device
        ):
            raise ValueError(
                "Optional index/metadata tensors must be int32 on q.device"
            )
    if cmp_ratio != 1 and mask_mode == 3 and cmp_residual_k is None:
        raise ValueError("Compressed causal attention requires cmp_residual_k")
    if candidate_block_indices.shape[:-1] != q.shape[:-2] + (1,):
        raise ValueError(
            "candidate_block_indices leading shape must match query rows and one key head"
        )
    if (
        candidate_block_indices.device != q.device
        or not candidate_block_indices.is_contiguous()
    ):
        raise ValueError("candidate_block_indices must be contiguous on q.device")
    return get_impl_by_name("indexer.sparse_lightning_indexer", impl)(
        q=q,
        k=k,
        w=w,
        topk=topk,
        candidate_block_indices=candidate_block_indices,
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
        candidate_block_length=candidate_block_length,
        candidate_block_size=candidate_block_size,
    )
