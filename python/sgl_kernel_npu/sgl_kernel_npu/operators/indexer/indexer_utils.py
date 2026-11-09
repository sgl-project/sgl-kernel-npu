"""Metadata-only input checks shared by the arch22 indexers."""

import torch


def validate_lightning_indexer_inputs(
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
):
    if layout_q not in ("BSND", "TND") or layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError("q requires BSND/TND; k requires BSND/TND/PA_BBND")
    if q.ndim != (4 if layout_q == "BSND" else 3) or k.ndim != (
        3 if layout_k == "TND" else 4
    ):
        raise ValueError("Tensor ranks must match the declared layouts")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or w.dtype != torch.float32
    ):
        raise TypeError("q/k require matching fp16 or bf16; w requires fp32")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or k.shape[-2] != 1:
        raise ValueError("head_dim must be 128 and k must have one head")
    if not 1 <= q.shape[-2] <= 64 or w.shape != q.shape[:-1]:
        raise ValueError("q requires 1..64 heads and w.shape == q.shape[:-1]")
    if not q.numel() or not k.numel():
        raise ValueError("q and k must be nonempty")
    if q.device != k.device or q.device != w.device:
        raise ValueError("q, k and w must be on the same device")
    if layout_q == layout_k == "BSND" and q.shape[0] != k.shape[0]:
        raise ValueError("BSND q and k must have the same batch size")
    if not q.is_contiguous() or not w.is_contiguous():
        raise ValueError("q and w must be contiguous")
    if not k.is_contiguous() and (
        layout_k != "PA_BBND" or k.stride()[1:] != (128, 128, 1)
    ):
        raise ValueError("Only PA_BBND k may have a noncontiguous first axis")
    if not (1 <= topk <= 2048 or topk in (3072, 4096, 5120, 6144, 7168, 8192)):
        raise ValueError("Unsupported arch22 topk")
    if (
        mask_mode not in (0, 3)
        or cmp_ratio not in (1, 2, 4, 8, 16, 32, 64, 128)
        or return_value not in (0, 1)
    ):
        raise ValueError("Unsupported mask_mode, cmp_ratio or return_value")
    if seqused_q is not None or output_idx_offset is not None or max_seqlen_q != -1:
        raise ValueError(
            "arch22 does not support seqused_q/output_idx_offset/max_seqlen_q"
        )
    if layout_q == "TND" and cu_seqlens_q is None:
        raise ValueError("TND q requires cu_seqlens_q")
    if layout_k == "TND" and cu_seqlens_k is None:
        raise ValueError("TND k requires cu_seqlens_k")
    if layout_k == "BSND" and cu_seqlens_k is not None:
        raise ValueError("BSND k does not accept cu_seqlens_k")
    if layout_k == "PA_BBND":
        if block_table is None or seqused_k is None:
            raise ValueError("PA_BBND requires block_table and seqused_k")
        if not 16 <= k.shape[1] <= 1024 or k.shape[1] % 16:
            raise ValueError("PA block size must be a multiple of 16 in [16, 1024]")


def validate_lightning_indexer_candidates(
    topk, block_size, *, width=None, indices=None, length=None
):
    if block_size not in (2, 4, 8, 16, 32, 64):
        raise ValueError("candidate_block_size must be a power of two in [2, 64]")
    if indices is not None:
        if indices.dtype != torch.int32 or indices.ndim not in (3, 4):
            raise ValueError("candidate_block_indices must be rank 3/4 int32")
        width = indices.shape[-1]
    if width != -1 and (width is None or not 0 < width <= 2048 or width % 64):
        raise ValueError("candidate width must be a multiple of 64 in [64, 2048]")
    if width != -1 and topk > 2048:
        raise ValueError("candidate mode requires topk <= 2048")
    if length is not None and (length.dtype != torch.int32 or length.numel()):
        raise ValueError("candidate_block_length is reserved and must be empty")
