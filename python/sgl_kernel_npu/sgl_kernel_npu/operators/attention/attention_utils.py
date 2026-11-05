"""Choose paired attention bindings before KV allocation or graph capture.

Metadata, readers and packed-KV writers must share one backend selection.
Operator-specific input conversions belong to their public operator modules.
"""

import functools

import torch

from ..utils.transformer_compat import (
    OperatorFamily,
    OperatorUnavailableError,
    _lookup,
    select_family,
)


@functools.lru_cache(maxsize=None)
def attention_family(quantized):
    prefix = "mixed_quant_sparse_flash_mla" if quantized else "sparse_flash_mla"
    old = (
        "npu_kv_quant_sparse_attn_sharedkv" if quantized else "npu_sparse_attn_sharedkv"
    )
    released = (
        ("cann_ops_transformer", prefix + "_metadata"),
        ("cann_ops_transformer", prefix),
    )
    legacy = (("custom", old + "_metadata"), ("custom", old))
    if quantized:
        # Reader, metadata and writer must agree on 608B/BF16-scale versus
        # 640B/E8M0-scale encoding before any cache is allocated.
        released += (("cann_ops_transformer", "kv_compress_epilog"),)
        legacy += (("npu", "kv_compress_epilog"),)
    return select_family(released, legacy)


@functools.lru_cache(maxsize=None)
def sparse_host_family(quantized):
    stem = "kv_quant_sparse_attn_sharedkv" if quantized else "sparse_attn_sharedkv"
    specs = (("npu", stem + "_metadata_host"), ("npu", stem))
    ops = _lookup(specs)
    if any(op is None for op in ops):
        raise OperatorUnavailableError(
            "Legacy explicit sparse attention requires the paired sgl-kernel-npu "
            f"host metadata and compute bindings: {specs}"
        )
    return OperatorFamily("cann_9_1", ops, specs)


def logical_to_physical(indices, lengths, q, table, cache, cu_q, layout):
    """Resolve original-KV indices for the native DSpark ABI using tensor ops."""
    if table is None:
        raise ValueError("Legacy sparse original KV requires ori_block_table")
    if layout == "TND":
        if cu_q is None:
            raise ValueError("TND sparse original KV requires cu_seqlens_q")
        token = torch.arange(q.shape[0], device=q.device)
        batch = (token[:, None] >= cu_q[None, 1:]).sum(-1).clamp(max=table.shape[0] - 1)
    elif layout == "BSND":
        batch = torch.arange(q.shape[0], device=q.device).repeat_interleave(q.shape[1])
    else:
        raise OperatorUnavailableError(
            "Legacy sparse indices support TND or BSND queries"
        )
    flat = indices.reshape(-1, indices.shape[-1]).long()
    valid = flat >= 0
    if lengths is not None:
        column = torch.arange(flat.shape[-1], device=q.device)
        valid = valid & (column < lengths.reshape(-1, 1))
    safe = flat.clamp(min=0, max=table.shape[1] * cache.shape[1] - 1)
    pages = table[batch[:, None], safe // cache.shape[1]].long()
    slots = pages * cache.shape[1] + safe % cache.shape[1]
    return torch.where(valid, slots, -1).to(torch.int32).reshape_as(indices)


def _legacy_indexer_kwargs(kwargs):
    args = dict(kwargs)
    ratio = args.pop("cmp_ratio", 1)
    cu_q = args.pop("cu_seqlens_q", None)
    q_lengths = args.pop("seqused_q", None)
    k_lengths = args.pop("seqused_k", None)
    residual = args.pop("cmp_residual_k", None)
    if cu_q is not None:
        actual_q = cu_q[1:].clone()
    elif q_lengths is not None:
        actual_q = q_lengths.cumsum(0).to(q_lengths.dtype)
    else:
        raise ValueError("Legacy indexer requires cu_seqlens_q or seqused_q")
    if k_lengths is None:
        raise ValueError("Legacy indexer requires compressed seqused_k")
    actual_k = k_lengths * ratio
    if residual is not None:
        actual_k = actual_k + residual
    for name in ("cu_seqlens_k", "output_idx_offset"):
        if args.pop(name, None) is not None:
            raise OperatorUnavailableError(f"Legacy indexer does not support {name}")
    args.pop("max_seqlen_q", None)
    if args.pop("return_value", 0) != 0:
        raise OperatorUnavailableError(
            "Legacy indexer score outputs are not part of the migrated contract"
        )
    layout_q = args.pop("layout_q", "BSND")
    layout_k = args.pop("layout_k", "BSND")
    if layout_k == "PA_BBND":
        layout_k = "PA_BSND"
    args.update(
        actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=actual_k,
        layout_query=layout_q,
        layout_key=layout_k,
        sparse_mode=args.pop("mask_mode", 0),
        cmp_ratio=ratio,
        query_quant_mode=0,
        key_quant_mode=0,
    )
    return args
