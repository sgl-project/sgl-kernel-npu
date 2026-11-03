"""Public quant_lightning_indexer operator with registered implementations."""

import torch

from ..utils._transformer_compat import OperatorUnavailableError
from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ._backend import indexer_family


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


@register_impl("attention.quant_lightning_indexer", "auto")
@register_impl("attention.quant_lightning_indexer", "cann_ops_transformer")
def _cann_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    family = indexer_family()
    if not family.legacy:
        return family.ops[1](
            q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
        )
    if quant_mode not in (1, 2):
        raise OperatorUnavailableError(
            "Legacy indexer supports the migrated FP8/INT8 modes only"
        )
    return family.ops[1](
        query=q,
        key=k,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        sparse_count=topk,
        **_legacy_indexer_kwargs(kwargs),
    )


@register_impl("attention.quant_lightning_indexer", "torch_native")
def _torch_quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
):
    from ...implementation.torch_native.attention.quant_lightning_indexer import (
        quant_lightning_indexer,
    )

    return quant_lightning_indexer(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )


@register_operator("attention.quant_lightning_indexer")
def quant_lightning_indexer(
    q, k, weights, q_scale, k_scale, topk, quant_mode, *, impl="auto", **kwargs
):
    """Return logical compressed-KV top-k indices and optional values.

    impl: auto (released/legacy compatibility), cann_ops_transformer, or torch_native.
    torch_native is an eager correctness reference, not a graph fallback.
    """
    indices, values = get_impl_by_name("attention.quant_lightning_indexer", impl)(
        q, k, weights, q_scale, k_scale, topk, quant_mode, **kwargs
    )
    return indices.to(torch.int32), (
        values
        if kwargs.get("return_value", 0)
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )
