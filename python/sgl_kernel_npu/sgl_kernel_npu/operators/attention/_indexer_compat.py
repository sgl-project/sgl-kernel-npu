"""Translate the original quantized indexer's full-length ABI."""

from ..utils._transformer_compat import OperatorUnavailableError


def legacy_indexer_kwargs(kwargs):
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
