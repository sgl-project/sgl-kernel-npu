"""Public pooled-key index selection backed by CANN ops-transformer."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_cann_fn = None


def _metadata_tensor(value):
    if value is None or isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.int64)


@register_impl("indexer.pool_key_indexer", "auto")
@register_impl("indexer.pool_key_indexer", "cann_ops_transformer")
def _cann_impl(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q,
    actual_seq_k,
    block_table,
    topk,
    pool_size,
    layout_q,
    layout_k,
    mask_mode,
    quant_mode,
    return_value,
):
    global _cann_fn
    if quant_mode != -1:
        raise ValueError("pool_key_indexer supports quant_mode=-1 only")
    if return_value:
        raise ValueError(
            "pool_key_indexer returns indices only; return_value must be False"
        )

    if _cann_fn is None:
        from ..._third_party import sgl_ops_transformer  # noqa: F401

        _cann_fn = torch.ops.cann_ops_transformer.pool_key_indexer.default

    return _cann_fn(
        query,
        pool_key,
        weights,
        _metadata_tensor(pool_tail_k),
        actual_seq_q=_metadata_tensor(actual_seq_q),
        actual_seq_k=_metadata_tensor(actual_seq_k),
        block_table=block_table,
        topk=topk,
        pool_size=pool_size,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
    )[0]


@register_operator("indexer.pool_key_indexer")
def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    topk=2048,
    pool_size=16,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
    *,
    impl="auto",
):
    """Select pooled top-k token indices and append the incomplete tail.

    ``auto`` and ``cann_ops_transformer`` select the public CANN binding, loaded
    on first execution.
    ``pool_tail_k`` and ``actual_seq_*`` accept host integer sequences or
    int64 Tensors; host sequences are converted to CPU tensors.
    Device metadata uses the Tensor workspace API without D2H and can change
    between graph replays. The integration supports BF16/FP16 with quant_mode=-1
    and no values. CANN derives the key stride from ``pool_key``.
    Returns int32 [T, topk + pool_size - 1] for TND queries or
    [B, S, topk + pool_size - 1] for BSND queries. Inputs remain read-only.
    """
    impl_fn = get_impl_by_name("indexer.pool_key_indexer", impl)
    return impl_fn(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q,
        actual_seq_k,
        block_table,
        topk,
        pool_size,
        layout_q,
        layout_k,
        mask_mode,
        quant_mode,
        return_value,
    )
