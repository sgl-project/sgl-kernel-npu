"""Public pooled-key index selection backed by the attentions extension."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_aclnn_fn = None


@register_impl("indexer.pool_key_indexer", "auto")
@register_impl("indexer.pool_key_indexer", "aclnn")
def _aclnn_impl(
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
    key_stride0,
):
    global _aclnn_fn
    if _aclnn_fn is None:
        import attentions  # noqa: F401 -- loads the native op registration
        import torch

        _aclnn_fn = torch.ops.attentions.pool_key_indexer

    return _aclnn_fn(
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
        key_stride0,
    )


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
    key_stride0=-1,
    *,
    impl="auto",
):
    """Select pooled top-k token indices and append the incomplete tail.

    ``auto`` and ``aclnn`` select the native attentions binding, loaded on first
    execution. Parameters/defaults mirror its schema; validation stays in C++.
    ``pool_tail_k`` and ``actual_seq_*`` are host integer sequences, not device
    tensors. The binding supports BF16/FP16 with quant_mode=-1 and no values.
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
        key_stride0,
    )
