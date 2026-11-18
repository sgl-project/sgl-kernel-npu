"""Public fused prefill indexer operator with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

OP = "indexer.flash_prefill_bnsd_indexer"

_triton_fn = None
_torch_native_fn = None


# Importing this module registers the three names without importing Triton.
@register_impl(OP, "auto")
@register_impl(OP, "triton")
def _triton_impl(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    topk,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    per_query_seq_lens=None,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.indexer.flash_prefill_bnsd_indexer import (
            flash_prefill_bnsd_indexer,
        )

        _triton_fn = flash_prefill_bnsd_indexer

    return _triton_fn(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        topk,
        sm_scale,
        score_type=score_type,
        qblock_mappings=qblock_mappings,
        per_query_seq_lens=per_query_seq_lens,
    )


@register_impl(OP, "torch_native")
def _torch_native_impl(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    topk,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    per_query_seq_lens=None,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.indexer.flash_prefill_bnsd_indexer import (
            flash_prefill_bnsd_indexer,
        )

        _torch_native_fn = flash_prefill_bnsd_indexer

    return _torch_native_fn(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        topk,
        sm_scale,
        score_type=score_type,
        qblock_mappings=qblock_mappings,
        per_query_seq_lens=per_query_seq_lens,
    )


@register_operator(OP)
def flash_prefill_bnsd_indexer(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    cu_seqlens,
    seq_lens,
    req_to_token,
    req_pool_indices,
    block_size_q,
    block_size_k,
    topk,
    sm_scale,
    score_type="max",
    qblock_mappings=None,
    per_query_seq_lens=None,
    *,
    impl="auto",
):
    """Fuse prefill index scoring, causal index-head attention, and block top-k.

    ``auto`` defaults to ``triton``; ``"triton"`` is the same entry and
    ``"torch_native"`` is an independent reference. Returns ``(idx_o, topk_idx)``:
    ``idx_o`` is ``q.dtype [total_q, num_idx_heads, head_dim]``; ``topk_idx`` is
    ``int32`` with last dim ``topk + 1`` when ``per_query_seq_lens`` is given and
    ``q.shape[1] == k_cache_bnsd.shape[2]``, else ``topk``; ``-1`` marks empty
    slots. Inputs are not modified.
    """
    return get_impl_by_name(OP, impl)(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        cu_seqlens,
        seq_lens,
        req_to_token,
        req_pool_indices,
        block_size_q,
        block_size_k,
        topk,
        sm_scale,
        score_type=score_type,
        qblock_mappings=qblock_mappings,
        per_query_seq_lens=per_query_seq_lens,
    )
