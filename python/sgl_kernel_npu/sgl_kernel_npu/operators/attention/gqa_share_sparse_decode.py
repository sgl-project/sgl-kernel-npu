"""Public sparse GQA decode attention."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.gqa_share_sparse_decode", "auto")
@register_impl("attention.gqa_share_sparse_decode", "triton")
def _triton(*args, **kwargs):
    from ...implementation.triton.attention.gqa_share_sparse_attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    return flash_decode_bnsd_with_gqa_share_sparse(*args, **kwargs, use_native=False)


@register_impl("attention.gqa_share_sparse_decode", "torch_native")
def _native(*args, **kwargs):
    from ...implementation.torch_native.attention.gqa_share_sparse_decode import (
        gqa_share_sparse_decode,
    )

    return gqa_share_sparse_decode(*args, **kwargs)


@register_operator("attention.gqa_share_sparse_decode")
def gqa_share_sparse_decode(
    q,
    sink,
    k_cache_bnsd,
    v_cache_bnsd,
    block_table,
    seq_lens,
    block_size,
    topk_idx,
    sm_scale=None,
    num_topk_chunks=None,
    max_num_topk_chunks=8,
    req_to_token=None,
    req_pool_indices=None,
    max_num_blocks=None,
    num_pages=None,
    sanitize_page_ids=False,
    topk_blocks_per_step=1,
    prefetch_idx=False,
    num_warps=None,
    num_stages=None,
    *,
    impl="auto",
):
    """Sparse decode attention over selected KV pages; auto selects Triton.

    q is [B,QH,D], KV caches are [pages,block_size,KVH,D], and topk_idx
    is [KVH or QH,B,topk]. sink, if provided, contributes a zero-value key.
    """
    return get_impl_by_name("attention.gqa_share_sparse_decode", impl)(
        q,
        sink,
        k_cache_bnsd,
        v_cache_bnsd,
        block_table,
        seq_lens,
        block_size,
        topk_idx,
        sm_scale,
        num_topk_chunks,
        max_num_topk_chunks,
        req_to_token,
        req_pool_indices,
        max_num_blocks,
        num_pages,
        sanitize_page_ids,
        topk_blocks_per_step,
        prefetch_idx,
        num_warps,
        num_stages,
    )
