"""Public flash prefill bnsd blockq sparse fia."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.flash_prefill_bnsd_blockq_sparse_fia", "auto")
@register_impl("attention.flash_prefill_bnsd_blockq_sparse_fia", "triton")
def _triton_impl(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    topk_idx,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    sm_scale,
    num_pages,
    max_num_blocks,
    block_table_out=None,
    actual_kvlen_out=None,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.fia_blockq_attention import (
            flash_prefill_bnsd_blockq_sparse_fia,
        )

        _triton_fn = flash_prefill_bnsd_blockq_sparse_fia
    return _triton_fn(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        topk_idx,
        seq_lens,
        per_query_req,
        req_to_token,
        block_size,
        sm_scale,
        num_pages,
        max_num_blocks,
        block_table_out,
        actual_kvlen_out,
    )


@register_impl("attention.flash_prefill_bnsd_blockq_sparse_fia", "torch_native")
def _torch_native_impl(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    topk_idx,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    sm_scale,
    num_pages,
    max_num_blocks,
    block_table_out=None,
    actual_kvlen_out=None,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.fia_blockq_attention import (
            flash_prefill_bnsd_blockq_sparse_fia,
        )

        _torch_native_fn = flash_prefill_bnsd_blockq_sparse_fia
    return _torch_native_fn(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        topk_idx,
        seq_lens,
        per_query_req,
        req_to_token,
        block_size,
        sm_scale,
        num_pages,
        max_num_blocks,
        block_table_out,
        actual_kvlen_out,
    )


@register_operator("attention.flash_prefill_bnsd_blockq_sparse_fia")
def flash_prefill_bnsd_blockq_sparse_fia(
    q,
    k_cache_bnsd,
    v_cache_bnsd,
    topk_idx,
    seq_lens,
    per_query_req,
    req_to_token,
    block_size,
    sm_scale,
    num_pages,
    max_num_blocks,
    block_table_out=None,
    actual_kvlen_out=None,
    *,
    impl="auto"
):
    """Computes sparse causal prefill attention from selected logical KV blocks. It gathers physical pages, orders the current block after past blocks, limits that block to the query's causal length, and evaluates grouped-query attention. Only one KV head is supported; optional metadata buffers are updated in place.

    Args:
        q (Tensor): KDA: [1,tokens,Hq,K]; FIA: FP16/BF16 [tokens,QH,D]. Default: Required.
        k_cache_bnsd (Tensor): FP16/BF16 [num_pages,block_size,1,D] cache. Default: Required.
        v_cache_bnsd (Tensor): Value cache, same shape/dtype as keys. Default: Required.
        topk_idx (Tensor): INT32 [1,tokens,topk+1] selected logical blocks, -1 padding, including the current block. Default: Required.
        seq_lens (Tensor): Integer per-request query lengths; FIA instead uses per-query causal KV lengths. Default: Required.
        per_query_req (Tensor): INT32 [tokens] request indices. Default: Required.
        req_to_token (Tensor): INT32 [requests,max_context] physical token slots. Default: Required.
        block_size (int): Tokens per physical KV page. Default: Required.
        sm_scale (float or None): Attention score scale; None uses head_dim**-0.5. Default: Required.
        num_pages (int): Physical cache page count, matching cache shape. Default: Required.
        max_num_blocks (int): Reserved maximum selected-block count. Default: Required.
        block_table_out (Tensor or None): Optional writable INT32 [tokens,topk+1] page table. Default: None.
        actual_kvlen_out (Tensor or None): Optional writable INT32 [tokens] causal lengths. Default: None.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [tokens,QH,D] output with q dtype/device.
    """
    return get_impl_by_name("attention.flash_prefill_bnsd_blockq_sparse_fia", impl)(
        q,
        k_cache_bnsd,
        v_cache_bnsd,
        topk_idx,
        seq_lens,
        per_query_req,
        req_to_token,
        block_size,
        sm_scale,
        num_pages,
        max_num_blocks,
        block_table_out,
        actual_kvlen_out,
    )
