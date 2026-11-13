"""Public attention sinks."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.attention_sinks", "auto")
@register_impl("attention.attention_sinks", "triton")
def _triton_impl(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.attention.sinks_attention import attention_sinks

        _triton_fn = attention_sinks
    return _triton_fn(
        query,
        k_cache,
        v_cache,
        sinks,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
    )


@register_impl("attention.attention_sinks", "torch_native")
def _torch_native_impl(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.sinks_attention import (
            attention_sinks,
        )

        _torch_native_fn = attention_sinks
    return _torch_native_fn(
        query,
        k_cache,
        v_cache,
        sinks,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
    )


@register_operator("attention.attention_sinks")
def attention_sinks(
    query,
    k_cache,
    v_cache,
    sinks,
    block_tables,
    context_lens,
    scale,
    sliding_window_size,
    q_head_num,
    k_head_num,
    *,
    impl="auto"
):
    """Computes paged grouped-query decode attention with a learned sink logit per query head. The sink participates in the softmax denominator with zero value contribution. Optional sliding windows limit visible KV tokens.

    Args:
        query (Tensor): Contiguous floating [tokens,QH*D] queries. Default: Required.
        k_cache (Tensor): Contiguous floating [pages,page_size,KVH,D] key cache. Default: Required.
        v_cache (Tensor): Value cache with the same shape and dtype as k_cache. Default: Required.
        sinks (Tensor): Floating [QH] sink logits. Default: Required.
        block_tables (Tensor): INT32/INT64 [batch,max_pages] physical page IDs. Default: Required.
        context_lens (Tensor): INT32/INT64 [batch] total KV lengths including appended queries. Default: Required.
        scale (float or None): QK scaling; KDA defaults to key_dim**-0.5 and requires positive values. Default: Required.
        sliding_window_size (int): Visible token window; -1 disables the window. Default: Required.
        q_head_num (int): Query heads; divisible by k_head_num. Default: Required.
        k_head_num (int): KV head count. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        [tokens,QH*D] attention output, query dtype/device.
    """
    return get_impl_by_name("attention.attention_sinks", impl)(
        query,
        k_cache,
        v_cache,
        sinks,
        block_tables,
        context_lens,
        scale,
        sliding_window_size,
        q_head_num,
        k_head_num,
    )
