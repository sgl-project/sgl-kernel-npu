"""Public alloc extend indices."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("mem_cache.alloc_extend_indices", "auto")
@register_impl("mem_cache.alloc_extend_indices", "triton")
def _triton_impl(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
):
    global _triton_fn
    if _triton_fn is None:
        from ...implementation.triton.mem_cache.alloc_extend_indices import (
            alloc_extend_indices,
        )

        _triton_fn = alloc_extend_indices
    return _triton_fn(
        pre_lens,
        seq_lens,
        last_loc,
        free_pages,
        page_size,
        out_indices,
        max_num_extend_tokens,
    )


@register_impl("mem_cache.alloc_extend_indices", "torch_native")
def _torch_native_impl(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.alloc_extend_indices import (
            alloc_extend_indices,
        )

        _torch_native_fn = alloc_extend_indices
    return _torch_native_fn(
        pre_lens,
        seq_lens,
        last_loc,
        free_pages,
        page_size,
        out_indices,
        max_num_extend_tokens,
    )


@register_operator("mem_cache.alloc_extend_indices")
def alloc_extend_indices(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    max_num_extend_tokens,
    *,
    impl="auto"
):
    """Fills packed slot indices for appended tokens, first using the old partial page and then consuming free page IDs in batch order. This interface returns no page-count buffer and does not mutate free_pages. It preserves the index dtypes of the caller-provided tensors.

    Args:
        pre_lens (Tensor): Contiguous INT32/INT64 [batch] existing sequence lengths. Default: Required.
        seq_lens (Tensor): Contiguous INT32/INT64 [batch] target lengths, at least pre_lens. Default: Required.
        last_loc (Tensor): Contiguous integer [batch] last occupied slot; -1 for empty prefixes. Default: Required.
        free_pages (Tensor): Contiguous integer [pages] free page IDs, sufficient for all extensions. Default: Required.
        page_size (int): Positive power-of-two slots per page. Default: Required.
        out_indices (Tensor): Writable integer [sum(seq_lens-pre_lens)] packed slots. Default: Required.
        max_num_extend_tokens (int): Upper bound on each request's appended token count. Default: Required.
        impl (str): auto selects triton; explicit backend selection. Default: 'auto'.

    Returns:
        Writes out_indices in place.
    """
    return get_impl_by_name("mem_cache.alloc_extend_indices", impl)(
        pre_lens,
        seq_lens,
        last_loc,
        free_pages,
        page_size,
        out_indices,
        max_num_extend_tokens,
    )
