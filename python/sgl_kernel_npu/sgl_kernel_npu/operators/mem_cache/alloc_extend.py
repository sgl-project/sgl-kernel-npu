"""Public in-place paged allocation with explicit implementation selection."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("mem_cache.alloc_extend", "auto")
@register_impl("mem_cache.alloc_extend", "ascendc")
def _ascendc_impl(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    return torch.ops.npu.alloc_extend(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )


@register_impl("mem_cache.alloc_extend", "torch_native")
def _torch_native_impl(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.alloc_extend import alloc_extend

        _torch_native_fn = alloc_extend
    return _torch_native_fn(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )


@register_operator("mem_cache.alloc_extend")
def alloc_extend(
    pre_lens,
    seq_lens,
    last_loc,
    free_pages,
    page_size,
    out_indices,
    values,
    *,
    impl="auto"
):
    """Compute slot indices for tokens appended to paged requests.

    Args:
        pre_lens: int64 tensor of shape [B]. Existing length of each request. B must be
            positive.
        seq_lens: int64 tensor of shape [B]. Target lengths, each at least its prefix length.
        last_loc: int64 tensor of shape [B]. Last occupied slot of each request. Use -1 for an
            empty prefix.
        free_pages: int64 tensor of shape [P]. Available page IDs in allocation order. Supply
            enough pages for the batch.
        page_size: Positive number of slots per page, representable as int32.
        out_indices: int64 tensor of shape [sum(seq_lens - pre_lens)]. Caller-allocated output
            for the new slot indices.
        values: int64 tensor of shape [N], N ≥ 1. Writes the batch’s new-page count to element
            0 only if the final request ends in a newly allocated partial page. Otherwise it
            retains its input value.
        impl: Keyword-only. Choices are "auto", "ascendc", "torch_native". Defaults to
            "ascendc" via "auto".

    Returns:
        None. Writes out_indices and conditionally values[0] in place.
    """
    return get_impl_by_name("mem_cache.alloc_extend", impl)(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )
