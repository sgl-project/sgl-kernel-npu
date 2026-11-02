"""Public in-place paged allocation with explicit implementation selection."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_cpu_fn = None


@register_impl("mem_cache.alloc_extend", "auto")
@register_impl("mem_cache.alloc_extend", "ascendc")
def _ascendc_impl(
    pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
):
    import torch

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


@register_impl("mem_cache.alloc_extend", "cpu")
def _cpu_impl(pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values):
    global _cpu_fn
    if _cpu_fn is None:
        from ...implementation.cpu.mem_cache.alloc_extend import alloc_extend

        _cpu_fn = alloc_extend
    # CPU computation uses private buffers and copies results into the caller's outputs.
    inputs = [tensor.cpu() for tensor in (pre_lens, seq_lens, last_loc, free_pages)]
    output = out_indices.cpu().clone()
    page_count = values.cpu().clone()
    _cpu_fn(*inputs, page_size, output, page_count)
    out_indices.copy_(output)
    values.copy_(page_count)


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
    """Write allocation slots and conditionally write values[0]; return None.

    All buffers use int64. Inputs must satisfy the dense one-dimensional storage,
    page-capacity and positive-batch preconditions. values is not a general page-count API.
    """
    import torch

    if any(
        t.dtype != torch.int64
        for t in (pre_lens, seq_lens, last_loc, free_pages, out_indices)
    ):
        raise ValueError("Only support int64 input dtype")
    return get_impl_by_name("mem_cache.alloc_extend", impl)(
        pre_lens, seq_lens, last_loc, free_pages, page_size, out_indices, values
    )
