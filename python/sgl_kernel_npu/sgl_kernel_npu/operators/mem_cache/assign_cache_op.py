"""In-place packed segment copies; return whether the input metadata is accepted."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("mem_cache.assign_cache_op", "auto")
@register_impl("mem_cache.assign_cache_op", "ascendc")
def _ascendc_impl(dst, src, dst_start, dst_end, src_start, src_end):
    return torch.ops.npu.assign_cache_op(
        dst, src, dst_start, dst_end, src_start, src_end
    )


@register_impl("mem_cache.assign_cache_op", "torch_native")
def _torch_native_impl(dst, src, dst_start, dst_end, src_start, src_end):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.mem_cache.assign_cache_op import (
            assign_cache_op,
        )

        _torch_native_fn = assign_cache_op
    return _torch_native_fn(dst, src, dst_start, dst_end, src_start, src_end)


@register_operator("mem_cache.assign_cache_op")
def assign_cache_op(dst, src, dst_start, dst_end, src_start, src_end, *, impl="auto"):
    """Copy packed source segments into destination rows in place.

    Args:
        dst: payload tensor of shape [B, W]. Dense destination. Payloads with element widths of
            1, 2, 4 or 8 bytes are copied bit for bit.
        src: payload tensor of shape [N]. Dense packed source. Use matching element widths for
            element-wise copies.
        dst_start: int64 tensor of shape [L], L ≥ B. Inclusive destination column per row.
        dst_end: int64 tensor of shape [L], L ≥ B. Exclusive destination end column.
        src_start: int64 tensor of shape [L], L ≥ B. Inclusive source offset per row.
        src_end: int64 tensor of shape [L], L ≥ B. Exclusive source end offset per row.
        impl: Keyword-only. Choices are "auto", "ascendc", "torch_native". Defaults to
            "ascendc" via "auto".

    Returns:
        False if an input is missing or scalar, or an index vector has fewer than B entries.
            Otherwise a normal return is True, including when an unsupported element width
            causes no copy.
    """
    tensors = (dst, src, dst_start, dst_end, src_start, src_end)
    if any(t is None or t.ndim < 1 for t in tensors):
        return False
    # Index arrays may contain an extra terminal offset or unused batch suffix.
    if any(t.shape[0] < dst.shape[0] for t in tensors[2:]):
        return False
    return get_impl_by_name("mem_cache.assign_cache_op", impl)(*tensors)
