"""In-place packed segment copies; return whether the input metadata is accepted."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None


@register_impl("mem_cache.assign_cache_op", "auto")
@register_impl("mem_cache.assign_cache_op", "ascendc")
def _ascendc_impl(dst, src, dst_start, dst_end, src_start, src_end):
    import torch

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
    """Copy src[src_start[i]:src_end[i]] to row i starting at dst_start[i].

    dst is dense [batch, width], src is packed 1D, and indices are int64.
    Matching interval lengths and in-bounds storage are caller preconditions.
    dst_end is retained in the interface; the copy length comes from src indices.
    """
    tensors = (dst, src, dst_start, dst_end, src_start, src_end)
    if any(t is None or t.ndim < 1 for t in tensors):
        return False
    # Index arrays may contain an extra terminal offset or unused batch suffix.
    if any(t.shape[0] < dst.shape[0] for t in tensors[2:]):
        return False
    return get_impl_by_name("mem_cache.assign_cache_op", impl)(*tensors)
