"""In-place ScatterND update with registered implementations."""

import torch
import torch_npu

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "mem_cache.scatter_nd_update_"
_native_impl = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "ops-nn")
def _torch_npu_impl(input, indices, updates):
    return torch_npu.npu_scatter_nd_update_(input, indices, updates)


@register_impl(_OP_NAME, "ascendc")
def _ascendc_impl(input, indices, updates):
    torch.ops.npu.scatter_nd_update_hp(input, indices, updates)
    return input


@register_impl(_OP_NAME, "torch_native")
def _torch_native_impl(*args, **kwargs):
    global _native_impl
    if _native_impl is None:
        from ...implementation.torch_native.mem_cache.paged_cache import (
            scatter_nd_update_,
        )

        _native_impl = scatter_nd_update_
    return _native_impl(*args, **kwargs)


@register_operator(_OP_NAME)
def scatter_nd_update_(input, indices, updates, *, impl="auto"):
    """Write updates at N-dimensional indices and return the mutated input.

    Each last-axis entry of indices is a coordinate vector of length N.
    updates must have shape indices.shape[:-1] + input.shape[N:].
    Auto selects ops-nn. The ascendc implementation skips -1 for row updates.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, indices, updates)
