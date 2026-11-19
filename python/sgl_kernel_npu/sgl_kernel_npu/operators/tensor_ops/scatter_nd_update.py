"""In-place TorchNPU ScatterND update."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator

_OP_NAME = "tensor_ops.scatter_nd_update_"
_torch_npu_fn = None


@register_impl(_OP_NAME, "auto")
@register_impl(_OP_NAME, "torch_npu")
def _torch_npu_impl(input, indices, updates):
    global _torch_npu_fn
    if _torch_npu_fn is None:
        from torch_npu import npu_scatter_nd_update_

        _torch_npu_fn = npu_scatter_nd_update_
    return _torch_npu_fn(input, indices, updates)


@register_operator(_OP_NAME)
def scatter_nd_update_(input, indices, updates, *, impl="auto"):
    """Write updates at N-dimensional indices and return the mutated input.

    Each last-axis entry of indices is a coordinate vector of length N.
    updates must have shape indices.shape[:-1] + input.shape[N:].
    Auto selects torch_npu.
    """
    return get_impl_by_name(_OP_NAME, impl)(input, indices, updates)
