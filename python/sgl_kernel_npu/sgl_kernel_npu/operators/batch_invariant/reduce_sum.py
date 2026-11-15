"""Public batch-invariant sum reduction operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops_fn = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.reduce_sum", "auto")
@register_impl("batch_invariant.reduce_sum", "batch_invariant_ops")
def _batch_invariant_ops_impl(input, dim, keepdim):
    global _batch_invariant_ops_fn
    if _batch_invariant_ops_fn is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops_fn = (
            torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant
        )

    return _batch_invariant_ops_fn(input, dim, keepdim)


@register_operator("batch_invariant.reduce_sum")
def reduce_sum_batch_invariant(input, dim, keepdim=False, *, impl="auto"):
    """Sum over one dimension, independent of the batch size.

    Each row is bitwise identical alone or inside a larger batch. Only a
    single dim is supported; impl selects auto or batch_invariant_ops.
    """
    impl_fn = get_impl_by_name("batch_invariant.reduce_sum", impl)
    return impl_fn(input, dim, keepdim)
