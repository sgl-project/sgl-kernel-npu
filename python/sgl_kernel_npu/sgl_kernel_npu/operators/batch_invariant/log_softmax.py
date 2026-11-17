"""Public batch-invariant log-softmax operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops_fn = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.log_softmax", "auto")
@register_impl("batch_invariant.log_softmax", "batch_invariant_ops")
def _batch_invariant_ops_impl(input, dim):
    global _batch_invariant_ops_fn
    if _batch_invariant_ops_fn is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops_fn = (
            torch.ops.batch_invariant_ops.npu_log_softmax_batch_invariant
        )

    return _batch_invariant_ops_fn(input, dim)


@register_operator("batch_invariant.log_softmax")
def log_softmax_batch_invariant(input, dim, *, impl="auto"):
    """Log-softmax along dim with a batch-size-independent reduction order.

    Each row is bitwise identical alone or inside a larger batch. Only the
    last dim is supported by the kernel. impl selects auto or
    batch_invariant_ops.
    """
    impl_fn = get_impl_by_name("batch_invariant.log_softmax", impl)
    return impl_fn(input, dim)
