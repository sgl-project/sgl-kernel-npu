"""Public batch-invariant batched matrix multiplication operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops_fn = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.matmul", "auto")
@register_impl("batch_invariant.matmul", "batch_invariant_ops")
def _batch_invariant_ops_impl(input, other):
    global _batch_invariant_ops_fn
    if _batch_invariant_ops_fn is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops_fn = (
            torch.ops.batch_invariant_ops.npu_matmul_batch_invariant
        )

    return _batch_invariant_ops_fn(input, other)


@register_operator("batch_invariant.matmul")
def matmul_batch_invariant(input, other, *, impl="auto"):
    """Batched matrix multiplication with a batch-size-independent reduction.

    Leading batch dimensions broadcast; each result is bitwise identical alone
    or inside a larger batch. impl selects auto or batch_invariant_ops.
    """
    impl_fn = get_impl_by_name("batch_invariant.matmul", impl)
    return impl_fn(input, other)
