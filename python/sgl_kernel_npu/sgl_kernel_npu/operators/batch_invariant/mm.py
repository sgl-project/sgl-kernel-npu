"""Public batch-invariant matrix multiplication operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops_fn = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.mm", "auto")
@register_impl("batch_invariant.mm", "batch_invariant_ops")
def _batch_invariant_ops_impl(input, mat2):
    global _batch_invariant_ops_fn
    if _batch_invariant_ops_fn is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops_fn = torch.ops.batch_invariant_ops.npu_mm_batch_invariant

    return _batch_invariant_ops_fn(input, mat2)


@register_operator("batch_invariant.mm")
def mm_batch_invariant(input, mat2, *, impl="auto"):
    """Multiply [N, K] by [K, M] with a batch-size-independent reduction.

    Each output row is bitwise identical alone or inside a larger batch.
    impl selects auto or batch_invariant_ops; both use the external library.
    """
    impl_fn = get_impl_by_name("batch_invariant.mm", impl)
    return impl_fn(input, mat2)
