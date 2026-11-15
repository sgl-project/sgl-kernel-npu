"""Public batch-invariant fused residual-add + RMSNorm operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops_fn = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.add_rms_norm", "auto")
@register_impl("batch_invariant.add_rms_norm", "batch_invariant_ops")
def _batch_invariant_ops_impl(x, residual, weight, eps):
    global _batch_invariant_ops_fn
    if _batch_invariant_ops_fn is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops_fn = (
            torch.ops.batch_invariant_ops.npu_add_rms_norm_batch_invariant
        )

    return _batch_invariant_ops_fn(x, residual, weight, eps)


@register_operator("batch_invariant.add_rms_norm")
def add_rms_norm_batch_invariant(x, residual, weight, eps, *, impl="auto"):
    """Add residual to x, then RMS-normalize the last dimension.

    Each row is bitwise identical alone or inside a larger batch. Returns the
    normalized output, the reciprocal standard deviation and the residual sum.
    impl selects auto or batch_invariant_ops.
    """
    impl_fn = get_impl_by_name("batch_invariant.add_rms_norm", impl)
    return impl_fn(x, residual, weight, eps)
