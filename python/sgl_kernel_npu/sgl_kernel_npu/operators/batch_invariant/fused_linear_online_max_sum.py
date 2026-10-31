"""Public batch-invariant fused linear + online max/sum operator."""

import torch
from sgl_kernel_npu.operators.utils.registry import (
    get_impl_by_name,
    register_impl,
    register_operator,
)

_batch_invariant_ops = None


# The external extension is imported lazily on the first call.
@register_impl("batch_invariant.fused_linear_online_max_sum", "auto")
@register_impl("batch_invariant.fused_linear_online_max_sum", "batch_invariant_ops")
def _batch_invariant_ops_impl(
    input, weight, target, vocab_start_index, vocab_end_index, return_logits
):
    global _batch_invariant_ops
    if _batch_invariant_ops is None:
        import batch_invariant_ops  # noqa: F401

        _batch_invariant_ops = torch.ops.batch_invariant_ops

    return _batch_invariant_ops.npu_fused_linear_online_max_sum_batch_invariant(
        input, weight, target, vocab_start_index, vocab_end_index, return_logits
    )


@register_operator("batch_invariant.fused_linear_online_max_sum")
def fused_linear_online_max_sum_batch_invariant(
    input,
    weight,
    target,
    vocab_start_index,
    vocab_end_index,
    return_logits=False,
    *,
    impl="auto",
):
    """Local logits plus online max/sum statistics for vocab-parallel CE.

    Each token's statistics are bitwise identical alone or inside a larger
    batch. Returns the row max, the sum of exp, the target logits, the packed
    target mask, the masked target and the logits when return_logits is True.
    impl selects auto or batch_invariant_ops.
    """
    impl_fn = get_impl_by_name("batch_invariant.fused_linear_online_max_sum", impl)
    return impl_fn(
        input, weight, target, vocab_start_index, vocab_end_index, return_logits
    )
