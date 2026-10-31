"""Public CGDR dispatch for the independent attention CANN operator."""

from ..utils.generation import Generation
from ..utils.registry import get_impl_by_name, register_impl, register_operator

_torch_native_fn = None
_GENERATIONS = (Generation.ARCH22, Generation.ARCH35)


@register_impl(
    "attention.chunk_gated_delta_rule", "auto", supported_generations=_GENERATIONS
)
@register_impl(
    "attention.chunk_gated_delta_rule", "ascendc", supported_generations=_GENERATIONS
)
def _ascendc_impl(
    query, key, value, beta, initial_state, actual_seq_lengths, scale, g, chunk_state
):
    import torch

    return torch.ops.npu.sgl_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        scale,
        g,
        chunk_state,
    )


@register_impl("attention.chunk_gated_delta_rule", "torch_native")
def _torch_native_impl(
    query, key, value, beta, initial_state, actual_seq_lengths, scale, g, chunk_state
):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ...implementation.torch_native.attention.chunk_gated_delta_rule import (
            chunk_gated_delta_rule,
        )

        _torch_native_fn = chunk_gated_delta_rule
    return _torch_native_fn(
        query,
        key,
        value,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        scale=scale,
        g=g,
        chunk_state=chunk_state,
    )


@register_operator("attention.chunk_gated_delta_rule")
def chunk_gated_delta_rule(
    query,
    key,
    value,
    *,
    beta=None,
    initial_state=None,
    actual_seq_lengths=None,
    scale=None,
    g=None,
    chunk_state=None,
    impl="auto",
):
    """Return (out, final_state); auto selects ascendc without fallback.

    scale=None means 1.0; explicit zero is preserved. beta, initial_state and
    actual_seq_lengths are required. State orientation is [B, Nv, Dv, Dk].
    A contiguous chunk_state receives entering states. Noncontiguous chunk_state
    uses a temporary with no copyback, preserving the old Host contract.
    """
    effective_scale = 1.0 if scale is None else scale
    return get_impl_by_name("attention.chunk_gated_delta_rule", impl)(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        effective_scale,
        g,
        chunk_state,
    )
