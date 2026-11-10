"""Public dispatch for the sequence-first, decomposed GDN chunk pipeline."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None


@register_impl("attention.gdn.chunk_gated_delta_rule_npu", "auto")
@register_impl("attention.gdn.chunk_gated_delta_rule_npu", "triton")
def _triton_impl(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    head_first,
    use_qk_l2norm_in_kernel,
):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.chunk import (
            chunk_gated_delta_rule_npu,
        )

        _triton_fn = chunk_gated_delta_rule_npu
    return _triton_fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


@register_operator("attention.gdn.chunk_gated_delta_rule_npu")
def chunk_gated_delta_rule_npu(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
    head_first=False,
    use_qk_l2norm_in_kernel=False,
    *,
    impl="auto",
):
    """Return (output, final_state, entering_chunk_states); auto selects Triton.

    Preserves the fla.chunk wrapper's sequence-first contract and optional
    GDN_USE_MEGA_GDN path. This is distinct from the fused CANN public
    chunk_gated_delta_rule operator, which has a different signature and state
    orientation. The implementation normalizes contiguous inputs and selects
    their NPU device before execution. head_first=True remains unsupported.
    """
    return get_impl_by_name("attention.gdn.chunk_gated_delta_rule_npu", impl)(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
    )
