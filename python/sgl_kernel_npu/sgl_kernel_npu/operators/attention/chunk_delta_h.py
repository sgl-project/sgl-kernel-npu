"""Public chunked gated delta-rule state update."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.chunk_delta_h", "auto")
@register_impl("attention.chunk_delta_h", "triton")
def _triton(*args, **kwargs):
    from ...implementation.triton.attention.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_npu,
    )

    return chunk_gated_delta_rule_fwd_h_npu(*args, **kwargs)


@register_impl("attention.chunk_delta_h", "torch_native")
def _native(*args, **kwargs):
    from ...implementation.torch_native.attention.chunk_delta_h import chunk_delta_h

    return chunk_delta_h(*args, **kwargs)


@register_operator("attention.chunk_delta_h")
def chunk_delta_h(
    k,
    w,
    u,
    g=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    *,
    impl="auto",
):
    """Compute per-chunk state snapshots, residual values and optional final state.

    k: [B,T,Hg,K], w: [B,T,H,K], u: [B,T,H,V], g: [B,T,H].
    The existing Triton path requires g. auto selects Triton.
    """
    if g is None:
        raise ValueError("g is required by the current Triton wrapper")
    return get_impl_by_name("attention.chunk_delta_h", impl)(
        k,
        w,
        u,
        g,
        initial_state,
        output_final_state,
        chunk_size,
        save_new_value,
        cu_seqlens,
    )
