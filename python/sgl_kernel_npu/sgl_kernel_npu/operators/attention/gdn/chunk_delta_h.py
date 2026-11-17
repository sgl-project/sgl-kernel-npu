"""Public chunked gated delta-rule state update."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.chunk_delta_h", "auto")
@register_impl("attention.gdn.chunk_delta_h", "triton")
def _triton(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.chunk_delta_h import (
            chunk_gated_delta_rule_fwd_h_npu,
        )

        _triton_fn = chunk_gated_delta_rule_fwd_h_npu
    return _triton_fn(*args, **kwargs)


@register_impl("attention.gdn.chunk_delta_h", "torch_native")
def _native(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.chunk_delta_h import (
            chunk_delta_h,
        )

        _torch_native_fn = chunk_delta_h
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.gdn.chunk_delta_h")
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
    Triton supports 1 <= K,V <= 128 and the validated chunk_size=64 only.
    """
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
        or chunk_size & (chunk_size - 1)
    ):
        raise ValueError("chunk_size must be a positive power of two")
    if impl in ("auto", "triton"):
        if not (1 <= k.shape[-1] <= 128 and 1 <= u.shape[-1] <= 128):
            raise ValueError("Triton chunk_delta_h requires 1 <= K,V <= 128")
        if chunk_size != 64:
            raise ValueError("Triton chunk_delta_h supports only chunk_size=64")
    if g is None:
        raise ValueError("g is required by the current Triton wrapper")
    return get_impl_by_name("attention.gdn.chunk_delta_h", impl)(
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
