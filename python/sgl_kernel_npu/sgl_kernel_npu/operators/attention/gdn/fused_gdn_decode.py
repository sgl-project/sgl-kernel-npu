"""Public fused single-token gated delta-rule update."""

from ...utils.registry import get_impl_by_name, register_impl, register_operator

_triton_fn = None
_torch_native_fn = None


@register_impl("attention.gdn.fused_gdn_decode", "auto")
@register_impl("attention.gdn.fused_gdn_decode", "triton")
def _triton(*args, **kwargs):
    global _triton_fn
    if _triton_fn is None:
        from ....implementation.triton.attention.gdn.fused_gdn_decode import (
            fused_sigmoid_gating_delta_rule_update_decode_npu,
        )

        _triton_fn = fused_sigmoid_gating_delta_rule_update_decode_npu
    return _triton_fn(*args, **kwargs)


@register_impl("attention.gdn.fused_gdn_decode", "torch_native")
def _native(*args, **kwargs):
    global _torch_native_fn
    if _torch_native_fn is None:
        from ....implementation.torch_native.attention.gdn.fused_gdn_decode import (
            fused_gdn_decode,
        )

        _torch_native_fn = fused_gdn_decode
    return _torch_native_fn(*args, **kwargs)


@register_operator("attention.gdn.fused_gdn_decode")
def fused_gdn_decode(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    *,
    impl="auto",
):
    """Update the selected recurrent state slots and return one output per token.

    q and k are [B,T,H,K], v and output are [B,T,HV,V]. State is addressed
    in [slot,HV,V,K] order and updated in place when contiguous. auto is Triton.
    """
    return get_impl_by_name("attention.gdn.fused_gdn_decode", impl)(
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        initial_state_source,
        initial_state_indices,
        scale,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
