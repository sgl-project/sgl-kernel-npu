"""Public fused single-token gated delta-rule update."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("attention.fused_gdn_decode", "auto")
@register_impl("attention.fused_gdn_decode", "triton")
def _triton(*args, **kwargs):
    from ...implementation.triton.attention.fused_gdn_decode import (
        fused_sigmoid_gating_delta_rule_update_decode_npu,
    )

    return fused_sigmoid_gating_delta_rule_update_decode_npu(*args, **kwargs)


@register_impl("attention.fused_gdn_decode", "torch_native")
def _native(*args, **kwargs):
    from ...implementation.torch_native.attention.fused_gdn_decode import (
        fused_gdn_decode,
    )

    return fused_gdn_decode(*args, **kwargs)


@register_operator("attention.fused_gdn_decode")
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
    return get_impl_by_name("attention.fused_gdn_decode", impl)(
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
