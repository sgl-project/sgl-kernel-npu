"""MoE gate selection with explicit torch_npu and independent torch backends."""

from ..utils.registry import get_impl_by_name, register_impl, register_operator


@register_impl("dsv4.moe_gating_top_k", "auto")
@register_impl("dsv4.moe_gating_top_k", "torch_npu")
def _torch_npu(
    x,
    k,
    *,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    routed_scaling_factor=1.0,
    eps=1e-20,
):
    from ...implementation.torch_npu.dsv4 import moe_gating_top_k

    return moe_gating_top_k(
        x,
        k,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )


@register_impl("dsv4.moe_gating_top_k", "torch_native")
def _torch_native(
    x,
    k,
    *,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    routed_scaling_factor=1.0,
    eps=1e-20,
):
    from ...implementation.torch_native.dsv4.moe import moe_gating_top_k

    return moe_gating_top_k(
        x,
        k,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )


@register_operator("dsv4.moe_gating_top_k")
def moe_gating_top_k(
    x,
    k,
    *,
    bias=None,
    input_ids=None,
    tid2eid=None,
    k_group=1,
    group_count=1,
    group_select_mode=0,
    renorm=0,
    norm_type=0,
    out_flag=False,
    routed_scaling_factor=1.0,
    eps=1e-20,
    impl="auto",
):
    """Return (weights, int32 expert IDs, optional fp32 normalized scores).

    Weights have x's dtype. With out_flag=False, the third output is an empty
    fp32 tensor on x's device (the raw binding's unused output is unspecified).
    bias affects expert selection, not the gathered weights. The backend uses
    the released torch.ops.npu binding; auto aliases torch_npu.
    """
    return get_impl_by_name("dsv4.moe_gating_top_k", impl)(
        x,
        k,
        bias=bias,
        input_ids=input_ids,
        tid2eid=tid2eid,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )
