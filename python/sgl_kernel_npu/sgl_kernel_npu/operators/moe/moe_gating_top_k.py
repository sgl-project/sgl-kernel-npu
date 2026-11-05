"""Public moe_gating_top_k operator with registered implementations."""

import torch

from ..utils.registry import get_impl_by_name, register_impl, register_operator
from ..utils.transformer_compat import checked_kwargs, select_family


@register_impl("moe.moe_gating_top_k", "auto")
@register_impl("moe.moe_gating_top_k", "torch_npu")
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
    family = select_family(
        (("npu", "npu_moe_gating_top_k"),),
        (("custom", "npu_moe_gating_top_k"),),
        required_arguments=(
            ("input_ids", "tid2eid")
            if input_ids is not None or tid2eid is not None
            else ()
        ),
    )
    if family.cann_9_1:
        kwargs = dict(
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
        kwargs = checked_kwargs(
            family.ops[0],
            kwargs,
            defaults={
                "k_group": 1,
                "group_count": 1,
                "group_select_mode": 0,
                "renorm": 0,
                "norm_type": 0,
                "out_flag": False,
                "routed_scaling_factor": 1.0,
                "eps": 1e-20,
            },
        )
        weights, indices, scores = family.ops[0](x, k, **kwargs)
        return (
            weights.to(x.dtype),
            indices.to(torch.int32),
            (
                scores.to(torch.float32)
                if out_flag
                else torch.empty(0, dtype=torch.float32, device=x.device)
            ),
        )
    import torch_npu  # noqa: F401 -- registers torch.ops.npu

    kwargs = dict(
        bias=bias,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )
    if input_ids is not None or tid2eid is not None:
        kwargs.update(input_ids=input_ids, tid2eid=tid2eid)
    weights, indices, scores = family.ops[0](x, k, **kwargs)
    if not out_flag:
        scores = torch.empty(0, dtype=torch.float32, device=x.device)
    return weights.to(x.dtype), indices.to(torch.int32), scores


@register_impl("moe.moe_gating_top_k", "torch_native")
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
    from ...implementation.torch_native.moe.moe_gating_top_k import moe_gating_top_k

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


@register_operator("moe.moe_gating_top_k")
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
    bias affects expert selection, not the gathered weights. The production
    wrapper resolves the native or original custom binding before execution;
    the registered auto/torch_npu names stay stable.
    """
    if (input_ids is None) != (tid2eid is None):
        raise ValueError("input_ids and tid2eid must be provided together")
    return get_impl_by_name("moe.moe_gating_top_k", impl)(
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
