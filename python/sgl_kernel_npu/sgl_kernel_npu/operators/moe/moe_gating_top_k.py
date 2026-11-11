"""Public moe_gating_top_k operator with registered implementations."""

import torch

from ..utils.cann import detect_cann_version
from ..utils.registry import get_impl_by_name, register_impl, register_operator


def _filter_moe_gating_kwargs(op, kwargs, *, defaults):
    """Omit only unused defaults absent from the original MoE schema."""
    if not isinstance(op, torch._ops.OpOverloadPacket):
        return kwargs
    names = {arg.name for arg in op.default._schema.arguments}
    result = {}
    for name, value in kwargs.items():
        if name in names:
            result[name] = value
        elif value is None or (
            name in defaults
            and not isinstance(value, torch.Tensor)
            and value == defaults[name]
        ):
            continue
        else:
            raise RuntimeError(
                f"{op} does not support requested argument {name}={value!r}"
            )
    return result


@register_impl("moe.moe_gating_top_k", "auto")
def _auto(x, k, **kwargs):
    cann_version = detect_cann_version()
    if cann_version.startswith("9.1."):
        backend = "cann_recipes_infer_cann910"
    elif cann_version.startswith("9.2."):
        backend = "cann_ops_transformers_cann920"
    else:
        raise RuntimeError(
            f"Unsupported CANN release {cann_version}; expected CANN 9.1.x or 9.2.x"
        )
    return get_impl_by_name("moe.moe_gating_top_k", backend)(x, k, **kwargs)


@register_impl("moe.moe_gating_top_k", "cann_ops_transformers_cann920")
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
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.2."):
        raise RuntimeError(
            f"This implementation requires CANN 9.2.x, got {cann_version}"
        )
    op = getattr(torch.ops.npu, "npu_moe_gating_top_k", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator npu.npu_moe_gating_top_k in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if isinstance(op, torch._ops.OpOverloadPacket):
        names = {arg.name for arg in op.default._schema.arguments}
        missing = (
            set(
                ("input_ids", "tid2eid")
                if input_ids is not None or tid2eid is not None
                else ()
            )
            - names
        )
        if missing:
            raise RuntimeError(
                f"{op} is missing arguments: {', '.join(sorted(missing))}"
            )

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
    weights, indices, scores = op(x, k, **kwargs)
    if not out_flag:
        scores = torch.empty(0, dtype=torch.float32, device=x.device)
    return weights.to(x.dtype), indices.to(torch.int32), scores


@register_impl("moe.moe_gating_top_k", "cann_recipes_infer_cann910")
def _cann_9_1(
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
    cann_version = detect_cann_version()
    if not cann_version.startswith("9.1."):
        raise RuntimeError(
            f"This implementation requires CANN 9.1.x, got {cann_version}"
        )
    from ..._third_party import sgl_cann_recipes_infer  # noqa: F401

    op = getattr(torch.ops.sgl_cann_recipes_infer, "npu_moe_gating_top_k", None)
    if op is None:
        raise RuntimeError(
            f"Missing operator sgl_cann_recipes_infer.npu_moe_gating_top_k in the installed sgl-kernel-npu wheel for CANN {cann_version}"
        )
    if isinstance(op, torch._ops.OpOverloadPacket):
        names = {arg.name for arg in op.default._schema.arguments}
        missing = (
            set(
                ("input_ids", "tid2eid")
                if input_ids is not None or tid2eid is not None
                else ()
            )
            - names
        )
        if missing:
            raise RuntimeError(
                f"{op} is missing arguments: {', '.join(sorted(missing))}"
            )
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
    kwargs = _filter_moe_gating_kwargs(
        op,
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
    weights, indices, scores = op(x, k, **kwargs)
    return (
        weights.to(x.dtype),
        indices.to(torch.int32),
        (
            scores.to(torch.float32)
            if out_flag
            else torch.empty(0, dtype=torch.float32, device=x.device)
        ),
    )


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
    impl selects auto, cann_recipes_infer_cann910,
    cann_ops_transformers_cann920, or torch_native.
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
