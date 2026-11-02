"""Released torch_npu bindings used by DSV4."""

import torch


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
):
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
    weights, indices, scores = torch.ops.npu.npu_moe_gating_top_k(x, k, **kwargs)
    if not out_flag:
        scores = torch.empty(0, dtype=torch.float32, device=x.device)
    return weights, indices, scores
