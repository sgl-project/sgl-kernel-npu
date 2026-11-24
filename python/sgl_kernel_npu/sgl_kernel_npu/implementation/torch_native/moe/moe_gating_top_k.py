"""Independent eager reference for moe_gating_top_k."""

import torch
import torch.nn.functional as F


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
    if x.ndim != 2 or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("x must be a 2-D float32/float16/bfloat16 tensor")
    experts = x.shape[1]
    if group_count < 1 or experts % group_count or not 1 <= k_group <= group_count:
        raise ValueError(
            "group_count must divide experts; k_group must select valid groups"
        )
    group_width = experts // group_count
    if not 1 <= k <= group_width * k_group:
        raise ValueError("k exceeds selected expert capacity")
    if (
        norm_type not in (0, 1, 2)
        or group_select_mode not in (0, 1)
        or renorm not in (0, 1)
    ):
        raise ValueError("unsupported normalization, group selection or renorm")
    if group_count > 1 and group_select_mode == 1 and group_width < 2:
        raise ValueError("top-2 group selection needs at least two experts per group")
    if bias is not None and (
        bias.shape != (experts,) or bias.dtype != x.dtype or bias.device != x.device
    ):
        raise ValueError("bias must match x's expert dimension, dtype and device")
    scores = x.float()
    if norm_type == 0:
        scores = scores.softmax(-1)
    elif norm_type == 1:
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()
    if (input_ids is None) != (tid2eid is None):
        raise ValueError("input_ids and tid2eid must be provided together")
    if input_ids is not None:
        if k > 64 or (k_group != group_count and group_count != experts):
            raise NotImplementedError(
                "hash routing requires k<=64 and a simplified grouping path"
            )
        if (
            input_ids.shape != (x.shape[0],)
            or tid2eid.ndim != 2
            or tid2eid.shape[1] != k
        ):
            raise ValueError("hash inputs must be input_ids[rows] and tid2eid[keys,k]")
        if input_ids.dtype not in (torch.int32, torch.int64) or tid2eid.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("hash indices must be int32 or int64")
        if (input_ids < 0).any() or (input_ids >= tid2eid.shape[0]).any():
            raise ValueError("input_ids contains an invalid table row")
        indices = tid2eid[input_ids.long()].long()
        if (indices < 0).any() or (indices >= experts).any():
            raise ValueError("tid2eid contains an invalid expert ID")
    else:
        selection = scores if bias is None else scores + bias.float()
        if group_count > 1:
            grouped = selection.reshape(-1, group_count, group_width)
            group_scores = (
                grouped.amax(-1)
                if group_select_mode == 0
                else grouped.topk(2, dim=-1).values.sum(-1)
            )
            group_ids = group_scores.topk(k_group, dim=-1).indices
            keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
                -1, group_ids, True
            )
            selection = grouped.masked_fill(~keep.unsqueeze(-1), -torch.inf).reshape_as(
                selection
            )
        indices = selection.topk(k, dim=-1).indices
    weights = scores.gather(-1, indices)
    # Sigmoid and sqrt-softplus always normalize selected weights; renorm only controls softmax.
    if norm_type in (1, 2) or renorm:
        weights = weights / (weights.sum(-1, keepdim=True) + eps)
    weights = (weights * routed_scaling_factor).to(x.dtype)
    return (
        weights,
        indices.to(torch.int32),
        scores if out_flag else torch.empty(0, dtype=torch.float32, device=x.device),
    )
