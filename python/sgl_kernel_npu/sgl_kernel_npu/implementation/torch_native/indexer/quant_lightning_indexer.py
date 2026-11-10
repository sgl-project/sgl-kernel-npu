"""Independent eager reference for quant_lightning_indexer."""

import torch

from .quant_lightning_indexer_metadata import ReferenceMetadata


def _fp8_to_float(value):
    if str(value.dtype).startswith("torch.float8_") and value.device.type != "cpu":
        return value.view(torch.uint8).cpu().view(value.dtype).float().to(value.device)
    return value.float()


def quant_lightning_indexer(
    q,
    k,
    weights,
    q_scale,
    k_scale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    metadata=None,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=4,
    return_value=0,
):
    if (
        layout_q != "TND"
        or layout_k != "PA_BBND"
        or mask_mode != 3
        or quant_mode not in (1, 2)
    ):
        raise NotImplementedError(
            "reference supports causal TND/PA_BBND FP8 or INT8 indexing"
        )
    if metadata is not None and not isinstance(metadata, ReferenceMetadata):
        raise TypeError("torch_native requires torch_native metadata")
    if k.shape[-2] != 1:
        raise NotImplementedError("reference supports one shared key head")
    query = _fp8_to_float(q) * q_scale.float().unsqueeze(-1)
    keys = _fp8_to_float(k).reshape(-1, k.shape[-1]) * k_scale.float().reshape(-1, 1)
    indices = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32, device=q.device)
    values = torch.full((q.shape[0], 1, topk), -torch.inf, device=q.device)
    cu = cu_seqlens_q.tolist()
    for b in range(len(seqused_k)):
        qlen = cu[b + 1] - cu[b]
        full_len = int(seqused_k[b]) * cmp_ratio + int(cmp_residual_k[b])
        for t in range(cu[b], cu[b + 1]):
            visible = min(
                int(seqused_k[b]), (full_len - qlen + t - cu[b] + 1) // cmp_ratio
            )
            if visible <= 0:
                continue
            logical = torch.arange(visible, device=q.device)
            slots = (
                block_table[b, logical // k.shape[1]].long() * k.shape[1]
                + logical % k.shape[1]
            )
            logits = (query[t] @ keys[slots].T).relu()
            scores = (logits * weights[t].float().unsqueeze(-1)).sum(0)
            count = min(topk, visible)
            vals, ids = scores.topk(count)
            indices[t, 0, :count] = ids.to(torch.int32)
            values[t, 0, :count] = vals
    return indices, values if return_value else torch.empty(0, device=q.device)
