"""Shared eager paged sparse-attention math."""

import torch

from .metadata import ReferenceMetadata


def _fp8_to_float(value):
    if str(value.dtype).startswith("torch.float8_") and value.device.type != "cpu":
        return value.view(torch.uint8).cpu().view(value.dtype).float().to(value.device)
    return value.float()


def unpack_kv(cache):
    rows = cache.view(torch.uint8).reshape(-1, cache.shape[-1])
    if rows.shape[-1] not in (608, 640):
        raise NotImplementedError("reference supports 608B and cann_9_1 640B KV rows")
    rope = rows[:, :128].contiguous().view(torch.bfloat16).float()
    nope = _fp8_to_float(
        rows[:, 128:576].contiguous().view(torch.float8_e4m3fn)
    ).reshape(-1, 7, 64)
    if rows.shape[-1] == 640:
        exponent = rows[:, 576:583].to(torch.int32) - 127
        scales = torch.pow(2.0, exponent.float()).reshape(-1, 7, 1)
    else:
        scales = (
            rows[:, 576:590].contiguous().view(torch.bfloat16).float().reshape(-1, 7, 1)
        )
    return torch.cat(((nope * scales).flatten(1), rope), -1)


def _paged_rows(cache, table, batch, positions, quantized):
    slots = (
        table[batch, positions // cache.shape[1]].long() * cache.shape[1]
        + positions % cache.shape[1]
    )
    rows = unpack_kv(cache) if quantized else cache.reshape(-1, cache.shape[-1]).float()
    return rows[slots]


def _attention(
    q,
    *,
    ori_kv,
    ori_block_table,
    seqused_ori_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    cmp_kv=None,
    cmp_block_table=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_sparse_indices=None,
    ori_topk_length=None,
    cmp_sparse_indices=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    quant_mode=None,
    rope_head_dim=64,
):
    if layout_q not in ("TND", "BSND") or layout_kv != "PA_BBND":
        raise NotImplementedError("reference supports TND/BSND queries with PA_BBND KV")
    if ori_kv.shape[-2] != 1 or (cmp_kv is not None and cmp_kv.shape[-2] != 1):
        raise NotImplementedError("reference supports one shared KV head")
    if ori_mask_mode not in (0, 4) or cmp_mask_mode not in (0, 3):
        raise NotImplementedError(
            "reference supports explicit/SWA original KV and causal compressed KV"
        )
    if metadata is not None and not isinstance(metadata, ReferenceMetadata):
        raise TypeError("torch_native requires torch_native metadata")
    quantized = quant_mode is not None
    if quantized and (quant_mode != 1 or rope_head_dim != 64):
        raise NotImplementedError("reference supports quant_mode=1, rope_head_dim=64")
    query = q.reshape(-1, q.shape[-2], q.shape[-1])
    batch = len(seqused_ori_kv)
    if cu_seqlens_q is not None:
        boundaries = cu_seqlens_q.tolist()
    elif layout_q == "BSND":
        boundaries = list(range(0, (batch + 1) * q.shape[1], q.shape[1]))
    else:
        boundaries = None
    if boundaries is None:
        if seqused_q is None:
            raise ValueError("TND queries need cu_seqlens_q or seqused_q")
        boundaries = [0] + seqused_q.cumsum(0).tolist()
    out = torch.zeros_like(query)
    lse = torch.full(query.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)
    for b in range(batch):
        length = int(seqused_ori_kv[b])
        qlen = boundaries[b + 1] - boundaries[b]
        for t in range(boundaries[b], boundaries[b + 1]):
            pos = length - qlen + t - boundaries[b]
            if ori_sparse_indices is not None:
                if ori_topk_length is not None and int(ori_topk_length[t, 0]) == 0:
                    continue
                positions = ori_sparse_indices[t, 0].long()
                if ori_topk_length is not None:
                    positions = positions[: int(ori_topk_length[t, 0])]
                positions = positions[positions >= 0]
            else:
                positions = torch.arange(
                    max(0, pos - ori_win_left),
                    min(length, pos + ori_win_right + 1),
                    device=q.device,
                )
            rows = [_paged_rows(ori_kv, ori_block_table, b, positions, quantized)]
            if cmp_kv is not None:
                visible = min(int(seqused_cmp_kv[b]), (pos + 1) // cmp_ratio)
                if cmp_sparse_indices is None:
                    selected = torch.arange(visible, device=q.device)
                else:
                    selected = cmp_sparse_indices[t, 0].long()
                    selected = selected[(selected >= 0) & (selected < visible)]
                rows.append(
                    _paged_rows(cmp_kv, cmp_block_table, b, selected, quantized)
                )
            kv = torch.cat(rows, 0)
            if kv.shape[0] == 0:
                continue
            scores = query[t].float() @ kv.T * softmax_scale
            if sinks is not None:
                sink = sinks.float().reshape(-1, 1).expand(query.shape[1], 1)
                scores = torch.cat((scores, sink), -1)
            weights = scores.softmax(-1)[..., : kv.shape[0]]
            out[t] = (weights @ kv).to(q.dtype)
            lse[t] = scores.logsumexp(-1)
    return out.reshape_as(q), (
        lse if return_softmax_lse else torch.empty(0, device=q.device)
    )


def sparse_flash_mla(q, **kwargs):
    return _attention(q, **kwargs)


def mixed_quant_sparse_flash_mla(q, **kwargs):
    return _attention(q, **kwargs)
