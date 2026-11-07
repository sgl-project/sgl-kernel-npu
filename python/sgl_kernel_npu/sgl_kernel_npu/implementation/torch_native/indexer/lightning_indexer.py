"""Independent FP32 reference: BSND, cmp_ratio=1, mask_mode=0/3.

Implements the formal source block-max/pinned-last-block and strict consumer
candidate mask from ops-transformer e7da18105f08b06e01ef0fff78f439098d483746.
This reference does not call the upstream bindings and is not a performance backend.
"""

import torch
import torch.nn.functional as F


def _scores(
    q,
    k,
    w,
    *,
    layout_q="BSND",
    layout_k="BSND",
    cmp_ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    mask_mode=0,
    **kwargs
):
    if layout_q != "BSND" or layout_k != "BSND" or cmp_ratio != 1:
        raise NotImplementedError(
            "torch_native supports BSND/BSND and cmp_ratio=1 only"
        )
    if (
        cu_seqlens_k is not None
        or block_table is not None
        or cmp_residual_k is not None
    ):
        raise NotImplementedError(
            "torch_native does not support packed/paged/compressed keys"
        )
    # B,Q,H,K -> B,Q,K, with one shared key head.
    dot = torch.einsum("bqhd,bkd->bqhk", q.float(), k[:, :, 0].float())
    score = (dot.relu() * w.float().unsqueeze(-1)).sum(dim=2)
    batch, nq, nk = score.shape
    qlen = torch.full((batch,), nq, device=q.device, dtype=torch.int64)
    if cu_seqlens_q is not None:
        qlen = cu_seqlens_q[1:].to(torch.int64)
    klen = torch.full((batch,), nk, device=q.device, dtype=torch.int64)
    if seqused_k is not None:
        klen = seqused_k.to(torch.int64)
    qi = torch.arange(nq, device=q.device)[None, :, None]
    ki = torch.arange(nk, device=q.device)[None, None, :]
    reachable = (qi < qlen[:, None, None]) & (ki < klen[:, None, None])
    if mask_mode == 3:
        reachable = reachable & (ki <= qi + (klen - qlen)[:, None, None])
    return score.masked_fill(~reachable, -torch.inf)


def _topk(score, topk, return_value, *, invalid_value=-torch.inf):
    width = min(topk, score.shape[-1])
    values, indices = score.topk(width, dim=-1)
    valid = values != -torch.inf
    indices = indices.to(torch.int32).masked_fill(~valid, -1)
    values = values.masked_fill(~valid, invalid_value)
    indices = F.pad(indices, (0, topk - width), value=-1).unsqueeze(-2)
    if return_value:
        values = F.pad(values, (0, topk - width), value=invalid_value).unsqueeze(-2)
    else:
        values = torch.empty(0, dtype=torch.float32, device=score.device)
    return indices, values


def lightning_indexer_v2(
    q,
    k,
    w,
    topk,
    *,
    return_value=0,
    candidate_topk_blocks=None,
    candidate_block_size=None,
    **kwargs
):
    score = _scores(q, k, w, **kwargs)
    result = _topk(score, topk, return_value)
    if candidate_topk_blocks is None:
        return result
    empty = torch.empty(0, dtype=torch.int32, device=q.device)
    if candidate_topk_blocks == -1:
        return (*result, empty, empty)
    block = 8 if candidate_block_size is None else candidate_block_size
    nk = score.shape[-1]
    padded = F.pad(score, (0, (-nk) % block), value=-torch.inf)
    block_scores = padded.reshape(*score.shape[:-1], -1, block).amax(dim=-1)
    # The last reachable block is pinned independently of its numeric score.
    lengths = (score != -torch.inf).sum(dim=-1)
    last = ((lengths - 1).clamp_min(0) // block).unsqueeze(-1)
    pinned = torch.full_like(last, torch.inf, dtype=torch.float32)
    pinned.masked_fill_(lengths.unsqueeze(-1) == 0, -torch.inf)
    block_scores.scatter_(-1, last, pinned)
    candidates, _ = _topk(block_scores, candidate_topk_blocks, 0)
    return (*result, candidates, empty)


def sparse_lightning_indexer(
    q,
    k,
    w,
    topk,
    candidate_block_indices,
    *,
    candidate_block_length=None,
    candidate_block_size=8,
    return_value=0,
    **kwargs
):
    score = _scores(q, k, w, **kwargs)
    blocks = torch.arange(score.shape[-1], device=q.device) // candidate_block_size
    candidates = candidate_block_indices[..., 0, :].long()
    num_blocks = (score.shape[-1] + candidate_block_size - 1) // candidate_block_size
    valid = (candidates >= 0) & (candidates < num_blocks)
    # Mark blocks once instead of materializing [B,Q,K,C] comparisons. Addition
    # preserves valid block zero when padded/duplicate candidates also occur.
    selected = torch.zeros(
        *score.shape[:-1], num_blocks, dtype=torch.int32, device=score.device
    )
    selected.scatter_add_(
        -1, candidates.clamp(0, num_blocks - 1), valid.to(torch.int32)
    )
    allowed = selected.index_select(-1, blocks) != 0
    score = score.masked_fill(~allowed, -torch.inf)
    return _topk(score, topk, return_value, invalid_value=0.0)
