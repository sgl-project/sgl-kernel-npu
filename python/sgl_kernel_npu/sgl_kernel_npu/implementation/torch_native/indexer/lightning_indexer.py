"""Independent FP32 reference: BSND, cmp_ratio=1, mask_mode=0/3.

Implements the formal source block-max/pinned-last-block and strict consumer
candidate mask from ops-transformer e7da18105f08b06e01ef0fff78f439098d483746.
This reference does not call the upstream bindings and is not a performance backend.
"""

from itertools import pairwise

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


def lightning_indexer(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query=None,
    actual_seq_lengths_key=None,
    block_table=None,
    layout_query="BSND",
    layout_key="BSND",
    sparse_count=2048,
    sparse_mode=3,
    pre_tokens=2**63 - 1,
    next_tokens=2**63 - 1,
    return_value=False
):
    """Reference for one shared key head in dense BSND or paged DSA layouts."""
    from ..attention.sparse_flash_attention import _paged_sequence

    q, k, w = (tensor.detach().cpu() for tensor in (query, key, weights))
    if sparse_mode not in (0, 3) or (pre_tokens, next_tokens) != (2**63 - 1, 2**63 - 1):
        raise NotImplementedError(
            "torch_native supports default windows in modes 0 and 3"
        )
    if layout_query == layout_key == "BSND":
        if k.size(-2) != 1:
            raise NotImplementedError("torch_native requires one key head")
        q_lengths = (
            None
            if actual_seq_lengths_query is None
            else F.pad(actual_seq_lengths_query.cpu(), (1, 0))
        )
        scores = _scores(
            q,
            k,
            w,
            cu_seqlens_q=q_lengths,
            seqused_k=(
                None if actual_seq_lengths_key is None else actual_seq_lengths_key.cpu()
            ),
            mask_mode=sparse_mode,
        )
        indices, scores = _topk(scores, sparse_count, return_value)
    elif layout_query in ("BSND", "TND") and layout_key == "PA_BSND":
        if block_table is None or actual_seq_lengths_key is None:
            raise ValueError(
                "Paged LightningIndexer requires a block table and KV lengths"
            )
        pages = block_table.detach().cpu()
        if k.size(-2) != 1:
            raise NotImplementedError("torch_native requires one key head")
        lengths = actual_seq_lengths_key.tolist()
        if layout_query == "TND":
            if actual_seq_lengths_query is None:
                raise ValueError("TND requires cumulative query lengths")
            spans = list(pairwise([0, *actual_seq_lengths_query.tolist()]))
        else:
            q_lengths = (
                [q.size(1)] * q.size(0)
                if actual_seq_lengths_query is None
                else actual_seq_lengths_query.tolist()
            )
            spans = [
                (i * q.size(1), i * q.size(1) + n) for i, n in enumerate(q_lengths)
            ]
        if len(spans) != len(lengths) or len(spans) != len(pages):
            raise ValueError("Query spans, KV lengths and page tables must match")
        q = q.reshape(-1, *q.shape[-2:])
        w = w.reshape(-1, w.size(-1))
        indices = torch.full(
            (q.size(0), 1, sparse_count), -1, dtype=torch.int32, device="cpu"
        )
        scores = (
            torch.full(indices.shape, -torch.inf, device="cpu")
            if return_value
            else q.new_empty(0)
        )
        for batch, ((left, right), length) in enumerate(zip(spans, lengths)):
            if left == right or length == 0:
                continue
            keys = _paged_sequence(k, pages[batch], length)
            score = _scores(
                q[left:right].unsqueeze(0),
                keys.unsqueeze(0),
                w[left:right].unsqueeze(0),
                mask_mode=sparse_mode,
            )
            selected, values = _topk(score, sparse_count, return_value)
            indices[left:right] = selected[0]
            if return_value:
                scores[left:right] = values[0]
        if layout_query == "BSND":
            indices = indices.reshape(*query.shape[:2], 1, sparse_count)
            if return_value:
                scores = scores.reshape_as(indices)
    else:
        raise NotImplementedError("torch_native supports dense BSND and PA_BSND keys")
    return indices.to(query.device), scores.to(device=query.device, dtype=query.dtype)
