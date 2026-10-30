"""PyTorch reference for the decode score/topk operator.

Only the score-only branch (``disable_index_value=True``) is implemented; the
attention-output branch has no reference. Within the score-only branch it
mirrors the Triton semantics: ``score_type`` ``"max"``/``"lse"``,
``init_blocks``/``local_blocks`` forcing, direct page source or
``block_table``, ``packed_seq_lens``, ``sanitize_page_ids`` and
``fused_append_local``.

The whole KV range of a row is processed in one pass, so memory and latency
grow linearly with the context length; this module exists for correctness
checks.
"""

from __future__ import annotations

from typing import Optional

import torch

# Same constant as the kernels' sm_scale * log2(e).
_LOG2E = 1.4426950409


def _blocks_from_request_map(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    batch_size: int,
    block_size: int,
    num_blocks: int,
    num_pages: int,
    sanitize_page_ids: bool,
) -> torch.Tensor:
    """Return physical block ids ``[B, num_blocks]`` for the direct page map.

    The kernel probes the request map at the first token of each logical block
    and divides the token slot by ``block_size``; ``sanitize_page_ids`` clamps
    the result to the page-table range.
    """
    rows = req_pool_indices[:batch_size].to(torch.long)
    cols = torch.arange(num_blocks, device=req_to_token.device) * block_size
    cols = cols.clamp(max=req_to_token.shape[1] - 1)
    slots = req_to_token.index_select(0, rows).index_select(1, cols).to(torch.long)
    physical = slots // block_size
    if sanitize_page_ids:
        physical = physical.clamp(0, num_pages - 1)
    return physical


def _blocks_from_page_table(
    block_table: torch.Tensor, batch_size: int, num_blocks: int
) -> torch.Tensor:
    """Return logical block ids ``[B, num_blocks]`` from a materialized table."""
    return block_table[:batch_size, :num_blocks].to(torch.long)


def _block_scores(qk: torch.Tensor, block_size: int, score_type: str) -> torch.Tensor:
    """Reduce per-token log2-domain scores ``[B, KVH, G, L]`` to block scores.

    ``L`` is a multiple of ``block_size``; positions beyond a row's causal
    length are already ``-inf``, so an empty block reduces to ``-inf``.
    """
    batch_size, num_kv_heads, gqa, length = qk.shape
    blocks = length // block_size
    qk_blocks = qk.reshape(batch_size, num_kv_heads, gqa, blocks, block_size)
    block_max = qk_blocks.amax(dim=-1)
    if score_type == "max":
        return block_max
    summed = torch.exp2(qk_blocks - block_max.unsqueeze(-1)).sum(dim=-1)
    scores = block_max + torch.log2(summed)
    # An all-masked block makes 0/0 -> NaN in the kernel; it normalizes to -inf.
    return torch.where(
        torch.isnan(scores), torch.full_like(scores, float("-inf")), scores
    )


def _select_topk(
    scores: torch.Tensor,
    num_blocks: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Pick ``topk`` block ids per row, ordered by (score desc, block asc).

    ``scores`` is ``[B, KVH, G, N]`` and ``num_blocks`` is ``[B, G]``. Slots
    beyond a row's block count repeat block 0, matching the pass-through
    branch; an empty row yields ``-1`` throughout.
    """
    if scores.shape[-1] == 0 or topk <= 0:
        return torch.full(
            (*scores.shape[:-1], max(topk, 0)),
            -1,
            dtype=torch.int32,
            device=scores.device,
        )
    countable = num_blocks.unsqueeze(1).expand(-1, scores.shape[1], -1)
    order = torch.argsort(-scores, dim=-1, stable=True)
    ranks = torch.arange(topk, device=scores.device)
    picked = order[..., :topk]
    if picked.shape[-1] < topk:
        pad = picked[..., :1].expand(*picked.shape[:-1], topk - picked.shape[-1])
        picked = torch.cat([picked, pad], dim=-1)
    valid = ranks < countable.unsqueeze(-1)
    pad = torch.where(
        (countable > 0).unsqueeze(-1),
        torch.zeros_like(order[..., :1]),
        torch.full_like(order[..., :1], -1),
    )
    picked = torch.where(valid, picked, pad.expand_as(picked))
    return picked.to(torch.int32)


def flash_decode_bnsd_with_topk_idx(
    q: torch.Tensor,
    k_cache_bnsd: torch.Tensor,
    block_table: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = True,
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    max_num_blocks: Optional[int] = None,
    num_pages: Optional[int] = None,
    sanitize_page_ids: bool = False,
    packed_seq_lens: bool = False,
    fused_append_local: bool = False,
) -> tuple[None, torch.Tensor]:
    """Reference implementation of the score-only branch.

    See the Triton implementation for the full contract. This reference covers
    ``disable_index_value=True`` with both score reductions (``max``/``lse``),
    ``init_blocks``/``local_blocks`` forcing, the direct page source and
    ``block_table``, ``packed_seq_lens``, ``sanitize_page_ids`` and
    ``fused_append_local``.

    The attention-output branch (``disable_index_value=False``) has no
    reference implementation and is rejected. ``topk <= 0`` returns the
    documented empty index tensor. The order of equally scored blocks is
    undefined; tie-free test data is expected for exact ``topk_idx``.
    """
    assert disable_index_value, (
        "torch_native implements the score-only branch only; the "
        "attention-output branch has no reference implementation"
    )
    assert score_type in (
        "max",
        "lse",
    ), f"score_type must be 'max' or 'lse', got {score_type!r}"

    assert q.dtype in (torch.float16, torch.bfloat16)
    assert k_cache_bnsd.dtype == q.dtype
    assert seq_lens.dtype in (torch.int32, torch.int64)
    assert seq_lens.ndim == 1, f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}"

    use_direct_page_lookup = req_to_token is not None
    assert (req_pool_indices is not None) == use_direct_page_lookup
    if use_direct_page_lookup:
        assert block_table is None
        assert req_to_token.ndim == 2
        assert req_to_token.dtype in (torch.int32, torch.int64)
        assert req_pool_indices.ndim == 1
        assert req_pool_indices.dtype in (torch.int32, torch.int64)
        assert max_num_blocks is not None and max_num_blocks > 0
        assert num_pages is not None and num_pages > 0
    else:
        assert block_table is not None
        assert block_table.dtype in (torch.int32, torch.int64)
        assert (
            block_table.ndim == 2
        ), f"block_table must be 2D, got shape={tuple(block_table.shape)}"

    assert q.ndim == 3, f"q must be [B, QH, D], got shape={tuple(q.shape)}"
    batch_size, num_q_heads, head_dim = q.shape
    assert k_cache_bnsd.ndim == 4, (
        "k_cache_bnsd must be [num_blocks, block_size, KVH, D], "
        f"got shape={tuple(k_cache_bnsd.shape)}"
    )
    _, block_size_from_cache, num_kv_heads, cache_head_dim = k_cache_bnsd.shape

    assert num_q_heads > 0 and num_kv_heads > 0, (
        f"num_q_heads and num_kv_heads must be positive, got "
        f"{num_q_heads} and {num_kv_heads}"
    )
    assert head_dim > 0, f"head_dim must be positive, got {head_dim}"
    assert block_size > 0, f"block_size must be positive, got {block_size}"

    assert block_size_from_cache == block_size
    assert cache_head_dim == head_dim
    assert num_q_heads % num_kv_heads == 0
    gqa = num_q_heads // num_kv_heads

    if packed_seq_lens:
        assert (
            num_kv_heads == 1
        ), f"packed_seq_lens requires num_kv_heads == 1, got {num_kv_heads}"
        assert seq_lens.shape[0] == batch_size * gqa
        lengths = seq_lens.reshape(batch_size, gqa).to(torch.int64)
    else:
        assert seq_lens.shape[0] == batch_size
        lengths = (
            seq_lens.reshape(batch_size, 1).expand(batch_size, gqa).to(torch.int64)
        )

    if use_direct_page_lookup:
        assert req_pool_indices.shape[0] == batch_size
        assert max_num_blocks * block_size <= req_to_token.shape[1]
    else:
        assert block_table.shape[0] == batch_size

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    if topk <= 0:
        return None, torch.empty(
            (num_q_heads, batch_size, 0), dtype=torch.int32, device=q.device
        )

    num_blocks = (lengths + block_size - 1) // block_size  # [B, G]
    max_blocks = int(num_blocks.max().item()) if num_blocks.numel() else 0
    if max_blocks == 0:
        columns = topk + (1 if fused_append_local else 0)
        empty = torch.full(
            (batch_size, num_q_heads, columns),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        if fused_append_local:
            assert max_num_blocks is not None and max_num_blocks > 0, (
                "fused_append_local requires max_num_blocks > 0, got "
                f"{max_num_blocks}"
            )
            # Every row is empty, so the appended causal local block is block 0.
            empty[..., -1] = 0
        return None, empty.permute(1, 0, 2).contiguous()

    if use_direct_page_lookup:
        policy_blocks = _blocks_from_request_map(
            req_to_token,
            req_pool_indices,
            batch_size,
            block_size,
            max_blocks,
            int(num_pages),
            sanitize_page_ids,
        )
    else:
        policy_blocks = _blocks_from_page_table(block_table, batch_size, max_blocks)

    # Columns beyond the group's block count are never scored, but their page
    # ids need not be valid; gather block 0 for them instead of failing.
    row_blocks = num_blocks.amax(dim=1, keepdim=True)  # [B, 1]
    block_cols = torch.arange(max_blocks, device=q.device).view(1, -1)
    policy_blocks = torch.where(
        block_cols < row_blocks, policy_blocks, torch.zeros_like(policy_blocks)
    )

    # Gather the KV blocks once: [B, max_blocks, block_size, KVH, D].
    length = max_blocks * block_size
    k_flat = k_cache_bnsd[policy_blocks].reshape(
        batch_size, length, num_kv_heads, head_dim
    )
    k_float = k_flat.permute(0, 2, 1, 3).float()  # [B, KVH, L, D]
    q_float = q.float().reshape(batch_size, num_kv_heads, gqa, head_dim)

    # Per-token scores in the kernels' log2 domain: [B, KVH, G, L].
    qk = torch.einsum("bkgd,bkld->bkgl", q_float, k_float)
    qk = qk * (sm_scale * _LOG2E)
    positions = torch.arange(length, device=q.device)
    token_valid = positions.view(1, 1, 1, -1) < lengths.reshape(batch_size, 1, gqa, 1)
    qk = qk.masked_fill(~token_valid, float("-inf"))

    scores = _block_scores(qk, block_size, score_type)  # [B, KVH, G, N]

    block_ids = torch.arange(max_blocks, device=q.device)
    available = block_ids.view(1, 1, 1, -1) < num_blocks.view(batch_size, 1, gqa, 1)
    if init_blocks > 0:
        is_init = (block_ids.view(1, 1, 1, -1) < init_blocks) & available
        scores = torch.where(is_init, torch.full_like(scores, 1e30), scores)
    if local_blocks > 0:
        local_start = (num_blocks - local_blocks).clamp(min=0)
        local_start_view = local_start.view(batch_size, 1, gqa, 1)
        is_local = (block_ids.view(1, 1, 1, -1) >= local_start_view) & available
        scores = torch.where(is_local, torch.full_like(scores, 1e29), scores)

    # The pass-through case (group-wide block count <= topk) scores every block
    # as 0 and skips the init/local rewrite (contract C4).
    pass_through = num_blocks.amax(dim=1, keepdim=True) <= topk  # [B, 1]
    scores = torch.where(
        pass_through.view(batch_size, 1, 1, 1), torch.zeros_like(scores), scores
    )
    scores = scores.masked_fill(~available, float("-inf"))

    picked = _select_topk(scores, num_blocks, topk)  # [B, KVH, G, topk]

    if fused_append_local:
        assert max_num_blocks is not None and max_num_blocks > 0, (
            "fused_append_local requires max_num_blocks > 0, got " f"{max_num_blocks}"
        )
        query_pos = (lengths - 1).clamp(min=0)  # [B, G]
        query_pos = query_pos.view(batch_size, 1, gqa, 1)
        local_blk = torch.minimum(
            query_pos.div(block_size, rounding_mode="floor"),
            torch.full_like(query_pos, max_num_blocks - 1),
        )
        valid = (
            (picked >= 0)
            & (picked < max_num_blocks)
            & (picked * block_size <= query_pos)
        )
        selected = torch.where(valid, picked, torch.full_like(picked, -1))
        local_hit = (selected == local_blk).any(dim=-1)  # [B, KVH, G]
        local_block = local_blk.squeeze(-1)
        appended = torch.where(
            local_hit,
            torch.full_like(local_block, -1),
            local_block,
        )
        picked = torch.cat([selected, appended.unsqueeze(-1)], dim=-1).to(torch.int32)

    topk_idx = picked.reshape(batch_size, num_q_heads, picked.shape[-1])
    return None, topk_idx.permute(1, 0, 2).contiguous()
