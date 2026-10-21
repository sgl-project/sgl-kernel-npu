"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch


def ceil_align(n, a):
    return (n + a - 1) // a * a


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor
    window_index: torch.Tensor


class BuildRaggedVerifyWindow:

    @classmethod
    def torch(
        cls,
        *,
        batch: Any,
        layout: Any,
        draft_block_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        bs: int,
        device: str,
        verify_num_draft_tokens: int,
        model_runner,
    ) -> RaggedVerifyWindow:
        return build_ragged_verify_window(
            batch=batch,
            layout=layout,
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            bs=bs,
            device=device,
            verify_num_draft_tokens=verify_num_draft_tokens,
            model_runner=model_runner,
        )


def build_ragged_verify_window(
    *,
    batch: Any,
    layout: Any,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    bs: int,
    device: str,
    verify_num_draft_tokens: int,
    model_runner,
) -> RaggedVerifyWindow:
    prefix_lens = batch.seq_lens
    verify_lens = layout.verify_lens.to(device=device, dtype=torch.int32)
    padded_total = layout.graph_num_tokens
    req_id, within, valid = compact_row_index(
        verify_lens=verify_lens, padded_total=padded_total, device=device
    )
    safe_req = req_id.clamp(max=bs - 1)
    positions = torch.where(
        valid, prefix_lens.to(torch.int64)[safe_req] + within, torch.zeros_like(within)
    )
    real_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=model_runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + verify_lens.to(prefix_lens.dtype),
        batch_size=bs,
        draft_token_num=verify_num_draft_tokens,
        device=device,
    )
    verify_cache_loc = torch.nn.functional.pad(
        real_cache_loc, (0, padded_total - real_cache_loc.shape[0])
    )
    verify_cache_loc = torch.where(
        valid, verify_cache_loc, torch.zeros_like(verify_cache_loc)
    )
    verify_ids = compact_verify_ids(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    return RaggedVerifyWindow(
        positions=positions,
        verify_cache_loc=verify_cache_loc,
        verify_ids=verify_ids,
        window_index=torch.where(
            valid, safe_req.to(torch.int64) * verify_num_draft_tokens + within, -1
        ),
    )


def compact_verify_ids(
    *,
    draft_block_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    layout: Any,
    device: str,
) -> torch.Tensor:
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    bs = layout.verify_lens.shape[0]
    safe_req = req_id.clamp(max=bs - 1)
    anchors = draft_block_ids[:, 0]
    drafts = draft_tokens[safe_req, (within - 1).clamp_min(0)]
    verify_ids = torch.where(within == 0, anchors[safe_req], drafts)
    verify_ids = torch.where(valid, verify_ids, torch.zeros_like(verify_ids))
    return verify_ids.to(torch.int64)


def compact_row_index(
    *, verify_lens: torch.Tensor, padded_total: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verify_lens = verify_lens.to(device=device, dtype=torch.int64)
    bs = int(verify_lens.numel())
    incl = torch.cumsum(verify_lens, dim=0)
    start = incl - verify_lens
    real_total = incl[-1]
    row = torch.arange(padded_total, device=device, dtype=torch.int64)
    valid = row < real_total
    req_id = torch.searchsorted(incl, row, right=True)
    req_id = torch.where(valid, req_id, torch.full_like(req_id, bs))
    within = torch.where(
        valid, row - start[req_id.clamp(max=bs - 1)], torch.zeros_like(row)
    )
    return (req_id, within, valid)


class ScatterCompactToStrided:

    @classmethod
    def torch(
        cls,
        *,
        compact: torch.Tensor,
        layout: Any,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )


def scatter_compact_to_strided(
    *,
    compact: torch.Tensor,
    layout: Any,
    fill_value: float,
    verify_num_draft_tokens: int,
) -> torch.Tensor:
    stride = verify_num_draft_tokens
    bs = layout.verify_lens.shape[0]
    dim = compact.shape[1]
    device = compact.device
    compact = compact[: layout.graph_num_tokens]
    strided = torch.full(
        (bs * stride + 1, dim), fill_value, dtype=compact.dtype, device=device
    )
    req_id, within, valid = compact_row_index(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    sink = bs * stride
    strided_pos = torch.where(
        valid, req_id.clamp(max=bs - 1) * stride + within, torch.full_like(within, sink)
    )
    strided.index_copy_(0, strided_pos, compact)
    return strided[: bs * stride]


class CommitInjectLayoutResult(msgspec.Struct):
    swa_loc: torch.Tensor
    positions: torch.Tensor


class BuildCommitInjectLayout:

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_pos_offsets: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        commit_lens: torch.Tensor,
        stride: int,
    ) -> CommitInjectLayoutResult:
        return build_commit_inject_layout(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )


def build_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
) -> CommitInjectLayoutResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(dtype=torch.int64)
    col = torch.arange(stride, device=device).view(1, -1)
    committed = col < commit_lens.to(torch.long).view(-1, 1)
    req_rows = req_pool_indices.to(torch.long).view(-1, 1).expand(bs, stride)
    rows = torch.where(committed, req_rows, 0)
    cols = torch.where(committed, positions_2d.long(), 0)
    cache_loc = torch.where(committed, req_to_token[rows, cols].long(), 0)
    swa_loc = torch.where(committed, full_to_swa_mapping[cache_loc], -1)
    return CommitInjectLayoutResult(
        swa_loc=swa_loc.to(torch.int32).reshape(-1), positions=positions
    )


class BuildOutTokens:

    @classmethod
    def torch(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        return build_out_tokens(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=verify_num_draft_tokens,
            gamma=gamma,
        )


def build_out_tokens(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    out_tokens = torch.empty(
        (bs, verify_num_draft_tokens), dtype=torch.int64, device=draft_tokens.device
    )
    out_tokens[:, :gamma].copy_(draft_tokens)
    out_tokens[:, gamma].fill_(0)
    out_tokens.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out_tokens


def assign_extend_cache_locs_func(
    *,
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size,
    draft_token_num,
    device,
):
    chunks = [
        req_to_token[
            int(req_pool_indices[i]), int(start_offset[i]) : int(end_offset[i])
        ]
        for i in range(batch_size)
    ]
    return torch.cat(chunks).to(device=device, dtype=torch.int32)
