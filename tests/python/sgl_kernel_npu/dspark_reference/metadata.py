"""Torch oracles copied from SGLang 7036f4b0ad for migrated CUDA cases.

Reference code is test-only and never imports or calls NPU implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch


def ceil_align(n, a):
    return (n + a - 1) // a * a


class DsparkWindowGather(msgspec.Struct, frozen=True):
    num_q: int
    bs: int
    context_lens: torch.Tensor
    req_pool_indices_per_request: torch.Tensor
    offsets: torch.Tensor
    invalid: torch.Tensor


class ComputeDsparkWindowGather:

    @classmethod
    def torch(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
        valid_from: Optional[torch.Tensor] = None,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
            valid_from=valid_from,
        )


class BuildDsparkSwaPageIndices:

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_per_request: torch.Tensor,
        offsets: torch.Tensor,
        invalid: torch.Tensor,
        out_loc: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_dspark_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            invalid=invalid,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )


def compute_dspark_window_gather(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
    valid_from: Optional[torch.Tensor] = None,
) -> DsparkWindowGather:
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    num_q = seq_lens_casual.size(0)
    assert (
        num_q % block_size == 0
    ), f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not divisible by block_size={block_size}."
    bs = num_q // block_size
    device = seq_lens_casual.device
    first_token = torch.arange(bs, device=device, dtype=torch.int64) * block_size
    prefix_lens = (seq_lens_casual[first_token] - 1).to(torch.int32)
    context_lens = torch.clamp(prefix_lens, max=swa_window).to(torch.int32)
    req_pool_indices_per_request = req_pool_indices_repeated[first_token]
    if valid_from is not None:
        rows_written = prefix_lens - valid_from[req_pool_indices_per_request.long()]
        context_lens = torch.minimum(context_lens, rows_written.clamp(min=0))
        context_lens = context_lens.to(torch.int32)
    offsets = (
        prefix_lens.to(torch.int64).unsqueeze(1)
        - swa_window
        + torch.arange(swa_window, device=device, dtype=torch.int64).unsqueeze(0)
    )
    invalid = offsets < 0
    offsets = offsets.clamp(min=0)
    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_indices_per_request,
        offsets=offsets,
        invalid=invalid,
    )


def build_dspark_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
    invalid: torch.Tensor,
    out_loc: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    swa_window: int,
    page_index_aligned_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets.ndim != 2 or offsets.shape[1] != swa_window:
        raise ValueError(
            f"offsets must be [bs, swa_window]; got shape={tuple(offsets.shape)} (swa_window={swa_window})."
        )
    bs = offsets.shape[0]
    device = offsets.device
    context_lens = context_lens.to(device=device, dtype=torch.int32)
    window_full_locs = req_to_token[
        req_pool_indices_per_request[:, None].to(torch.int64), offsets
    ]
    window_full_locs = window_full_locs.masked_fill(invalid, 0)
    window_swa_locs = full_to_swa_mapping[window_full_locs].to(torch.int32)
    window_swa_locs = window_swa_locs.masked_fill(invalid, -1)
    block_full_locs = out_loc[: bs * block_size].view(bs, block_size)
    block_swa_locs = full_to_swa_mapping[block_full_locs].to(torch.int32)
    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)
    swa_page_indices = _compact_dspark_window_then_block(
        window_swa_locs=window_swa_locs,
        block_swa_locs=block_swa_locs,
        context_lens=context_lens,
        target_width=target_width,
        block_size=block_size,
        swa_window=swa_window,
    )
    swa_page_indices = (
        swa_page_indices.view(bs, 1, target_width)
        .expand(bs, block_size, target_width)
        .reshape(bs * block_size, target_width)
        .contiguous()
    )
    swa_topk_lengths = (
        (context_lens + block_size)
        .view(bs, 1)
        .expand(bs, block_size)
        .reshape(bs * block_size)
        .contiguous()
        .to(torch.int32)
    )
    return (swa_page_indices, swa_topk_lengths)


def _compact_dspark_window_then_block(
    *,
    window_swa_locs: torch.Tensor,
    block_swa_locs: torch.Tensor,
    context_lens: torch.Tensor,
    target_width: int,
    block_size: int,
    swa_window: int,
) -> torch.Tensor:
    bs = window_swa_locs.shape[0]
    device = window_swa_locs.device
    out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)
    j = torch.arange(swa_window, device=device, dtype=torch.int32).view(1, -1)
    shift = (swa_window - context_lens.view(-1, 1)).to(torch.int32)
    src_col = (shift + j).clamp_(min=0, max=swa_window - 1).to(torch.int64)
    gathered = torch.gather(window_swa_locs, dim=1, index=src_col)
    valid = j < context_lens.view(-1, 1)
    out[:, :swa_window] = torch.where(valid, gathered, -1)
    block_col = context_lens.view(-1, 1) + torch.arange(
        block_size, device=device, dtype=torch.int32
    ).view(1, -1)
    block_rows = torch.arange(bs, device=device).view(-1, 1).expand(-1, block_size)
    out[block_rows, block_col] = block_swa_locs
    return out


def swa_ring_rows(
    slots: torch.Tensor, positions: torch.Tensor, ring_size: int
) -> torch.Tensor:
    """Row of ``positions`` in a per-request SWA ring: request slot ``s`` keeps
    position ``p`` at row ``s * ring_size + p % ring_size``."""
    return slots.to(torch.int64) * ring_size + positions.to(torch.int64) % ring_size


class BuildDsparkSwaRingIndices:
    """``BuildDsparkSwaPageIndices`` over a per-request SWA ring (`swa_ring_rows`),
    with each request's window starting no earlier than its ``valid_from``."""

    @classmethod
    def torch(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        valid_from: torch.Tensor,
        ring_size: int,
        block_size: int,
        swa_window: int,
        page_index_aligned_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_q = seq_lens_casual.size(0)
        assert (
            num_q % block_size == 0
        ), f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not divisible by block_size={block_size}."
        bs = num_q // block_size
        device = seq_lens_casual.device
        first_token = torch.arange(bs, device=device, dtype=torch.int64) * block_size
        prefix_lens = seq_lens_casual[first_token].to(torch.int64) - 1
        slots = req_pool_indices_repeated[first_token].to(torch.int64).view(-1, 1)
        start = torch.maximum(
            prefix_lens - swa_window, valid_from[slots.view(-1)]
        ).clamp_(min=0)
        context_lens = (prefix_lens - start).clamp_(min=0)
        cols = torch.arange(swa_window, device=device, dtype=torch.int64).view(1, -1)
        window_rows = swa_ring_rows(slots, start.view(-1, 1) + cols, ring_size)
        window_rows = torch.where(cols < context_lens.view(-1, 1), window_rows, -1)
        block_offsets = torch.arange(block_size, device=device, dtype=torch.int64)
        block_rows = swa_ring_rows(
            slots, prefix_lens.view(-1, 1) + block_offsets.view(1, -1), ring_size
        )
        target_width = ceil_align(swa_window + block_size, page_index_aligned_size)
        out = torch.full((bs, target_width), -1, dtype=torch.int32, device=device)
        out[:, :swa_window] = window_rows.to(torch.int32)
        out.scatter_(
            1,
            context_lens.view(-1, 1) + block_offsets.view(1, -1),
            block_rows.to(torch.int32),
        )
        swa_page_indices = (
            out.view(bs, 1, target_width)
            .expand(bs, block_size, target_width)
            .reshape(bs * block_size, target_width)
            .contiguous()
        )
        swa_topk_lengths = (
            (context_lens + block_size)
            .view(bs, 1)
            .expand(bs, block_size)
            .reshape(bs * block_size)
            .contiguous()
            .to(torch.int32)
        )
        return (swa_page_indices, swa_topk_lengths)


class BuildBlockSeqLensCausal:

    @classmethod
    def torch(
        cls, *, seq_lens: torch.Tensor, block_size: int, device: torch.device
    ) -> torch.Tensor:
        return build_block_seq_lens_causal(
            seq_lens=seq_lens, block_size=block_size, device=device
        )


def build_block_seq_lens_causal(
    *, seq_lens: torch.Tensor, block_size: int, device: torch.device
) -> torch.Tensor:
    prefix = seq_lens.to(torch.int32)
    steps = torch.arange(1, block_size + 1, device=device, dtype=torch.int32)
    return (prefix[:, None] + steps[None, :]).reshape(-1)
