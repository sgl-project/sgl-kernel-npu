"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally_triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )

    execute = triton


@triton.jit
def _expand_prefill_causally_kernel(
    req_pool_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    seq_lens_casual_ptr,
    req_pool_repeated_ptr,
    bs,
    num_tokens,
    total_tokens,
    BLOCK: tl.constexpr,
    BS_P2: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_tokens
        b = tl.arange(0, BS_P2)
        bmask = b < bs
        extend = tl.load(extend_seq_lens_ptr + b, mask=bmask, other=0).to(tl.int32)
        start_locs = tl.cumsum(extend, axis=0) - extend
        is_real = offs < num_tokens
        t = tl.where(is_real, offs, 0).to(tl.int32)
        started = (start_locs[None, :] <= t[:, None]) & bmask[None, :]
        r = tl.sum(started.to(tl.int32), axis=1) - 1
        r = tl.where(is_real, r, bs - 1).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
        ext = tl.load(extend_seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
        start_loc = tl.max(tl.where(started, start_locs[None, :], -1), axis=1)
        causal = seq_len - ext + 1 + (t - start_loc)
        causal = tl.where(is_real, causal, 1)
        rp = tl.load(req_pool_ptr + r, mask=mask, other=0)
        tl.store(seq_lens_casual_ptr + offs, causal, mask=mask)
        tl.store(req_pool_repeated_ptr + offs, rp, mask=mask)


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )
    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=device
    )
    # Bound the broadcast [token_tile, batch] intermediates to 16K elements.
    BLOCK = min(256, max(1, 16384 // triton.next_power_of_2(max(bs, 1))))
    _expand_prefill_causally_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(total_tokens, BLOCK))),
    ](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        seq_lens_casual,
        req_pool_indices_repeated,
        bs,
        num_tokens,
        total_tokens,
        BLOCK=BLOCK,
        BS_P2=triton.next_power_of_2(max(bs, 1)),
        _grid_0=triton.cdiv(total_tokens, BLOCK),
        num_warps=1,
    )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


class PageTablePositionsResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor
    page_table: torch.Tensor
    swa_topk_lengths: torch.Tensor


class BuildPageTablePositions:

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
    ) -> PageTablePositionsResult:
        return build_page_table_positions_triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )

    execute = triton


@triton.jit
def _page_table_positions_kernel(
    req_to_token_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    seq_lens_out_ptr,
    positions_out_ptr,
    page_table_ptr,
    topk_out_ptr,
    rt_stride,
    num_pages,
    page_size,
    swa_window,
    BLOCK_P: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        row = _job
        seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
        tl.store(seq_lens_out_ptr + row, seq_len)
        tl.store(positions_out_ptr + row, seq_len - 1)
        tl.store(topk_out_ptr + row, tl.minimum(seq_len, swa_window))
        rp = tl.load(req_pool_ptr + row).to(tl.int64)
        base = req_to_token_ptr + rp * rt_stride
        out_base = page_table_ptr + row.to(tl.int64) * num_pages
        for p0 in range(0, num_pages, BLOCK_P):
            p = p0 + tl.arange(0, BLOCK_P)
            pmask = p < num_pages
            tok = tl.load(base + p.to(tl.int64) * page_size, mask=pmask, other=0).to(
                tl.int32
            )
            tl.store(out_base + p, tok // page_size, mask=pmask)


def build_page_table_positions_triton(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    num_q = seq_lens_casual.shape[0]
    num_pages = (max_seq_len + page_size - 1) // page_size
    device = seq_lens_casual.device
    seq_lens_out = torch.empty(num_q, dtype=torch.int32, device=device)
    positions_out = torch.empty(num_q, dtype=torch.int32, device=device)
    page_table = torch.empty((num_q, num_pages), dtype=torch.int32, device=device)
    topk_out = torch.empty(num_q, dtype=torch.int32, device=device)
    BLOCK_P = 256
    _page_table_positions_kernel[max(1, min(get_device_properties()[1], num_q)),](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        seq_lens_out,
        positions_out,
        page_table,
        topk_out,
        req_to_token.stride(0),
        num_pages,
        page_size,
        swa_window,
        BLOCK_P=BLOCK_P,
        _grid_0=num_q,
        num_warps=1,
    )
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_out,
        positions_casual=positions_out,
        page_table=page_table,
        swa_topk_lengths=topk_out,
    )


class BuildCausalSwaPageIndices:

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
        swa_replay_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
            swa_replay_start=swa_replay_start,
        )

    execute = triton


@triton.jit
def _causal_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    swa_replay_start_ptr,
    out_ptr,
    rt_stride,
    swa_window,
    padded_width,
    HAS_SWA_REPLAY_START: tl.constexpr,
    BLOCK_K: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        row = _job
        pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
        rp = tl.load(req_pool_ptr + row).to(tl.int64)
        base = req_to_token_ptr + rp * rt_stride
        out_base = out_ptr + row.to(tl.int64) * padded_width
        if HAS_SWA_REPLAY_START:
            floor = tl.load(swa_replay_start_ptr + row).to(tl.int64)
        else:
            floor = tl.zeros((), dtype=tl.int64)
        for k0 in range(0, padded_width, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            kmask = k < padded_width
            off = pos - k.to(tl.int64)
            valid = (k < swa_window) & (off >= floor) & kmask
            full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
                tl.int64
            )
            swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
            tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
    swa_replay_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_qo_tokens = seq_lens_casual.size(0)
    padded_width = (
        (swa_window + page_index_aligned_size - 1)
        // page_index_aligned_size
        * page_index_aligned_size
    )
    out = torch.empty(
        (num_qo_tokens, padded_width), dtype=torch.int32, device=seq_lens_casual.device
    )
    BLOCK_K = 256
    has_swa_replay_start = swa_replay_start is not None
    _causal_swa_page_indices_kernel[
        max(1, min(get_device_properties()[1], num_qo_tokens)),
    ](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        swa_replay_start if has_swa_replay_start else seq_lens_casual,
        out,
        req_to_token.stride(0),
        swa_window,
        padded_width,
        HAS_SWA_REPLAY_START=has_swa_replay_start,
        BLOCK_K=BLOCK_K,
        _grid_0=num_qo_tokens,
        num_warps=1,
    )
    return out
