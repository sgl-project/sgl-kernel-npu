"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


def ceil_align(n, alignment):
    return triton.cdiv(n, alignment) * alignment


class DsparkWindowGather(msgspec.Struct, frozen=True):
    num_q: int
    bs: int
    context_lens: torch.Tensor
    req_pool_indices_per_request: torch.Tensor
    offsets: torch.Tensor
    invalid: torch.Tensor


class ComputeDsparkWindowGather:

    @classmethod
    def triton(
        cls,
        *,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        block_size: int,
        swa_window: int,
        valid_from: Optional[torch.Tensor] = None,
    ) -> DsparkWindowGather:
        return compute_dspark_window_gather_triton(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=swa_window,
            valid_from=valid_from,
        )

    execute = triton


class BuildDsparkSwaPageIndices:

    @classmethod
    def triton(
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
        return build_dspark_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_per_request=req_pool_indices_per_request,
            offsets=offsets,
            out_loc=out_loc,
            context_lens=context_lens,
            block_size=block_size,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
        )

    execute = triton


@triton.jit
def _window_gather_kernel(
    seq_lens_casual_ptr,
    req_pool_rep_ptr,
    context_lens_ptr,
    req_pool_out_ptr,
    offsets_ptr,
    invalid_ptr,
    valid_from_ptr,
    block_size,
    swa_window,
    W_BLOCK: tl.constexpr,
    HAS_VALID_FROM: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        i = _job
        ft = i * block_size
        prefix = tl.load(seq_lens_casual_ptr + ft).to(tl.int64) - 1
        req_pool = tl.load(req_pool_rep_ptr + ft)
        context_len = tl.minimum(prefix, swa_window)
        if HAS_VALID_FROM:
            valid_from = tl.load(valid_from_ptr + req_pool.to(tl.int64))
            context_len = tl.minimum(context_len, tl.maximum(prefix - valid_from, 0))
        tl.store(context_lens_ptr + i, context_len.to(tl.int32))
        tl.store(req_pool_out_ptr + i, req_pool)
        col = tl.arange(0, W_BLOCK)
        cmask = col < swa_window
        off = prefix - swa_window + col
        tl.store(invalid_ptr + i * swa_window + col, off < 0, mask=cmask)
        tl.store(offsets_ptr + i * swa_window + col, tl.maximum(off, 0), mask=cmask)


def compute_dspark_window_gather_triton(
    *,
    seq_lens_casual: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    block_size: int,
    swa_window: int,
    valid_from: Optional[torch.Tensor] = None,
) -> DsparkWindowGather:
    seq_lens_casual = seq_lens_casual.to(torch.int32).contiguous()
    num_q = seq_lens_casual.size(0)
    assert (
        num_q % block_size == 0
    ), f"DSpark draft block forward must be uniform-gamma: num_q={num_q} not divisible by block_size={block_size}."
    bs = num_q // block_size
    device = seq_lens_casual.device
    req_pool_indices_repeated = req_pool_indices_repeated.to(device=device).contiguous()
    context_lens = torch.empty(bs, dtype=torch.int32, device=device)
    req_pool_out = torch.empty(bs, dtype=req_pool_indices_repeated.dtype, device=device)
    offsets = torch.empty((bs, swa_window), dtype=torch.int64, device=device)
    invalid = torch.empty((bs, swa_window), dtype=torch.bool, device=device)
    W_BLOCK = triton.next_power_of_2(swa_window)
    _window_gather_kernel[max(1, min(get_device_properties()[1], bs)),](
        seq_lens_casual,
        req_pool_indices_repeated,
        context_lens,
        req_pool_out,
        offsets,
        invalid,
        valid_from if valid_from is not None else context_lens,
        block_size,
        swa_window,
        W_BLOCK=W_BLOCK,
        HAS_VALID_FROM=valid_from is not None,
        _grid_0=bs,
        num_warps=1,
    )
    return DsparkWindowGather(
        num_q=num_q,
        bs=bs,
        context_lens=context_lens,
        req_pool_indices_per_request=req_pool_out,
        offsets=offsets,
        invalid=invalid,
    )


@triton.jit
def _swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    offsets_ptr,
    out_loc_ptr,
    context_lens_ptr,
    out_ptr,
    topk_ptr,
    rt_stride,
    swa_window,
    block_size,
    target_width,
    TW_BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        q = _job
        i = q // block_size
        cl = tl.load(context_lens_ptr + i)
        rp = tl.load(req_pool_ptr + i).to(tl.int64)
        k = tl.arange(0, TW_BLOCK)
        kmask = k < target_width
        in_window = k < cl
        src_col = tl.minimum(tl.maximum(swa_window - cl + k, 0), swa_window - 1)
        wmask = kmask & in_window
        off = tl.load(offsets_ptr + i * swa_window + src_col, mask=wmask, other=0).to(
            tl.int64
        )
        win_full = tl.load(
            req_to_token_ptr + rp * rt_stride + off, mask=wmask, other=0
        ).to(tl.int64)
        win_swa = tl.load(full_to_swa_ptr + win_full, mask=wmask, other=-1).to(tl.int32)
        in_block = (k >= cl) & (k < cl + block_size)
        bmask = kmask & in_block
        bcol = tl.maximum(k - cl, 0)
        blk_full = tl.load(out_loc_ptr + i * block_size + bcol, mask=bmask, other=0).to(
            tl.int64
        )
        blk_swa = tl.load(full_to_swa_ptr + blk_full, mask=bmask, other=-1).to(tl.int32)
        val = tl.where(in_window, win_swa, tl.where(in_block, blk_swa, -1))
        tl.store(out_ptr + q * target_width + k, val.to(tl.int32), mask=kmask)
        tl.store(topk_ptr + q, (cl + block_size).to(tl.int32))


def build_dspark_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_per_request: torch.Tensor,
    offsets: torch.Tensor,
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
    req_pool = req_pool_indices_per_request.to(device=device).contiguous()
    offsets = offsets.to(torch.int64).contiguous()
    out_loc = out_loc[: bs * block_size].contiguous()
    context_lens = context_lens.to(device=device, dtype=torch.int32).contiguous()
    rt_stride = req_to_token.stride(0)
    target_width = ceil_align(swa_window + block_size, page_index_aligned_size)
    n_q = bs * block_size
    swa_page_indices = torch.empty(
        (n_q, target_width), dtype=torch.int32, device=device
    )
    swa_topk_lengths = torch.empty(n_q, dtype=torch.int32, device=device)
    TW_BLOCK = triton.next_power_of_2(target_width)
    _swa_page_indices_kernel[max(1, min(get_device_properties()[1], n_q)),](
        req_to_token,
        full_to_swa_mapping,
        req_pool,
        offsets,
        out_loc,
        context_lens,
        swa_page_indices,
        swa_topk_lengths,
        rt_stride,
        swa_window,
        block_size,
        target_width,
        TW_BLOCK=TW_BLOCK,
        _grid_0=n_q,
        num_warps=1,
    )
    return (swa_page_indices, swa_topk_lengths)


class BuildBlockSeqLensCausal:

    @classmethod
    def triton(
        cls, *, seq_lens: torch.Tensor, block_size: int, device: torch.device
    ) -> torch.Tensor:
        return build_block_seq_lens_causal_triton(
            seq_lens=seq_lens, block_size=block_size, device=device
        )

    execute = triton


@triton.jit
def _block_seq_lens_casual_kernel(
    seq_lens_ptr, out_ptr, block_size, n_out, BLOCK: tl.constexpr, _grid_0: tl.constexpr
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        pid = _job
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        row = offs // block_size
        col = offs % block_size
        prefix = tl.load(seq_lens_ptr + row, mask=mask, other=0)
        tl.store(out_ptr + offs, (prefix + col + 1).to(tl.int32), mask=mask)


def build_block_seq_lens_causal_triton(
    *, seq_lens: torch.Tensor, block_size: int, device: torch.device
) -> torch.Tensor:
    seq_lens = seq_lens.to(device=device, dtype=torch.int64).contiguous()
    n_rows = seq_lens.shape[0]
    n_out = n_rows * block_size
    out = torch.empty(n_out, dtype=torch.int32, device=device)
    BLOCK = 256
    _block_seq_lens_casual_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(n_out, BLOCK))),
    ](
        seq_lens,
        out,
        block_size,
        n_out,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(n_out, BLOCK),
        num_warps=1,
    )
    return out


from sgl_kernel_npu.attention.dspark_ring import (
    BuildDsparkSwaRingIndices as BuildDsparkSwaRingIndices,
)
