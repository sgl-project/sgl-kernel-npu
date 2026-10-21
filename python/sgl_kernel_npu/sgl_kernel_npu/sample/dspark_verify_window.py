"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.sample.dspark_ragged import BuildQoIndptr
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor
    window_index: torch.Tensor


class ScatterCompactToStrided:

    @classmethod
    def triton(
        cls,
        *,
        compact: torch.Tensor,
        layout: Any,
        fill_value: float,
        verify_num_draft_tokens: int,
    ) -> torch.Tensor:
        return scatter_compact_to_strided_triton(
            compact=compact,
            layout=layout,
            fill_value=fill_value,
            verify_num_draft_tokens=verify_num_draft_tokens,
        )

    execute = triton


@triton.jit
def _scatter_compact_to_strided_kernel(
    compact_ptr,
    verify_lens_ptr,
    start_ptr,
    out_ptr,
    stride,
    dim,
    fill_value,
    BLOCK_D: tl.constexpr,
    _grid_0: tl.constexpr,
    _grid_1: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0 * _grid_1, tl.num_programs(0)):
        o = (_job // _grid_1).to(tl.int64)
        dblk = _job % _grid_1
        i = o // stride
        w = o % stride
        vl_i = tl.load(verify_lens_ptr + i)
        start_i = tl.load(start_ptr + i)
        d = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d < dim
        in_range = w < vl_i
        src = tl.where(in_range, start_i + w, 0)
        val = tl.load(compact_ptr + src * dim + d, mask=dmask & in_range, other=0)
        val = tl.where(in_range, val, fill_value)
        tl.store(out_ptr + o * dim + d, val, mask=dmask)


def scatter_compact_to_strided_into(
    *,
    compact: torch.Tensor,
    verify_lens: torch.Tensor,
    out: torch.Tensor,
    stride: int,
    fill_value: float,
) -> torch.Tensor:
    dim = compact.shape[1]
    fill_value = float(fill_value) if out.dtype.is_floating_point else int(fill_value)
    compact = compact.contiguous()
    verify_lens = verify_lens.to(dtype=torch.int64).contiguous()
    start = BuildQoIndptr.execute(verify_lens=verify_lens).extend_start_loc
    n_out = out.shape[0]
    BLOCK_D = 1024
    _scatter_compact_to_strided_kernel[
        max(1, min(get_device_properties()[1], n_out * triton.cdiv(dim, BLOCK_D))),
    ](
        compact,
        verify_lens,
        start,
        out,
        stride,
        dim,
        fill_value,
        BLOCK_D=BLOCK_D,
        _grid_0=n_out,
        _grid_1=triton.cdiv(dim, BLOCK_D),
        num_warps=1,
    )
    return out


def scatter_compact_to_strided_triton(
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
    out = torch.empty((bs * stride, dim), dtype=compact.dtype, device=device)
    return scatter_compact_to_strided_into(
        compact=compact,
        verify_lens=layout.verify_lens.to(device=device),
        out=out,
        stride=stride,
        fill_value=fill_value,
    )


class CommitInjectLayoutResult(msgspec.Struct):
    swa_loc: torch.Tensor
    positions: torch.Tensor


class BuildCommitInjectLayout:

    @classmethod
    def triton(
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
        return build_commit_inject_layout_triton(
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            full_to_swa_mapping=full_to_swa_mapping,
            commit_lens=commit_lens,
            stride=stride,
        )

    execute = triton


@triton.jit
def _commit_inject_layout_kernel(
    req_pool_ptr,
    req_to_token_ptr,
    prefix_lens_ptr,
    block_pos_offsets_ptr,
    full_to_swa_ptr,
    commit_lens_ptr,
    swa_loc_ptr,
    positions_ptr,
    rt_stride,
    stride,
    n,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        r = offs // stride
        c = offs % stride
        prefix = tl.load(prefix_lens_ptr + r, mask=mask, other=0).to(tl.int64)
        pos_off = tl.load(block_pos_offsets_ptr + c, mask=mask, other=0).to(tl.int64)
        rp = tl.load(req_pool_ptr + r, mask=mask, other=0).to(tl.int64)
        commit_len = tl.load(commit_lens_ptr + r, mask=mask, other=0).to(tl.int64)
        committed = mask & (c.to(tl.int64) < commit_len)
        full_loc = tl.load(
            req_to_token_ptr + rp * rt_stride + prefix + pos_off,
            mask=committed,
            other=0,
        ).to(tl.int64)
        swa = tl.load(full_to_swa_ptr + full_loc, mask=committed, other=-1).to(tl.int32)
        tl.store(swa_loc_ptr + offs, swa, mask=mask)
        tl.store(positions_ptr + offs, prefix + pos_off, mask=mask)


def build_commit_inject_layout_triton(
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
    n = bs * stride
    device = req_pool_indices.device
    swa_loc = torch.empty(n, dtype=torch.int32, device=device)
    positions = torch.empty(n, dtype=torch.int64, device=device)
    BLOCK = 256
    _commit_inject_layout_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(n, BLOCK))),
    ](
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa_mapping,
        commit_lens,
        swa_loc,
        positions,
        req_to_token.stride(0),
        stride,
        n,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(n, BLOCK),
        num_warps=1,
    )
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)


class BuildOutTokens:

    @classmethod
    def triton(
        cls,
        *,
        draft_tokens: torch.Tensor,
        correct_len: torch.Tensor,
        bonus: torch.Tensor,
        verify_num_draft_tokens: int,
        gamma: int,
    ) -> torch.Tensor:
        return build_out_tokens_triton(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=verify_num_draft_tokens,
            gamma=gamma,
        )

    execute = triton


@triton.jit
def _build_out_tokens_kernel(
    draft_tokens_ptr,
    correct_len_ptr,
    bonus_ptr,
    out_ptr,
    gamma,
    T,
    n_out,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        pid = _job
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        b = offs // T
        k = offs % T
        cl = tl.load(correct_len_ptr + b, mask=mask, other=0).to(tl.int32)
        bonus = tl.load(bonus_ptr + b, mask=mask, other=0)
        draft_mask = mask & (k < gamma)
        draft = tl.load(draft_tokens_ptr + b * gamma + k, mask=draft_mask, other=0)
        select = -(k == cl).to(tl.int64)
        val = (bonus.to(tl.int64) & select) | (draft.to(tl.int64) & ~select)
        tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def build_out_tokens_triton(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    verify_num_draft_tokens: int,
    gamma: int,
) -> torch.Tensor:
    bs = draft_tokens.shape[0]
    T = verify_num_draft_tokens
    device = draft_tokens.device
    draft_tokens = draft_tokens.to(torch.int64).contiguous()
    correct_len_i = correct_len.to(torch.int64).contiguous()
    bonus_i = bonus.to(torch.int64).contiguous()
    out = torch.empty((bs, T), dtype=torch.int64, device=device)
    n_out = bs * T
    BLOCK = 256
    _build_out_tokens_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(n_out, BLOCK))),
    ](
        draft_tokens,
        correct_len_i,
        bonus_i,
        out,
        gamma,
        T,
        n_out,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(n_out, BLOCK),
        num_warps=1,
    )
    return out


from sgl_kernel_npu.sample.dspark_compact import (
    BuildRaggedVerifyWindow as BuildRaggedVerifyWindow,
)
from sgl_kernel_npu.sample.dspark_compact import CompactRowIndex as CompactRowIndex
from sgl_kernel_npu.sample.dspark_compact import CompactVerifyIds as CompactVerifyIds
from sgl_kernel_npu.sample.dspark_compact import (
    compact_row_index_triton as compact_row_index_triton,
)
from sgl_kernel_npu.sample.dspark_compact import (
    compact_verify_ids_triton as compact_verify_ids_triton,
)
