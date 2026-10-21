"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _gather_two_level_bonus_kernel(
    accept_index_ptr,
    predicts_ptr,
    correct_len_ptr,
    out_ptr,
    cols,
    n,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
        accept_pos = tl.load(
            accept_index_ptr + offs * cols + cl, mask=mask, other=0
        ).to(tl.int64)
        bonus = tl.load(predicts_ptr + accept_pos, mask=mask, other=0)
        tl.store(out_ptr + offs, bonus.to(tl.int64), mask=mask)


def gather_two_level_bonus_triton(
    *, accept_index: torch.Tensor, predicts: torch.Tensor, correct_len: torch.Tensor
) -> torch.Tensor:
    bs, cols = accept_index.shape
    accept_index = accept_index.contiguous()
    predicts = predicts.contiguous()
    correct_len = correct_len.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=accept_index.device)
    BLOCK = 256
    _gather_two_level_bonus_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(bs, BLOCK))),
    ](
        accept_index,
        predicts,
        correct_len,
        out,
        cols,
        bs,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(bs, BLOCK),
        num_warps=1,
    )
    return out


class SoftmaxTemp:

    @classmethod
    def triton(
        cls, *, logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int
    ) -> torch.Tensor:
        return softmax_temp_triton(
            logits=logits, temperatures=temperatures, rows_per_request=rows_per_request
        )

    execute = triton


@triton.jit
def _softmax_temp_kernel(
    logits_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    rows_per_request,
    logits_row_stride,
    BLOCK_V: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        row = _job
        temp = tl.load(temp_ptr + row // rows_per_request).to(tl.float32)
        base = logits_ptr + row.to(tl.int64) * logits_row_stride
        out_base = out_ptr + row.to(tl.int64) * vocab
        row_max = -float("inf")
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            row_max = tl.maximum(row_max, tl.max(x, axis=0))
        sum_exp = 0.0
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            e = tl.exp(x - row_max)
            e = tl.where(vmask, e, 0.0)
            sum_exp += tl.sum(e, axis=0)
        for v0 in range(0, vocab, BLOCK_V):
            offs = v0 + tl.arange(0, BLOCK_V)
            vmask = offs < vocab
            x = tl.load(base + offs, mask=vmask, other=-float("inf")).to(tl.float32)
            x = x / temp
            e = tl.exp(x - row_max)
            tl.store(out_base + offs, e / sum_exp, mask=vmask)


def softmax_temp_triton(
    *, logits: torch.Tensor, temperatures: torch.Tensor, rows_per_request: int
) -> torch.Tensor:
    num_rows, vocab = (logits.shape[0], logits.shape[-1])
    bs = num_rows // rows_per_request
    assert (
        bs * rows_per_request == num_rows
    ), f"num_rows {num_rows} not divisible by rows_per_request {rows_per_request}"
    temperatures = temperatures.reshape(bs).to(torch.float32).contiguous()
    out = torch.empty((num_rows, vocab), dtype=torch.float32, device=logits.device)
    BLOCK_V = 4096
    _softmax_temp_kernel[max(1, min(get_device_properties()[1], num_rows)),](
        logits,
        temperatures,
        out,
        vocab,
        rows_per_request,
        logits.stride(0),
        BLOCK_V=BLOCK_V,
        _grid_0=num_rows,
        num_warps=1,
    )
    return out


class MixedAcceptSelectResult(msgspec.Struct):
    correct_len: torch.Tensor
    bonus: torch.Tensor
    cap_trim_lens: torch.Tensor


class SelectMixedAccept:

    @classmethod
    def triton(
        cls,
        *,
        greedy_mask: torch.Tensor,
        greedy_len: torch.Tensor,
        greedy_bonus: torch.Tensor,
        greedy_trim: torch.Tensor,
        sampling_len: torch.Tensor,
        sampling_bonus: torch.Tensor,
        sampling_trim: torch.Tensor,
    ) -> MixedAcceptSelectResult:
        return select_mixed_accept_triton(
            greedy_mask=greedy_mask,
            greedy_len=greedy_len,
            greedy_bonus=greedy_bonus,
            greedy_trim=greedy_trim,
            sampling_len=sampling_len,
            sampling_bonus=sampling_bonus,
            sampling_trim=sampling_trim,
        )

    execute = triton


@triton.jit
def _mixed_accept_select_kernel(
    greedy_mask_ptr,
    greedy_len_ptr,
    greedy_bonus_ptr,
    greedy_trim_ptr,
    sampling_len_ptr,
    sampling_bonus_ptr,
    sampling_trim_ptr,
    correct_len_ptr,
    bonus_ptr,
    cap_trim_ptr,
    bs,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < bs
        is_greedy = tl.load(greedy_mask_ptr + offs, mask=mask, other=0) != 0
        g_len = tl.load(greedy_len_ptr + offs, mask=mask, other=0)
        s_len = tl.load(sampling_len_ptr + offs, mask=mask, other=0)
        tl.store(correct_len_ptr + offs, tl.where(is_greedy, g_len, s_len), mask=mask)
        g_bonus = tl.load(greedy_bonus_ptr + offs, mask=mask, other=0)
        s_bonus = tl.load(sampling_bonus_ptr + offs, mask=mask, other=0)
        tl.store(bonus_ptr + offs, tl.where(is_greedy, g_bonus, s_bonus), mask=mask)
        g_trim = tl.load(greedy_trim_ptr + offs, mask=mask, other=0)
        s_trim = tl.load(sampling_trim_ptr + offs, mask=mask, other=0)
        tl.store(cap_trim_ptr + offs, tl.where(is_greedy, g_trim, s_trim), mask=mask)


def select_mixed_accept_triton(
    *,
    greedy_mask: torch.Tensor,
    greedy_len: torch.Tensor,
    greedy_bonus: torch.Tensor,
    greedy_trim: torch.Tensor,
    sampling_len: torch.Tensor,
    sampling_bonus: torch.Tensor,
    sampling_trim: torch.Tensor,
) -> MixedAcceptSelectResult:
    bs = greedy_mask.shape[0]
    device = greedy_mask.device
    correct_len = torch.empty(bs, dtype=sampling_len.dtype, device=device)
    bonus = torch.empty(bs, dtype=sampling_bonus.dtype, device=device)
    cap_trim_lens = torch.empty(bs, dtype=sampling_trim.dtype, device=device)
    BLOCK = 256
    _mixed_accept_select_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(bs, BLOCK))),
    ](
        greedy_mask,
        greedy_len,
        greedy_bonus,
        greedy_trim,
        sampling_len,
        sampling_bonus,
        sampling_trim,
        correct_len,
        bonus,
        cap_trim_lens,
        bs,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(bs, BLOCK),
        num_warps=1,
    )
    return MixedAcceptSelectResult(
        correct_len=correct_len, bonus=bonus, cap_trim_lens=cap_trim_lens
    )


@triton.jit
def _gather_row_bonus_kernel(
    table_ptr, idx_ptr, out_ptr, cols, n, BLOCK: tl.constexpr, _grid_0: tl.constexpr
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(table_ptr + offs * cols + idx, mask=mask, other=0)
        tl.store(out_ptr + offs, val.to(tl.int64), mask=mask)


def gather_row_bonus_triton(*, table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bs, cols = table.shape
    table = table.contiguous()
    idx = idx.contiguous()
    out = torch.empty(bs, dtype=torch.int64, device=table.device)
    BLOCK = 256
    _gather_row_bonus_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(bs, BLOCK))),
    ](
        table,
        idx,
        out,
        cols,
        bs,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(bs, BLOCK),
        num_warps=1,
    )
    return out


class FinalizeAcceptLensResult(msgspec.Struct):
    commit_lens: torch.Tensor
    new_seq_lens: torch.Tensor
    cap_trim_lens: torch.Tensor


class FinalizeAcceptLens:

    @classmethod
    def triton(
        cls,
        *,
        correct_len: torch.Tensor,
        cap_trim_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> FinalizeAcceptLensResult:
        return finalize_accept_lens_triton(
            correct_len=correct_len,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=prefix_lens,
        )

    execute = triton


@triton.jit
def _finalize_accept_lens_kernel(
    correct_len_ptr,
    cap_trim_ptr,
    prefix_lens_ptr,
    commit_lens_ptr,
    new_seq_lens_ptr,
    cap_trim_out_ptr,
    bs,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < bs
        commit = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int32) + 1
        prefix = tl.load(prefix_lens_ptr + offs, mask=mask, other=0)
        trim = tl.load(cap_trim_ptr + offs, mask=mask, other=0).to(tl.int32)
        tl.store(commit_lens_ptr + offs, commit, mask=mask)
        tl.store(new_seq_lens_ptr + offs, prefix + commit, mask=mask)
        tl.store(cap_trim_out_ptr + offs, trim, mask=mask)


def finalize_accept_lens_triton(
    *, correct_len: torch.Tensor, cap_trim_lens: torch.Tensor, prefix_lens: torch.Tensor
) -> FinalizeAcceptLensResult:
    bs = correct_len.shape[0]
    device = correct_len.device
    commit_lens = torch.empty(bs, dtype=torch.int32, device=device)
    new_seq_lens = torch.empty(bs, dtype=prefix_lens.dtype, device=device)
    cap_trim_out = torch.empty(bs, dtype=torch.int32, device=device)
    BLOCK = 256
    _finalize_accept_lens_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(bs, BLOCK))),
    ](
        correct_len,
        cap_trim_lens,
        prefix_lens,
        commit_lens,
        new_seq_lens,
        cap_trim_out,
        bs,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(bs, BLOCK),
        num_warps=1,
    )
    return FinalizeAcceptLensResult(
        commit_lens=commit_lens, new_seq_lens=new_seq_lens, cap_trim_lens=cap_trim_out
    )


class CapCorrectLen:

    @classmethod
    def triton(
        cls, *, correct_len: torch.Tensor, verify_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cap_correct_len_triton(correct_len=correct_len, verify_lens=verify_lens)

    execute = triton


@triton.jit
def _cap_correct_len_kernel(
    correct_len_ptr,
    verify_lens_ptr,
    capped_ptr,
    trim_ptr,
    n,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        pid = _job
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        cl = tl.load(correct_len_ptr + offs, mask=mask, other=0).to(tl.int64)
        vl = tl.load(verify_lens_ptr + offs, mask=mask, other=0).to(tl.int64)
        ell = vl - 1
        capped = tl.minimum(cl, ell)
        trim = cl - capped
        tl.store(capped_ptr + offs, capped, mask=mask)
        tl.store(trim_ptr + offs, trim, mask=mask)


def cap_correct_len_triton(
    *, correct_len: torch.Tensor, verify_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    device = correct_len.device
    correct_len = correct_len.contiguous()
    verify_lens = verify_lens.to(device=device).contiguous()
    n = correct_len.shape[0]
    capped = torch.empty_like(correct_len)
    trim = torch.empty_like(correct_len)
    BLOCK = 1024
    _cap_correct_len_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(n, BLOCK))),
    ](
        correct_len,
        verify_lens,
        capped,
        trim,
        n,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(n, BLOCK),
        num_warps=1,
    )
    return (capped, trim)


from sgl_kernel_npu.sample.dspark_greedy import AcceptGreedy as AcceptGreedy
