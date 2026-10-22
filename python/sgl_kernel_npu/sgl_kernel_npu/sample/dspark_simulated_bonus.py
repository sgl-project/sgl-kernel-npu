"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

import random

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit(do_not_specialize=["seed"])
def _simulated_bonus_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    part_val_ptr,
    part_idx_ptr,
    seed,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
    _grid_1: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0 * _grid_1, tl.num_programs(0)):
        row = _job // _grid_1
        split = _job % _grid_1
        t = tl.load(temperatures_ptr + row // ROWS).to(tl.float32)
        inv_t = 1.0 / tl.maximum(t, 1e-05)
        offs = split * BLOCK + tl.arange(0, BLOCK)
        mask = offs < vocab
        x = tl.load(
            logits_ptr + row.to(tl.int64) * logits_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        u = tl.rand(seed, row * vocab + offs)
        v = tl.where(mask, x * inv_t - tl.log(-tl.log(u)), float("-inf"))
        top = tl.max(v, axis=0)
        idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
        tl.store(part_val_ptr + row * NSPLIT + split, top)
        tl.store(part_idx_ptr + row * NSPLIT + split, idx)


@triton.jit
def _simulated_bonus_select_kernel(
    part_val_ptr,
    part_idx_ptr,
    correct_len_ptr,
    out_ptr,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        req = _job
        row = req * ROWS + tl.load(correct_len_ptr + req)
        s = tl.arange(0, NSPLIT)
        val = tl.load(part_val_ptr + row * NSPLIT + s)
        idx = tl.load(part_idx_ptr + row * NSPLIT + s)
        top = tl.max(val, axis=0)
        tl.store(
            out_ptr + req, tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
        )


@triton.jit
def _simulated_bonus_row_partial_kernel(
    logits_ptr,
    logits_stride,
    temperatures_ptr,
    correct_len_ptr,
    seed_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab,
    ROWS: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
    _grid_1: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0 * _grid_1, tl.num_programs(0)):
        req = _job // _grid_1
        split = _job % _grid_1
        row = req * ROWS + tl.load(correct_len_ptr + req)
        t = tl.load(temperatures_ptr + req).to(tl.float32)
        offs = split * BLOCK + tl.arange(0, BLOCK)
        mask = offs < vocab
        x = tl.load(
            logits_ptr + row.to(tl.int64) * logits_stride + offs,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        seed = tl.load(seed_ptr)
        u = tl.rand(seed, req * vocab + offs)
        noisy = x / tl.maximum(t, 1e-05) - tl.log(-tl.log(u))
        v = tl.where(mask, tl.where(t > 0.0, noisy, x), float("-inf"))
        top = tl.max(v, axis=0)
        idx = tl.min(tl.where(v == top, offs, vocab), axis=0)
        tl.store(part_val_ptr + req * NSPLIT + split, top)
        tl.store(part_idx_ptr + req * NSPLIT + split, idx)


@triton.jit
def _simulated_bonus_row_select_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    vocab,
    NSPLIT: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        req = _job
        s = tl.arange(0, NSPLIT)
        val = tl.load(part_val_ptr + req * NSPLIT + s)
        idx = tl.load(part_idx_ptr + req * NSPLIT + s)
        top = tl.max(val, axis=0)
        tl.store(
            out_ptr + req, tl.min(tl.where(val == top, idx, vocab), axis=0).to(tl.int64)
        )


def simulated_bonus_sample_graph(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    seed: torch.Tensor,
    bs: int,
    rows_per_request: int,
) -> torch.Tensor:
    """Graph-safe variant of simulated_bonus_sample: seed ([1] int) and per-request
    temperatures ([bs] float; <= 0 means argmax) are read from device buffers, and
    only the row at ``correct_len`` is sampled. [bs] int64."""
    assert target_logits.stride(1) == 1
    device = target_logits.device
    vocab = target_logits.shape[1]
    block = 1024
    nsplit = triton.next_power_of_2(triton.cdiv(vocab, block))
    part_val = torch.empty((bs, nsplit), dtype=torch.float32, device=device)
    part_idx = torch.empty((bs, nsplit), dtype=torch.int32, device=device)
    _simulated_bonus_row_partial_kernel[
        max(1, min(get_device_properties()[1], bs * nsplit)),
    ](
        target_logits,
        target_logits.stride(0),
        temperatures,
        correct_len.contiguous(),
        seed,
        part_val,
        part_idx,
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        BLOCK=block,
        _grid_0=bs,
        _grid_1=nsplit,
        num_warps=1,
    )
    out = torch.empty((bs,), dtype=torch.int64, device=device)
    _simulated_bonus_row_select_kernel[max(1, min(get_device_properties()[1], bs)),](
        part_val, part_idx, out, vocab, NSPLIT=nsplit, _grid_0=bs, num_warps=1
    )
    return out


def simulated_bonus_sample(
    *,
    target_logits: torch.Tensor,
    correct_len: torch.Tensor,
    temperatures: torch.Tensor,
    bs: int,
    rows_per_request: int,
) -> torch.Tensor:
    """Gumbel-max temperature sample of every verify row; returns, per request, the
    sample of the row at ``correct_len``. [bs] int64."""
    assert target_logits.stride(1) == 1
    device = target_logits.device
    vocab = target_logits.shape[1]
    block = 1024
    nsplit = triton.next_power_of_2(triton.cdiv(vocab, block))
    rows = bs * rows_per_request
    part_val = torch.empty((rows, nsplit), dtype=torch.float32, device=device)
    part_idx = torch.empty((rows, nsplit), dtype=torch.int32, device=device)
    _simulated_bonus_partial_kernel[
        max(1, min(get_device_properties()[1], rows * nsplit)),
    ](
        target_logits,
        target_logits.stride(0),
        temperatures.view(-1),
        part_val,
        part_idx,
        random.getrandbits(31),
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        BLOCK=block,
        _grid_0=rows,
        _grid_1=nsplit,
        num_warps=1,
    )
    out = torch.empty((bs,), dtype=torch.int64, device=device)
    _simulated_bonus_select_kernel[max(1, min(get_device_properties()[1], bs)),](
        part_val,
        part_idx,
        correct_len.contiguous(),
        out,
        vocab,
        ROWS=rows_per_request,
        NSPLIT=nsplit,
        _grid_0=bs,
        num_warps=1,
    )
    return out
