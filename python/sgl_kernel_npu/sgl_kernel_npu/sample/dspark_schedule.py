"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class ScheduleVerifyLensTopk:

    @classmethod
    def triton(cls, *, confidence: torch.Tensor, budget: int, cfg: Any) -> torch.Tensor:
        return schedule_verify_lens_topk_triton(
            confidence=confidence, budget=budget, cfg=cfg
        )

    execute = triton


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    selected_extra_ptr,
    gamma,
    cols,
    G_P2: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        row = _job
        g = tl.arange(0, G_P2)
        conf = tl.load(
            confidence_ptr + row.to(tl.int64) * gamma + g, mask=g < gamma, other=1.0
        ).to(tl.float32)
        surv = tl.cumprod(conf, axis=0)
        tl.store(survival_ptr + row.to(tl.int64) * cols + g, surv, mask=g < cols)
        tl.store(selected_extra_ptr + row, 0)


@triton.jit
def _schedule_topk_finalize_kernel(
    selected_extra_ptr,
    out_ptr,
    min_verify_len,
    lower_bound,
    max_len,
    bs,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        offs = _job * BLOCK + tl.arange(0, BLOCK)
        mask = offs < bs
        extra = tl.load(selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
        lens = min_verify_len + extra
        lens = tl.maximum(lens, lower_bound)
        lens = tl.minimum(lens, max_len)
        tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_selected_extra_kernel(
    survival,
    selected_extra,
    budget,
    cols,
    n,
    survival_eps,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        p = tl.arange(0, COLS)
        prob = tl.load(survival + row * cols + p, p < cols, other=0.0)
        valid = (p < cols) & (prob >= survival_eps)
        rank = tl.zeros((COLS,), tl.int32)
        for start in range(0, n, BLOCK_CP):
            cp = start + tl.arange(0, BLOCK_CP)
            other = tl.load(survival + cp, cp < n, other=0.0)
            valid_other = (cp < n) & (other >= survival_eps)
            greater = other[None, :] > prob[:, None]
            equal = other[None, :] == prob[:, None]
            pos_lt = cp[None, :] % cols < p[:, None]
            pos_eq = cp[None, :] % cols == p[:, None]
            req_lt = cp[None, :] // cols < row
            before = valid_other[None, :] & (
                greater | equal & (pos_lt | pos_eq & req_lt)
            )
            rank += tl.sum(before.to(tl.int32), 1)
        count = tl.sum((valid & (rank < budget)).to(tl.int32), 0)
        tl.store(selected_extra + row, count)


def schedule_verify_lens_topk_triton(
    *, confidence: torch.Tensor, budget: int, cfg: Any
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols
    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    _schedule_topk_prep_kernel[max(1, min(get_device_properties()[1], num_requests)),](
        confidence.contiguous(),
        survival,
        selected_extra,
        gamma,
        cols,
        G_P2=triton.next_power_of_2(max(gamma, 1)),
        _grid_0=num_requests,
        num_warps=1,
    )
    if budget > 0 and n > 0:
        _schedule_topk_selected_extra_kernel[
            max(1, min(num_requests, get_device_properties()[1])),
        ](
            survival,
            selected_extra,
            int(budget),
            cols,
            n,
            float(cfg.survival_eps),
            ROWS=num_requests,
            COLS=triton.next_power_of_2(max(cols, 1)),
            BLOCK_CP=128,
            num_warps=1,
        )
    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    BLOCK = 256
    _schedule_topk_finalize_kernel[
        max(1, min(get_device_properties()[1], triton.cdiv(num_requests, BLOCK))),
    ](
        selected_extra,
        verify_lens,
        int(cfg.min_verify_len),
        max(cfg.min_verify_len, 1),
        int(max_len),
        num_requests,
        BLOCK=BLOCK,
        _grid_0=triton.cdiv(num_requests, BLOCK),
        num_warps=1,
    )
    return verify_lens
