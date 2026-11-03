"""DSpark top-k scheduling, adapted from SGLang dspark_schedule.py."""

import torch
import triton
import triton.language as tl

_schedule_launchers = {}


@triton.jit
def _schedule_topk_prep_kernel(
    confidence_ptr,
    survival_ptr,
    selected_extra_ptr,
    gamma: tl.constexpr,
    cols: tl.constexpr,
    G_P2: tl.constexpr,
):
    row = tl.program_id(0)
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
    min_verify_len: tl.constexpr,
    lower_bound: tl.constexpr,
    max_len: tl.constexpr,
    bs: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    extra = tl.load(selected_extra_ptr + offs, mask=mask, other=0).to(tl.int32)
    lens = min_verify_len + extra
    lens = tl.maximum(lens, lower_bound)
    lens = tl.minimum(lens, max_len)
    tl.store(out_ptr + offs, lens, mask=mask)


@triton.jit
def _schedule_topk_selected_extra_kernel(
    survival_ptr,
    selected_extra_ptr,
    budget,
    cols: tl.constexpr,
    n: tl.constexpr,
    survival_eps,
    BLOCK_C: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < n
    r = c // cols
    p = c % cols
    sp = tl.load(survival_ptr + c, mask=cmask, other=0.0)
    valid_c = sp >= survival_eps
    mp = tl.where(valid_c, sp, float("-inf"))
    rank = tl.zeros([BLOCK_C], dtype=tl.int32)
    for cp0 in range(0, n, BLOCK_CP):
        cp = cp0 + tl.arange(0, BLOCK_CP)
        cpmask = cp < n
        rp = cp // cols
        pp = cp % cols
        spp = tl.load(survival_ptr + cp, mask=cpmask, other=0.0)
        validp = spp >= survival_eps
        mpp = tl.where(validp, spp, float("-inf"))
        gt = mpp[None, :] > mp[:, None]
        eq = mpp[None, :] == mp[:, None]
        pos_lt = pp[None, :] < p[:, None]
        pos_eq = pp[None, :] == p[:, None]
        req_lt = rp[None, :] < r[:, None]
        before = gt | (eq & (pos_lt | (pos_eq & req_lt)))
        before = before & cpmask[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)
    selected = valid_c & (rank < budget)
    tl.atomic_add(selected_extra_ptr + r, selected.to(tl.int32), mask=cmask)


def schedule_verify_lens_topk_triton(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg,
) -> torch.Tensor:
    num_requests, gamma = confidence.shape
    max_len = cfg.resolved_max_verify_len()
    device = confidence.device
    cols = min(max_len, gamma)
    n = num_requests * cols

    if num_requests == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)

    selected_extra = torch.empty(num_requests, dtype=torch.int32, device=device)
    survival = torch.empty((num_requests, cols), dtype=torch.float32, device=device)
    verify_lens = torch.empty(num_requests, dtype=torch.int32, device=device)
    confidence = confidence.contiguous()
    # Cache compiled launchers, not tensors. Budget, threshold, pointers and the
    # current stream remain runtime inputs, including during graph capture.
    key = (
        device,
        confidence.dtype,
        num_requests,
        gamma,
        max_len,
        int(cfg.min_verify_len),
        confidence.data_ptr() % 16,
    )
    launchers = _schedule_launchers.get(key)
    if launchers is None:
        prep = _schedule_topk_prep_kernel.warmup(
            confidence,
            survival,
            selected_extra,
            gamma,
            cols,
            G_P2=triton.next_power_of_2(max(gamma, 1)),
            grid=(num_requests,),
        )
        finalize = _schedule_topk_finalize_kernel.warmup(
            selected_extra,
            verify_lens,
            int(cfg.min_verify_len),
            max(cfg.min_verify_len, 1),
            int(max_len),
            num_requests,
            BLOCK=256,
            grid=(triton.cdiv(num_requests, 256),),
        )
        launchers = [
            prep[(num_requests, 1, 1)],
            None,
            finalize[(triton.cdiv(num_requests, 256), 1, 1)],
        ]
        _schedule_launchers[key] = launchers
    launchers[0](confidence, survival, selected_extra)
    if budget > 0 and n > 0:
        if launchers[1] is None:
            # Representative non-special scalar values keep budget/epsilon in
            # the compiled ABI (even when a later call uses budget=1 or eps=0).
            rank = _schedule_topk_selected_extra_kernel.warmup(
                survival,
                selected_extra,
                2,
                cols,
                n,
                0.5,
                BLOCK_C=16,
                BLOCK_CP=128,
                grid=(triton.cdiv(n, 16),),
            )
            launchers[1] = rank[(triton.cdiv(n, 16), 1, 1)]
        launchers[1](
            survival, selected_extra, min(int(budget), n), float(cfg.survival_eps)
        )
    launchers[2](selected_extra, verify_lens)
    return verify_lens
