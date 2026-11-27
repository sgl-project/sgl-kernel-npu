"""Fused shared-expert MLP for Qwen3.5 MoE on NPU (F1).

Two Triton kernels replacing the eager chain
    gate_up GEMM -> SwiGLU -> down GEMM -> gate GEMV -> sigmoid * out:

- K1 ``_fused_shared_expert_k1``: gate_up GEMM + SwiGLU in ONE tl.dot per
  k-chunk. The gate/up weight rows are pre-packed in gate/up alternating
  pairs (row 2j = gate row j, row 2j+1 = up row j of an n-chunk; built once
  and cached per weight tensor), so a single [BLOCK_K, 2*BLOCK_OUT] tile
  feeds one dot whose output columns interleave the pair; reshape + tl.split
  separates them register-only and SwiGLU runs in registers. The
  shared-expert gate GEMV is a second tl.dot against a zero-padded
  [BLOCK_OUT, IN] weight (row 0 is the gate weight; the padded copy is
  cached). Rounding points mirror the eager chain: the dot result rounds to
  bf16 (as MatMulV2's output does) before SwiGLU.
- K2 ``_fused_shared_expert_k2``: down GEMM (tl.dot, fp32 accum, rounds to
  bf16) fused with the sigmoid(gate) broadcast multiply, optionally adding a
  residual (kept off by default: production ordering runs the shared expert
  before the routed-expert combine output exists).

Why it is written this way (all measured, see benchmark/fused_shared_expert_tune.py):
  * This stack charges a ~1us fixed cost PER tl.dot instruction regardless
    of tile size (1/2/3 dots per iter on identical tiles: 7.7/22.2/21.6us).
    Minimizing the dot count dominates every other knob: num_stages,
    num_warps, persistent grids and pre-transposed weights all made no
    difference.
  * BLOCK_OUT must stay >= 32 (thin tiles starve the cube).
  * K loads are masked (other=0) so any IN/Half divisible by 64 is safe.

Tuning: BLOCK_K is autotuned per (IN, HALF); BLOCK_OUT is the packing
width (fixed 32) and is NOT autotuned. First call per shape must happen
outside NPU graph capture (sglang warms up eagerly before capture).

GEMM note: tl.dot and MatMulV2 reduce K in identical order (the K1 gate dot
matches MatMulV2 bit-for-bit), but the eager chain computes gu as ONE
N-wide MatMulV2 while we chunk over K, so ~1 bf16 ulp differences on a
fraction of elements are expected; the strict gate is the e2e greedy-token
check (see sim/twin_check.py and the L1 test's per-stage reports).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Packing chunk width: HALF is cut into chunks of this many columns; each
# chunk's gate/up rows are interleaved (2*_K1_PACK_BO rows per chunk). Also
# the BLOCK_OUT of K1's dot. Fixed (not autotuned) because the packed weight
# layout depends on it; 32 keeps the dot's N width at 64 (cube-friendly).
_K1_PACK_BO = 32

# Zero-padded gate-weight rows; must cover BLOCK_OUT of the gate dot.
_GATE_PAD_ROWS = 64

# Cached derived weights; each entry keeps the source tensor alive so its
# data_ptr cannot be recycled while the entry lives.
_PACK_CACHE: dict = {}
_GATE_PAD_CACHE: dict = {}

# Autotune space: only BLOCK_K (dot count = 2 * IN/BLOCK_K per program).
_K1_CONFIGS = [
    triton.Config({"BLOCK_K": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=2),
]
_K2_CONFIGS = [
    triton.Config({"BLOCK_K": 256, "BLOCK_OUT": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 256, "BLOCK_OUT": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 128, "BLOCK_OUT": 64}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_K1_CONFIGS, key=["IN", "HALF"])
@triton.jit
def _fused_shared_expert_k1(
    hidden_ptr,
    wpack_ptr,
    wgate_pad_ptr,
    inter_ptr,
    gate_ptr,
    M,
    IN: tl.constexpr,
    HALF: tl.constexpr,
    M_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # grid = m_blocks * (HALF // BLOCK_OUT); n-index fast-varying so
    # concurrently scheduled programs read disjoint weight slices.
    pid = tl.program_id(0)
    num_n = HALF // BLOCK_OUT
    m_block = pid // num_n
    n_idx = pid % num_n

    offs_m = m_block * M_BLOCK + tl.arange(0, M_BLOCK)
    m_mask = offs_m < M
    # Packed rows for this chunk: gate/up pairs occupy 2*BLOCK_OUT rows.
    offs_np = n_idx * (2 * BLOCK_OUT) + tl.arange(0, 2 * BLOCK_OUT)

    # acc columns interleave (gate j, up j); split after the loop.
    acc = tl.zeros((M_BLOCK, 2 * BLOCK_OUT), dtype=tl.float32)
    acc_gv = tl.zeros((M_BLOCK, BLOCK_OUT), dtype=tl.float32)
    offs_g = tl.arange(0, BLOCK_OUT)

    for k0 in tl.range(0, IN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IN
        h = tl.load(
            hidden_ptr + offs_m[:, None] * IN + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # [BLOCK_K, 2*BLOCK_OUT] tile straight from the packed [2*HALF, IN]
        # row-major weight (element (k, n) sits at n*IN + k).
        wp = tl.load(
            wpack_ptr + offs_np[None, :] * IN + offs_k[:, None],
            mask=k_mask[:, None],
            other=0.0,
        )
        # Padded gate weight tile: columns 1..BLOCK_OUT-1 are exactly zero, so
        # column 0 of acc_gv is the gate GEMV. offs_g has NO chunk offset —
        # the padded tensor is shared by all chunks.
        wg = tl.load(
            wgate_pad_ptr + offs_g[None, :] * IN + offs_k[:, None],
            mask=k_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(h, wp)
        acc_gv += tl.dot(h, wg)

    x1, x2 = tl.split(tl.reshape(acc, (M_BLOCK, BLOCK_OUT, 2)))

    # Round the GEMM output to bf16 first — the eager chain materializes the
    # MatMulV2 result in bf16 and swiglu reads the rounded values. sigmoid in
    # the exp2 form: measurably faster than tl.sigmoid on this stack.
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    inter = (x1 / (1.0 + tl.exp2(x1 * -1.4426950408889634)) * x2).to(tl.bfloat16)

    offs_n = n_idx * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    inter_offsets = offs_m[:, None] * HALF + offs_n[None, :]
    tl.store(inter_ptr + inter_offsets, inter, mask=m_mask[:, None])
    if n_idx == 0:
        # Only the n_idx==0 program's acc_gv column 0 is stored; summing is
        # exact because columns 1..BLOCK_OUT-1 are zeros.
        g = tl.sum(acc_gv, axis=1)
        tl.store(gate_ptr + offs_m, g.to(tl.bfloat16), mask=m_mask)


@triton.autotune(configs=_K2_CONFIGS, key=["K", "N"])
@triton.jit
def _fused_shared_expert_k2(
    inter_ptr,
    wd_ptr,
    gate_ptr,
    out_ptr,
    residual_ptr,
    M,
    HAS_RESIDUAL: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    M_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # grid = m_blocks * (N // BLOCK_OUT); n-index fast-varying.
    pid = tl.program_id(0)
    num_n = N // BLOCK_OUT
    m_block = pid // num_n
    n_idx = pid % num_n

    offs_m = m_block * M_BLOCK + tl.arange(0, M_BLOCK)
    m_mask = offs_m < M
    n0 = n_idx * BLOCK_OUT
    offs_n = n0 + tl.arange(0, BLOCK_OUT)
    n_mask = offs_n < N

    acc = tl.zeros((M_BLOCK, BLOCK_OUT), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        it = tl.load(
            inter_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # [BLOCK_K, BLOCK_OUT] tile straight from the [N, K] row-major weight
        # (element (k, n) at n*K + k); no tl.trans.
        w = tl.load(
            wd_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(it, w)

    # Round the GEMM2 output to bf16 (the eager chain materializes it), then
    # apply the sigmoid gate broadcast exactly like fused_sigmoid_mul.
    down = acc.to(tl.bfloat16).to(tl.float32)
    g = tl.load(gate_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    out = down * tl.sigmoid(g)[:, None]

    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    if HAS_RESIDUAL:
        res = tl.load(residual_ptr + out_offsets, mask=m_mask[:, None], other=0.0)
        out = (out.to(tl.bfloat16).to(tl.float32) + res.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def _pack_gate_up_weight(gate_up_weight: torch.Tensor, half: int) -> torch.Tensor:
    """[2*HALF, IN] -> [2*HALF, IN] with gate/up rows interleaved per chunk.

    Chunk c (BLOCK_OUT columns wide) occupies packed rows
    [c*2*BO, c*2*BO + 2*BO): row 2j = gate row c*BO+j, row 2j+1 = up row
    HALF + c*BO + j. Same size as the source — a pure permutation.
    """
    key = (gate_up_weight.data_ptr(), tuple(gate_up_weight.shape))
    entry = _PACK_CACHE.get(key)
    if entry is None:
        in_features = gate_up_weight.shape[1]
        chunks = half // _K1_PACK_BO
        cols = torch.arange(chunks * _K1_PACK_BO, device=gate_up_weight.device)
        cols = cols.view(chunks, _K1_PACK_BO)
        src = torch.stack([cols, half + cols], dim=2).reshape(-1)
        packed = gate_up_weight[src].contiguous()
        entry = (packed, gate_up_weight)
        _PACK_CACHE[key] = entry
    return entry[0]


def _padded_gate_weight(gate_weight: torch.Tensor) -> torch.Tensor:
    """[1, IN] -> [_GATE_PAD_ROWS, IN], row 0 copied, the rest zero."""
    key = (gate_weight.data_ptr(), tuple(gate_weight.shape), gate_weight.dtype)
    entry = _GATE_PAD_CACHE.get(key)
    if entry is None:
        pad = torch.zeros(
            _GATE_PAD_ROWS,
            gate_weight.shape[1],
            dtype=gate_weight.dtype,
            device=gate_weight.device,
        )
        pad[0].copy_(gate_weight[0])
        entry = (pad, gate_weight)
        _GATE_PAD_CACHE[key] = entry
    return entry[0]


def _launch_k1(hidden_states, wpack, gate_weight_padded, inter, gate, M, IN, HALF):
    # Grid depends only on fixed BLOCK_OUT (the packing width).
    num_n1 = HALF // _K1_PACK_BO
    grid1 = (triton.cdiv(M, 32) * num_n1,)
    _fused_shared_expert_k1[grid1](
        hidden_states,
        wpack,
        gate_weight_padded,
        inter,
        gate,
        M,
        IN=IN,
        HALF=HALF,
        M_BLOCK=32,
        BLOCK_OUT=_K1_PACK_BO,
        multibuffer=True,
    )


def _launch_k2(inter, down_weight, gate, out, residual, M, K, N):
    grid2 = lambda META: (triton.cdiv(M, 32) * (N // META["BLOCK_OUT"]),)
    _fused_shared_expert_k2[grid2](
        inter,
        down_weight,
        gate,
        out,
        residual if residual is not None else out,
        M,
        HAS_RESIDUAL=residual is not None,
        K=K,
        N=N,
        M_BLOCK=32,
        multibuffer=True,
    )


def _k2_block_out(K: int, N: int) -> int:
    for cfg in _K2_CONFIGS:
        bo = cfg.kwargs["BLOCK_OUT"]
        if N % bo == 0:
            return bo
    return 0


def _prepare(hidden_states, gate_up_weight, down_weight, gate_weight, residual):
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(f"fused_shared_expert_mlp expects bf16, got {hidden_states.dtype}")
    M, IN = hidden_states.shape
    two_half, IN_w = gate_up_weight.shape
    N, K = down_weight.shape
    if IN_w != IN or two_half % 2 != 0 or K != two_half // 2:
        raise ValueError(
            f"shape mismatch: hidden {hidden_states.shape}, gate_up {gate_up_weight.shape}, "
            f"down {down_weight.shape}"
        )
    HALF = two_half // 2
    if gate_weight.shape != (1, IN):
        raise ValueError(f"gate_weight must be [1, {IN}], got {gate_weight.shape}")
    if gate_up_weight.dtype != hidden_states.dtype or down_weight.dtype != hidden_states.dtype:
        raise ValueError("gate_up_weight/down_weight dtype must match hidden_states (bf16)")
    if residual is not None and residual.shape != (M, N):
        raise ValueError(f"residual must be [{M}, {N}], got {residual.shape}")
    if IN % 64 or HALF % _K1_PACK_BO or _k2_block_out(K, N) == 0:
        raise ValueError(
            f"unsupported dims: need IN%64==0, HALF%{_K1_PACK_BO}==0, "
            f"N divisible by a K2 BLOCK_OUT (got IN={IN}, HALF={HALF}, K={K}, N={N})"
        )

    hidden_states = hidden_states.contiguous()
    gate_up_weight = gate_up_weight.contiguous()
    down_weight = down_weight.contiguous()
    gate_weight = gate_weight.contiguous()

    inter = torch.empty((M, HALF), dtype=hidden_states.dtype, device=hidden_states.device)
    gate = torch.empty((M,), dtype=hidden_states.dtype, device=hidden_states.device)
    out = torch.empty((M, N), dtype=hidden_states.dtype, device=hidden_states.device)
    return hidden_states, gate_up_weight, down_weight, gate_weight, residual, M, IN, HALF, N, K, inter, gate, out


def fused_shared_expert_mlp(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused shared-expert MLP: gate_up + SwiGLU + down + sigmoid gate.

    Semantics (rounding points included) match the eager chain
    ``sigmoid_mul(shared_mlp(x), shared_gate(x))``.

    The kernels autotune BLOCK_K on first call per (IN, HALF) / (K, N) — that
    first call MUST happen outside NPU graph capture (sglang warms up
    eagerly before capture, so this is guaranteed in production).

    Args:
        hidden_states: [M, IN] bf16.
        gate_up_weight: [2 * HALF, IN] bf16 (F.linear layout, contiguous);
            a gate/up interleaved copy is built once and cached.
        down_weight: [N, K] bf16 with K == HALF (F.linear layout, contiguous).
        gate_weight: [1, IN] bf16.
        residual: optional [M, N] bf16 added after the sigmoid multiply
            (not used by the current MoE ordering; reserved for follow-ups).

    Returns:
        [M, N] bf16.
    """
    (hidden_states, gate_up_weight, down_weight, gate_weight, residual,
     M, IN, HALF, N, K, inter, gate, out) = _prepare(
        hidden_states, gate_up_weight, down_weight, gate_weight, residual
    )
    _launch_k1(
        hidden_states,
        _pack_gate_up_weight(gate_up_weight, HALF),
        _padded_gate_weight(gate_weight),
        inter, gate, M, IN, HALF,
    )
    _launch_k2(inter, down_weight, gate, out, residual, M, K, N)
    return out


def fused_shared_expert_mlp_debug(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    residual: torch.Tensor | None = None,
):
    """Diagnostic variant returning (out, inter, gate) for stage-wise audits."""
    (hidden_states, gate_up_weight, down_weight, gate_weight, residual,
     M, IN, HALF, N, K, inter, gate, out) = _prepare(
        hidden_states, gate_up_weight, down_weight, gate_weight, residual
    )
    _launch_k1(
        hidden_states,
        _pack_gate_up_weight(gate_up_weight, HALF),
        _padded_gate_weight(gate_weight),
        inter, gate, M, IN, HALF,
    )
    _launch_k2(inter, down_weight, gate, out, residual, M, K, N)
    return out, inter, gate
