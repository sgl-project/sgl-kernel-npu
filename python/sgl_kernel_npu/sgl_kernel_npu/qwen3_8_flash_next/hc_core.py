"""Hyperconnection core: FP32 norm and native-GEMM hybrid mix/combine.

Norm follows the GPU FP32 statistics/scaling stages without compensation.
Reduction order and backend math are not bitwise CUDA parity guarantees.
Mix/combine use FP32 intermediates and row chunking.
In particular, mix does not add the GPU small-row fused BF16 SiLU boundary.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@lru_cache(None)
def _vector_cores(device):
    # Cache immutable device metadata only, never tensor data or converted weights.
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    return max(1, int(properties['num_vectorcore']))


def _grid(tensor, tasks):
    return (min(tasks, _vector_cores(tensor.device.index)),)


@triton.jit
def _norm(X, W, Y, GROUPS: tl.constexpr, EPS: tl.constexpr):
    col = tl.arange(0, 4096)
    for group in range(tl.program_id(0), GROUPS, tl.num_programs(0)):
        offset = group.to(tl.int64) * 2560 + col
        x = tl.load(X + offset, col < 2560, other=0).to(tl.float32)
        w = tl.load(W + (group % 4) * 2560 + col, col < 2560, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / 2560.0 + EPS)
        y = (x * inv) * (1.0 + w)
        tl.store(Y + offset, y, col < 2560)


@triton.jit
def _silu(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    for task in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
        i = task.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, other=0) * 0.25
        tl.store(Y + i, x * tl.sigmoid(x), i < N)


@triton.jit
def _mix(X, G, Y, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.arange(0, 4)
    for task in range(tl.program_id(0), ROWS * tl.cdiv(2560, BLOCK), tl.num_programs(0)):
        row = (task // tl.cdiv(2560, BLOCK)).to(tl.int64)
        col = (task % tl.cdiv(2560, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
        offset = row * 10240 + head[:, None] * 2560 + col[None, :]
        x = tl.load(X + offset, col[None, :] < 2560, other=0).to(tl.float32)
        logits = tl.load(G + offset, col[None, :] < 2560, other=0)
        y = tl.sum(x * tl.sigmoid(logits), 0) * 0.25
        tl.store(Y + row * 2560 + col, y, col < 2560)


@triton.jit
def _combine(B, R, G, Y, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.arange(0, 4)
    for task in range(tl.program_id(0), ROWS * tl.cdiv(2560, BLOCK), tl.num_programs(0)):
        row = (task // tl.cdiv(2560, BLOCK)).to(tl.int64)
        col = (task % tl.cdiv(2560, BLOCK)) * BLOCK + tl.arange(0, BLOCK)
        logits = tl.load(G + row * 4 + head)
        gate = 2.0 * tl.sigmoid(logits * 0.25)
        block = tl.load(B + row * 2560 + col, col < 2560, other=0).to(tl.float32)
        offset = row * 10240 + head[:, None] * 2560 + col[None, :]
        residual = tl.load(R + offset, col[None, :] < 2560, other=0).to(tl.float32)
        y = tl.fma(block[None, :], gate[:, None], residual)
        tl.store(Y + offset, y, col[None, :] < 2560)


def grouped_norm(x, weight, group_size, eps):
    y = torch.empty_like(x)
    _norm[_grid(x, x.shape[0] * 4)](x, weight, y, x.shape[0] * 4, eps,
                                     enable_fp_fusion=False)
    return y


# At 4096 rows the measured unchunked FP32 mix uses about 203 MiB extra.
# This block bounds projection temporaries; full public outputs still grow with R.
# It is a memory scheduling choice, not a model input limit or R32 fast path.
ROW_BLOCK = 4096
# With a preallocated full output, chunking begins to save combine scratch
# beyond two blocks. Use the same resource boundary for both operations.
DIRECT_ROWS = 2 * ROW_BLOCK


def _mix_project(x, down32, up32, out):
    hidden = F.linear(x.float(), down32)
    activated = torch.empty_like(hidden)
    _silu[_grid(x, triton.cdiv(hidden.numel(), 256))](
        hidden, activated, hidden.numel(), 256, enable_fp_fusion=False)
    gates = F.linear(activated, up32)
    _mix[_grid(x, x.shape[0] * 10)](x, gates, out, x.shape[0], 256,
                                  enable_fp_fusion=False)


def mix(x, down, up, hc, hs):
    # Native GEMMs retain FP32 intermediates. Converted weights belong to this
    # invocation, so graph replay observes in-place updates at stable addresses.
    hidden = F.linear(x.float(), down.float())
    activated = torch.empty_like(hidden)
    _silu[_grid(x, triton.cdiv(hidden.numel(), 256))](
        hidden, activated, hidden.numel(), 256, enable_fp_fusion=False)
    gates = F.linear(activated, up.float())
    y = x.new_empty((x.shape[0], 2560))
    _mix[_grid(x, x.shape[0] * 10)](x, gates, y, x.shape[0], 256,
                                     enable_fp_fusion=False)
    return y


def mix_chunked(x, down, up, hc, hs):
    out = x.new_empty((x.shape[0], hs))
    # Per-call conversions remain in the graph; no stale transformed weights.
    down32, up32 = down.float(), up.float()
    for start in range(0, x.shape[0], ROW_BLOCK):
        stop = min(start + ROW_BLOCK, x.shape[0])
        _mix_project(x[start:stop], down32, up32, out[start:stop])
    return out


def _combine_project(block, residual, normed, weight32, out):
    gates = F.linear(normed.float(), weight32)
    _combine[_grid(residual, residual.shape[0] * 5)](
        block, residual, gates, out, residual.shape[0], 512, enable_fp_fusion=False)


def combine(block, residual, normed, weight, hc, hs):
    gates = F.linear(normed.float(), weight.float())
    y = torch.empty_like(residual)
    _combine[_grid(residual, residual.shape[0] * 5)](
        block, residual, gates, y, residual.shape[0], 512, enable_fp_fusion=False)
    return y


def combine_chunked(block, residual, normed, weight, hc, hs):
    out = torch.empty_like(residual)
    weight32 = weight.float()
    for start in range(0, residual.shape[0], ROW_BLOCK):
        stop = min(start + ROW_BLOCK, residual.shape[0])
        _combine_project(block[start:stop], residual[start:stop], normed[start:stop],
                         weight32, out[start:stop])
    return out
