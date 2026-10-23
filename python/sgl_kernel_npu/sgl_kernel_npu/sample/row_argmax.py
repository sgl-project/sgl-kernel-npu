"""Row-wise FP32 argmax for Triton-Ascend, one program per row.

The operator is memory bound: the logits block is read once, so the kernel is
shaped around Ascend DMA behaviour rather than around the reduction.

* A 32 KB tile reaches ~1.3 TB/s per chip where an 8 KB tile stalls at ~0.25
  TB/s, so a row is streamed in tiles of 8192 FP32 elements.
* Deriving the index inside a tile costs four extra full-length vector passes
  (scalar broadcast, compare, select, second reduce) and roughly triples the
  time, so a tile only reports its maximum and the winning tile is re-read to
  resolve the index -- one extra tile of traffic instead of repeated index work.
* The row scan and index pass share one kernel. Its compiled launcher is reused
  to avoid repeating Python JIT dispatch on every call.
"""

import torch
import triton
import triton.language as tl

# 8192 FP32 elements = 32 KB, the largest tile that still leaves room for the
# double buffer the compiler allocates for the loop-carried load.
_TILE = 8192
_row_argmax_launchers = {}


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _row_argmax_kernel(
    X,
    OUT,
    N: tl.constexpr,
    SX: tl.constexpr,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
    TILE: tl.constexpr,
):
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        base = X + row.to(tl.int64) * SX
        best = float("-inf")
        best_start = 0
        for tile in range(TILES):
            start = tile * TILE
            off = start + tl.arange(0, TILE)
            v = tl.load(base + off, off < N, float("-inf"))
            cur = tl.reduce(v, 0, _maximum_with_nan)
            # NaN is maximal, matching torch.argmax, and only the first is taken;
            # Explicit propagation avoids relying on backend-specific max defaults.
            take = (cur > best) | ((cur != cur) & (best == best))  # noqa: PLR0124
            best_start = tl.where(take, start, best_start)
            best = tl.where(take, cur, best)
        idx = best_start + tl.arange(0, TILE)
        w = tl.load(base + idx, idx < N, float("-inf"))
        sel = tl.where(best != best, w != w, w == best)  # noqa: PLR0124
        tl.store(OUT + row, tl.min(tl.where(sel, idx, N), 0).to(tl.int64))


def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """FP32 row argmax with unit column stride; ties use the first index.

    Matches ``torch.argmax(dim=-1)`` for every row count, NaN included.
    """
    assert x.ndim == 2 and x.dtype == torch.float32 and x.stride(1) == 1
    rows, n = x.shape
    assert n > 0
    out = torch.empty(rows, dtype=torch.int64, device=x.device)
    if rows == 0:
        return out
    # Cache only compiled launchers. Tensor data and the current stream remain
    # runtime arguments. Include pointer alignment in the specialization key.
    key = (x.device, rows, n, x.stride(0), x.data_ptr() % 16)
    launch = _row_argmax_launchers.get(key)
    if launch is None:
        kernel = _row_argmax_kernel.warmup(
            x,
            out,
            n,
            x.stride(0),
            rows,
            triton.cdiv(n, _TILE),
            TILE=_TILE,
            grid=(rows,),
        )
        launch = kernel[(rows, 1, 1)]
        _row_argmax_launchers[key] = launch
    launch(x, out)
    return out
