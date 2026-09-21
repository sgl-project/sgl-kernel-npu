"""Model-scoped QSA MQA v1.1: one tiled scoring strategy, two K layouts.

Inputs and output
-----------------
Packed:
    Q       [R, 4, 128]   contiguous BF16 queries
    K       [M, 1, 128]   contiguous BF16 packed compressed keys
    starts  [R]          contiguous int32 inclusive packed offsets
    ends    [R]          contiguous int32 exclusive packed offsets
    output  [R, M]       contiguous FP32; invisible columns are -inf
Paged:
    Q       [R, 4, 128]       contiguous BF16 queries
    cache   [P, 16, 1, 128]   contiguous BF16 physical compressed-key pages
    table   [R, L]           contiguous int32 logical-to-physical page map
    lengths [R]              contiguous int32 valid logical-key counts
    width                    Python int, exactly 16 * L
    output  [R, 16 * L]      contiguous FP32; invalid columns are -inf
All tensors must reside on the same NPU. R includes caller-padded query rows.
M may concatenate complete compressed blocks from multiple requests. P > 0
is physical allocation, not context length; 16 * L is output capacity.

Upstream input contract
----------------------
The supported configuration has four replicated indexer Q heads (not divided
by TP), one K head, D=128, BF16 model execution, full page size 64 and compression
ratio 4. The pool explicitly stores BF16 keys. NPU Q norm/RoPE preparation and
packed gathers provide contiguous tensors; metadata builders produce int32
arrays. These facts are configuration/call-chain requirements, not consequences
of this module's validator. GPU wrapper casting/contiguity/head adaptation is
not reproduced here. Other model configurations require separate review.
_validate checks shape, dtype, layout, device and width metadata only. Callers
must guarantee 0 <= starts[r] <= ends[r] <= M; 0 <= lengths[r] <= 16 * L;
live page IDs in [0, P); and finite, model-valid Q/live K with representable
FP32 scores. Device contents are neither scanned nor repaired. No runtime
rejection of bad device values is promised. Unused paged IDs/cache values are
irrelevant. Packed tiles may read keys outside a particular row's interval.

Examples and computation paths
------------------------------
The caller chooses the entry by K storage; the kernel does not guess:
  packed(...): flat K[M,1,128], with starts/ends giving each row's key range.
  paged(...):  cache[P,16,1,128], with table/lengths locating each row's keys.
Both use the same scoring algorithm. Q row count or a prefill/decode/verify
label does not switch K storage. In particular, R=1 can still use packed.

Q and K/cache below are BF16; outputs are FP32.
BQ x BN means query rows x key columns computed per tile.

Q shape        | K / cache shape | Table  | Output      | Entry  | BQ x BN
---------------+-----------------+--------+-------------+--------+--------
[1,4,128]      | [48,1,128]      | --     | [1,48]      | packed | 4 x 128
[33,4,128]     | [257,1,128]     | --     | [33,257]    | packed | 16 x 128
[4096,4,128]   | [1024,1,128]    | --     | [4096,1024] | packed | 8 x 256
[1,4,128]      | [65536,1,128]   | --     | [1,65536]   | packed | 4 x 256
[1,4,128]      | [P,16,1,128]    | [1,64] | [1,1024]    | paged  | 1 x 256
[32,4,128]     | [P,16,1,128]    | [32,65]| [32,1040]   | paged  | 1 x 256

Packed output width is M; crossing M=1024 changes tiles only, not the algorithm.
Paged output width is 16*L, independent of physical page count P or live length.
Extra tile rows/columns are masked: BQ=4 with R=1 still returns only one row.

Condition                  | Output behavior                 | Launch kernel?
---------------------------+---------------------------------+---------------
Paged width1040, length17   | 17 scores, then 1023 -inf values | Yes
Nonempty, all-masked row    | Full-width row of -inf          | Yes
R=0                        | Empty [0, width]                | No
Packed M=0 / paged L=0      | Empty [R, 0]                    | No
Empty inputs still undergo metadata checks; paged requires P>0 and width=16*L.

Tiles and scheduling
--------------------
Packed M<1024: BQ=min(16,max(4,next_power_of_2(R))), BN=128, HQ=4*BQ.
Packed M>=1024: BQ=min(8,max(4,next_power_of_2(R))), BN=256, HQ=4*BQ.
1024 is a coarse performance choice, not a support or semantic boundary.
Paged: BQ=1, BN=256; four real heads are zero-padded to HQ=16 inside the
matrix tile, without changing the external H=4 contract. Each key tile spans
16 logical pages. Tail rows, keys and internal heads use masked loads/stores.
There are ceil(R/BQ)*ceil(width/BN) tasks. Launch min(num_aicore,tasks) programs;
program p handles p, p+programs, ... in row-major tile order. The core count is
a cached runtime capability, not a device-name generation test. Bounded launch
size addresses historical grid limits; tile sizes manage compiler resources
and measured performance. They do not certify arbitrary device-memory sizes.
Output is always full width and compiler workspace is additional memory.

How scores are computed
-----------------------
score[r,j] = sum_h relu(sum_d Q[r,h,d] * K_for_row_r[j,d]) / sqrt(128).
Matrix operands are BF16 with FP32 accumulation/output; FP fusion is disabled.
Packed visibility is starts[r] <= j < ends[r]. Packed computes each matrix
tile unconditionally, then selects visibility. Historical conditional packed
trials were slower and a long-prefix graph trial failed; the exact cause was
not established. This implementation retains the verified unconditional form.
Paged addresses cache[table[r,j//16], j%16, 0, :]. It loads page IDs/keys only
for valid logical keys and skips matrix work for wholly invisible key tiles,
while still writing -inf to invalid output columns. Valid zero scores stay zero.
There is no V, softmax, head weighting, TopK, custom scale or input mutation.

Graph replay and limitations
----------------------------
Register the Torch NPU backend, select the device and run eager warmup for each
shape before capture; this compiles kernels and caches the compute-core count.
Capture with fixed shapes, dtypes, strides, width and device. Keep graph, input
buffers, cache and captured output alive. Update Q/K/bounds/table/lengths in
place at stable addresses, order updates before replay, and consume output
before overwriting it on the next replay. New shapes require warmup/recapture.
All device values remain on device; no host synchronization or content scan is
added by these entries. Standalone replay tests do not prove whole-backend
capture or model/TopK equivalence. No generic fallback, alternate dtype/layout,
external padded H8, live-page clamp, NaN/inf repair or reserved-memory guarantee
is provided. See HANDOFF.md for finite validation and integration acceptance.
"""
import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

__all__ = ["packed", "paged"]


@lru_cache(maxsize=None)
def _compute_cores(device_index):
    # Query a capability, not a device-name or generation string. Warmup caches
    # this host-side property before graph capture; tensor values stay on device.
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    cores = properties["num_aicore"]
    if cores <= 0:
        raise RuntimeError("Invalid NPU compute-core count")
    return cores


@triton.jit
def _score(Q, K, First, Last, Out, R: tl.constexpr, W: tl.constexpr,
           L: tl.constexpr, PAGED: tl.constexpr, BQ: tl.constexpr,
           BN: tl.constexpr, HQ: tl.constexpr, PROGRAMS: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(R, BQ) * tl.cdiv(W, BN), PROGRAMS):
        row_base = (tile // tl.cdiv(W, BN)) * BQ
        col_tile = tile % tl.cdiv(W, BN)
        cols = col_tile * BN + tl.arange(0, BN)
        dims = tl.arange(0, 128)
        heads = tl.arange(0, HQ)
        rows = row_base + tl.arange(0, BQ)
        if PAGED:
            length = tl.load(Last + row_base)
            valid = (cols < W) & (cols < length)
            result = tl.full((BN,), -float("inf"), tl.float32)
            if col_tile * BN < length:
                pages = tl.load(First + row_base * L + cols // 16,
                                valid, other=0).to(tl.int64)
                k = tl.load(K + (pages[:, None] * 16 + cols[:, None] % 16) * 128
                            + dims[None, :], valid[:, None], other=0)
                q = tl.load(Q + row_base.to(tl.int64) * 512
                            + heads[:, None] * 128 + dims[None, :],
                            heads[:, None] < 4, other=0)
                scores = tl.dot(q, tl.trans(k))
                result = tl.sum(tl.maximum(scores, 0), axis=0) / math.sqrt(128)
                result = tl.where(valid, result, -float("inf"))
            tl.store(Out + row_base.to(tl.int64) * W + cols, result, cols < W)
        else:
            starts = tl.load(First + rows, rows < R, other=W)
            ends = tl.load(Last + rows, rows < R, other=0)
            valid = ((rows[:, None] < R) & (cols[None, :] < W)
                     & (cols[None, :] >= starts[:, None])
                     & (cols[None, :] < ends[:, None]))
            # Compute packed tiles unconditionally, then apply row visibility.
            # The conditional version stalled during long-prefix graph replay
            # on the recorded toolchain; its exact failure cause is not established.
            q = tl.load(Q + (row_base * 4 + heads[:, None]).to(tl.int64) * 128
                        + dims[None, :], row_base * 4 + heads[:, None] < R * 4,
                        other=0)
            k = tl.load(K + cols[:, None].to(tl.int64) * 128 + dims[None, :],
                        cols[:, None] < W, other=0)
            scores = tl.dot(q, tl.trans(k))
            scores = tl.reshape(tl.maximum(scores, 0), (BQ, 4, BN))
            result = tl.sum(scores, axis=1) / math.sqrt(128)
            result = tl.where(valid, result, -float("inf"))
            tl.store(Out + rows[:, None].to(tl.int64) * W + cols[None, :], result,
                     (rows[:, None] < R) & (cols[None, :] < W))


def _validate(q, k, first, last, paged, width=None):
    if q.ndim != 3 or q.shape[1:] != (4, 128):
        raise ValueError("Expected Q[R,4,128]")
    if q.device.type != "npu" or any(x.device != q.device for x in (k, first, last)):
        raise ValueError("All inputs must be on the same NPU")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q and K must be BF16")
    if any(not x.is_contiguous() for x in (q, k, first, last)):
        raise ValueError("Inputs must be contiguous")
    if first.dtype != torch.int32 or last.dtype != torch.int32:
        raise ValueError("Metadata must be int32")
    if last.shape != (q.shape[0],):
        raise ValueError("Expected one length or end per query row")
    if paged:
        if k.ndim != 4 or k.shape[1:] != (16, 1, 128) or k.shape[0] == 0:
            raise ValueError("Expected cache[P,16,1,128] with P > 0")
        if first.ndim != 2 or first.shape[0] != q.shape[0]:
            raise ValueError("Expected table[R,L]")
        if not isinstance(width, int) or width != first.shape[1] * 16:
            raise ValueError("Output width must equal table.shape[1] * 16")
    elif k.ndim != 3 or k.shape[1:] != (1, 128) or first.shape != last.shape:
        raise ValueError("Expected packed K[M,1,128] and starts[R]")


def packed(q, k, starts, ends):
    """BF16 packed K[M,1,128] -> FP32[R,M]; bounds are device int32 arrays."""
    _validate(q, k, starts, ends, False)
    rows, width = q.shape[0], k.shape[0]
    out = torch.empty((rows, width), device=q.device, dtype=torch.float32)
    if not rows or not width:
        return out
    # For K widths of at least 1024, four or more 256-key tiles amortize
    # matrix/vector handoffs. Q[4096,4,128], K[1024,1,128] uses 8x256 tiles;
    # Q[1,4,128], K[65536,1,128] uses four padded rows with the same width.
    # Short K[48,1,128] or K[257,1,128] keeps 16x128 tiles to avoid wasting
    # a wide key tile and splitting an already small query workload.
    # This changes tile geometry only; packed ranges and output stay identical.
    bq_limit, bn = (8, 256) if width >= 1024 else (16, 128)
    bq = min(bq_limit, max(4, triton.next_power_of_2(rows)))
    programs = min(_compute_cores(q.device.index),
                   triton.cdiv(rows, bq) * triton.cdiv(width, bn))
    _score[(programs,)](
        q, k, starts, ends, out, rows, width, 0, False, bq, bn, bq * 4, programs,
        enable_fp_fusion=False)
    return out


def paged(q, cache, table, lengths, width):
    """BF16 cache[P,16,1,128] -> FP32[R,16*L]; each row has its own page map."""
    _validate(q, cache, table, lengths, True, width)
    rows = q.shape[0]
    out = torch.empty((rows, width), device=q.device, dtype=torch.float32)
    if not rows or not width:
        return out
    # Q[32,4,128] can represent B8 verify-W4, but each row has its own map.
    # Logical columns address cache[P,16,1,128] through table[R,L]. Never
    # reinterpret this cache as packed K, even when R=1. Write all 16*L
    # columns so short and empty graph rows retain exact -inf padding.
    # A 256-key tile joins 16 logical pages through masked cache loads. Pad
    # four heads to 16 only inside the matrix tile; the API still requires H4.
    # R=129 uses this same scoring strategy; the bounded grid loops over rows.
    bn = 256
    programs = min(_compute_cores(q.device.index), rows * triton.cdiv(width, bn))
    _score[(programs,)](
        q, cache, table, lengths, out, rows, width, table.shape[1], True,
        1, bn, 16, programs, enable_fp_fusion=False)
    return out
