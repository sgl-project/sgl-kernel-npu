"""Triton kernels for the D256 QSA prefill pipeline.

Two helpers surround the Ascend C main attention:

- ``pack_queries`` reorders Q into the sixteen-query sharing-group layout the
  native kernel reads and flags query groups whose selected blocks are equal
  entry by entry, in the same order. Base attention is shared only within
  exact 16/8/4/2/1 groups.
- ``merge_causal_tail`` adds each query's independent causal remainder of 0--3
  tokens to the float32 base output and LSE.

The Ascend C entry point in ``operators/attention/qsa_prefill.py`` calls these
between ``torch.ops.npu.qsa_prefill_prepare`` and
``torch.ops.npu.qsa_prefill_runs``, so the complete implementation needs both
runtimes.
"""

import torch
import triton
import triton.language as tl

HEAD_DIM = 256
# Sixteen queries form one sharing group; the native kernel reads Q packed as
# [groups, 16 * H, 256] and the tail merge walks the same grid.
GROUP_QUERIES = 16
TAIL_MERGE_GRID = 256


@triton.jit(do_not_specialize=[4])
def _pack(
    Q, B, C, QQ, R, H: tl.constexpr, KV: tl.constexpr, WRITE_Q: tl.constexpr = True
):
    p = tl.program_id(0)
    kv = tl.program_id(1)
    part = tl.program_id(2)
    if WRITE_Q:
        # Copy one head row at a time. Source and destination rows are both 256
        # contiguous elements, so deriving the row and head as scalars keeps
        # every access affine. A single 8192-element gather over
        # (row * H + head) * 256 + dim instead compiles to an element-wise
        # gather that leaves the vector cores in a state which deadlocks a
        # following cross-core kernel and cannot be captured in an NPU graph.
        d = tl.arange(0, 256)
        group: tl.constexpr = H // KV
        for part_head in range(32):
            local_head = part * 32 + part_head
            if local_head < 16 * group:
                row = 16 * p + local_head // group
                head = kv * group + local_head % group
                x = tl.load(Q + (row * H + head) * 256 + d, row < R, other=0)
                tl.store(QQ + (p * 16 * H + kv * 16 * group + local_head) * 256 + d, x)
    if (kv == 0) & (part == 0):
        j = tl.arange(0, 512)
        b0 = tl.load(B + (16 * p + 0) * 512 + j, 16 * p + 0 < R, other=-1)
        b1 = tl.load(B + (16 * p + 1) * 512 + j, 16 * p + 1 < R, other=-1)
        b2 = tl.load(B + (16 * p + 2) * 512 + j, 16 * p + 2 < R, other=-1)
        b3 = tl.load(B + (16 * p + 3) * 512 + j, 16 * p + 3 < R, other=-1)
        b4 = tl.load(B + (16 * p + 4) * 512 + j, 16 * p + 4 < R, other=-1)
        b5 = tl.load(B + (16 * p + 5) * 512 + j, 16 * p + 5 < R, other=-1)
        b6 = tl.load(B + (16 * p + 6) * 512 + j, 16 * p + 6 < R, other=-1)
        b7 = tl.load(B + (16 * p + 7) * 512 + j, 16 * p + 7 < R, other=-1)
        b8 = tl.load(B + (16 * p + 8) * 512 + j, 16 * p + 8 < R, other=-1)
        b9 = tl.load(B + (16 * p + 9) * 512 + j, 16 * p + 9 < R, other=-1)
        b10 = tl.load(B + (16 * p + 10) * 512 + j, 16 * p + 10 < R, other=-1)
        b11 = tl.load(B + (16 * p + 11) * 512 + j, 16 * p + 11 < R, other=-1)
        b12 = tl.load(B + (16 * p + 12) * 512 + j, 16 * p + 12 < R, other=-1)
        b13 = tl.load(B + (16 * p + 13) * 512 + j, 16 * p + 13 < R, other=-1)
        b14 = tl.load(B + (16 * p + 14) * 512 + j, 16 * p + 14 < R, other=-1)
        b15 = tl.load(B + (16 * p + 15) * 512 + j, 16 * p + 15 < R, other=-1)
        e2_0 = (tl.sum((b0 != b1).to(tl.int32), 0) == 0) & (16 * p + 1 < R)
        e2_2 = (tl.sum((b2 != b3).to(tl.int32), 0) == 0) & (16 * p + 3 < R)
        e2_4 = (tl.sum((b4 != b5).to(tl.int32), 0) == 0) & (16 * p + 5 < R)
        e2_6 = (tl.sum((b6 != b7).to(tl.int32), 0) == 0) & (16 * p + 7 < R)
        e2_8 = (tl.sum((b8 != b9).to(tl.int32), 0) == 0) & (16 * p + 9 < R)
        e2_10 = (tl.sum((b10 != b11).to(tl.int32), 0) == 0) & (16 * p + 11 < R)
        e2_12 = (tl.sum((b12 != b13).to(tl.int32), 0) == 0) & (16 * p + 13 < R)
        e2_14 = (tl.sum((b14 != b15).to(tl.int32), 0) == 0) & (16 * p + 15 < R)
        e4_0 = e2_0 & e2_2 & (tl.sum((b0 != b2).to(tl.int32), 0) == 0)
        e4_4 = e2_4 & e2_6 & (tl.sum((b4 != b6).to(tl.int32), 0) == 0)
        e4_8 = e2_8 & e2_10 & (tl.sum((b8 != b10).to(tl.int32), 0) == 0)
        e4_12 = e2_12 & e2_14 & (tl.sum((b12 != b14).to(tl.int32), 0) == 0)
        e8_0 = e4_0 & e4_4 & (tl.sum((b0 != b4).to(tl.int32), 0) == 0)
        e8_8 = e4_8 & e4_12 & (tl.sum((b8 != b12).to(tl.int32), 0) == 0)
        e16_0 = e8_0 & e8_8 & (tl.sum((b0 != b8).to(tl.int32), 0) == 0)
        tl.store(C + (16 * p) * 16 + 1, e16_0.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 2, e8_0.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 3, e8_8.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 4, e4_0.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 5, e4_4.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 6, e4_8.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 7, e4_12.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 8, e2_0.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 9, e2_2.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 10, e2_4.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 11, e2_6.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 12, e2_8.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 13, e2_10.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 14, e2_12.to(tl.int32))
        tl.store(C + (16 * p) * 16 + 15, e2_14.to(tl.int32))


@triton.jit(do_not_specialize=[8, 9, 10, 11, 13], do_not_specialize_on_alignment=[4])
def _tail_merge(
    Q,
    K,
    V,
    T,
    REQ,
    O,
    L,
    Y,
    STRIDE,
    LENGTH,
    BASE,
    CAP,
    SCALE: tl.constexpr,
    R,
    GRID: tl.constexpr,
    H: tl.constexpr,
    KV: tl.constexpr,
    GROUP_P2: tl.constexpr,
):
    kv = tl.program_id(1)
    d = tl.arange(0, 256)
    t = tl.arange(0, 4)
    group: tl.constexpr = H // KV
    # tl.arange needs a power-of-two extent, so the caller passes
    # next_power_of_2(H // KV) and valid_head masks the surplus lanes.
    # Calling triton.next_power_of_2 inside the kernel works on triton-ascend
    # 3.2.2 but raises on 3.2.1, the version CANN 9.0.0 installs: H // KV is a
    # tl.constexpr there and the host helper calls arg.bit_length(), giving
    # "'constexpr' object has no attribute 'bit_length'".
    h = tl.arange(0, GROUP_P2)
    valid_head = h < group
    req = tl.load(REQ).to(tl.int64)
    for row in range(tl.program_id(0), R, GRID):
        head = kv * group + h
        visible = BASE + row.to(tl.int32) + 1
        rem = ((visible % 4) + 4) % 4
        pos = visible - rem + t
        valid = (t < rem) & (pos >= 0) & (pos < LENGTH)
        safe_pos = tl.where(valid, pos, 0)
        slot = tl.load(T + req * STRIDE + safe_pos, valid, other=-1)
        valid = valid & (slot >= 0) & (slot < CAP)
        slot = tl.where(valid, slot, 0)
        q = tl.load(
            Q + (row * H + head[:, None]) * 256 + d[None, :],
            valid_head[:, None],
            other=0,
        ).to(tl.float32)
        k = tl.load(
            K + (slot[:, None] * KV + kv) * 256 + d[None, :], valid[:, None], other=0
        ).to(tl.float32)
        logits = tl.sum(k[None, :, :] * q[:, None, :], 2) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        pair_head = kv * 16 * group + (row % 16) * group + h
        off = (row // 16) * 16 * H + pair_head
        lse = tl.load(L + off, valid_head, other=-float("inf"))
        peak = tl.maximum(lse, tl.max(logits, 1))
        peak = tl.where(peak == -float("inf"), 0.0, peak)
        w = tl.exp(logits - peak[:, None])
        bw = tl.exp(lse - peak)
        denominator = bw + tl.sum(w, 1)
        denominator = tl.where(denominator > 0, denominator, 1.0)
        value = tl.load(
            V + (slot[:, None] * KV + kv) * 256 + d[None, :], valid[:, None], other=0
        ).to(tl.float32)
        base_out = tl.load(
            O + off[:, None] * 256 + d[None, :], valid_head[:, None], other=0
        ).to(tl.float32)
        out = (
            bw[:, None] * base_out + tl.sum(w[:, :, None] * value[None, :, :], 1)
        ) / denominator[:, None]
        tl.store(
            Y + (row * H + head[:, None]) * 256 + d[None, :], out, valid_head[:, None]
        )


def pack_queries(q, blocks, kv_heads):
    """Pack Q per sharing group and flag groups with identical selections.

    Args:
        q (torch.Tensor): Queries of shape [R, H, 256], BF16, contiguous, NPU.
        blocks (torch.Tensor): Selected logical blocks, int32 [R, 512].
        kv_heads (int): Local KV head count; H must be a multiple of it.

    Returns:
        tuple: Packed Q of shape [ceil(R / 16), 16 * H, 256] with q's dtype,
        and the int32 sharing flags of shape [R * 16] that
        ``torch.ops.npu.qsa_prefill_prepare`` consumes at stride 16.
    """
    rows, heads = q.shape[0], q.shape[1]
    groups = (rows + GROUP_QUERIES - 1) // GROUP_QUERIES
    counts = torch.empty((rows * GROUP_QUERIES,), device=q.device, dtype=torch.int32)
    packed = torch.empty(
        (groups, GROUP_QUERIES * heads, HEAD_DIM), device=q.device, dtype=q.dtype
    )
    _pack[(groups, kv_heads, triton.cdiv(GROUP_QUERIES * (heads // kv_heads), 32))](
        q, blocks, counts, packed, rows, heads, kv_heads
    )
    return packed, counts


def merge_causal_tail(
    q, k, v, request_table, request_row, base_out, lse, length, base, scale
):
    """Merge each query's causal remainder into the float32 base attention.

    Args:
        q (torch.Tensor): Queries of shape [R, H, 256], BF16, contiguous, NPU.
        k (torch.Tensor): Key cache of shape [capacity, KV, 256], BF16.
        v (torch.Tensor): Value cache with k's shape, dtype and device.
        request_table (torch.Tensor): Slot table, int32, unit column stride.
        request_row (torch.Tensor): Contiguous int64 [1] row index.
        base_out (torch.Tensor): Float32 base output from the native kernel.
        lse (torch.Tensor): Float32 log-sum-exp from the native kernel.
        length (int): Sequence length in tokens.
        base (int): Logical position preceding q's first row.
        scale (float): Logit multiplier.

    Returns:
        torch.Tensor: Newly allocated output with q's shape, dtype and device.
    """
    out = torch.empty_like(q)
    _tail_merge[(TAIL_MERGE_GRID, k.shape[1])](
        q,
        k,
        v,
        request_table,
        request_row,
        base_out,
        lse,
        out,
        request_table.stride(0),
        length,
        base,
        k.shape[0],
        scale,
        q.shape[0],
        TAIL_MERGE_GRID,
        q.shape[1],
        k.shape[1],
        triton.next_power_of_2(q.shape[1] // k.shape[1]),
        enable_fp_fusion=False,
    )
    return out
