"""Exact common-base sixteen-query grouping, with independent descriptors for unequal queries.

Pairs share work only if every logical selected block is equal, in the same order.
Causal tails remain per original query; duplicates are never removed.

The QK/online-softmax/PV/rescale pipeline and physical run preparation are
Ascend C (``torch.ops.npu.qsa_prefill_prepare`` and
``torch.ops.npu.qsa_prefill_runs``). Query packing, full selection equality
and the per-query causal tail merge are Triton NPU kernels, so this
implementation orchestrates both runtimes instead of forwarding a single
native binding. The public operator validates the shared contract before
calling in; see ``operators/attention/qsa_prefill.py``.
"""

import torch
import triton
import triton.language as tl


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
):
    kv = tl.program_id(1)
    d = tl.arange(0, 256)
    t = tl.arange(0, 4)
    group: tl.constexpr = H // KV
    h = tl.arange(0, triton.next_power_of_2(group))
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


# Head layouts the native pipeline launches directly.
NATIVE_HEAD_LAYOUTS = ((16, 2), (8, 1), (6, 1), (3, 1), (2, 1))
# A sharing group of 16 queries occupies 16 * (H / KV) rows of the per-stage GM
# workspace block, which holds 128 rows, so the native kernel requires
# H / KV <= 8. Wider layouts are split into slices of this many query heads.
SPLIT_QUERY_HEADS = 6


def qsa_prefill(q, k, v, blocks, request_table, request_row, length, base, scale):
    """Run base attention natively, splitting layouts the kernel cannot take.

    Attention is independent per query head, so a layout whose queries per KV
    head exceed the kernel's limit is evaluated as several native launches
    over consecutive query-head slices and concatenated in head order. The
    slices are copied because they are not contiguous; with more than one KV
    head, each KV head is also copied once and reused across its slices.
    """
    heads, kv_heads = q.shape[1], k.shape[1]
    if (heads, kv_heads) in NATIVE_HEAD_LAYOUTS:
        return _launch(q, k, v, blocks, request_table, request_row, length, base, scale)
    group = heads // kv_heads
    parts = []
    for kv_head in range(kv_heads):
        if kv_heads == 1:
            keys, values = k, v
        else:
            keys = k[:, kv_head : kv_head + 1].contiguous()
            values = v[:, kv_head : kv_head + 1].contiguous()
        for start in range(0, group, SPLIT_QUERY_HEADS):
            head = kv_head * group + start
            parts.append(
                _launch(
                    q[:, head : head + SPLIT_QUERY_HEADS].contiguous(),
                    keys,
                    values,
                    blocks,
                    request_table,
                    request_row,
                    length,
                    base,
                    scale,
                )
            )
    return torch.cat(parts, dim=1)


def _launch(q, k, v, blocks, request_table, request_row, length, base, scale):
    """Run one native launch for a layout in NATIVE_HEAD_LAYOUTS.

    The public operator has already validated shapes, dtypes, the head layout,
    device placement, contiguity and the CPU metadata range. Tiling is
    immutable and cached natively by full shape/scale/device values, including
    both local head counts. No device-to-host metadata read happens here.
    """
    rows, heads = q.shape[0], q.shape[1]
    kv_heads = k.shape[1]
    if length == 0 or rows == 0:
        # Without visible tokens every row's attention output is zero, and no
        # packed-Q write is queued against storage the caller may reuse.
        return torch.zeros_like(q)
    groups = (rows + 15) // 16
    counts = torch.empty((rows * 16,), device=q.device, dtype=torch.int32)
    packed = torch.empty((groups, 16 * heads, 256), device=q.device, dtype=q.dtype)
    _pack[(groups, kv_heads, triton.cdiv(16 * (heads // kv_heads), 32))](
        q, blocks, counts, packed, rows, heads, kv_heads
    )
    starts, runs, counts = torch.ops.npu.qsa_prefill_prepare(
        blocks, request_table, request_row, counts, length, base, k.shape[0]
    )
    base_out, lse = torch.ops.npu.qsa_prefill_runs(
        packed, k, v, starts, runs, counts, scale
    )
    out = torch.empty_like(q)
    _tail_merge[(256, kv_heads)](
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
        rows,
        256,
        heads,
        kv_heads,
        enable_fp_fusion=False,
    )
    return out
