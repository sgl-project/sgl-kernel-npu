"""Exact common-base sixteen-query grouping, with independent descriptors for unequal queries.
Pairs share work only if every logical selected block is equal, in the same order.
Causal tails remain per original query; duplicates are never removed.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=[4])
def _pack(Q, B, C, QQ, R):
    p = tl.program_id(0)
    kv = tl.program_id(1)
    part = tl.program_id(2)
    idx = tl.arange(0, 16384)
    head_local = idx // 256
    dim = idx % 256
    row = 16 * p + part * 8 + head_local // 8
    head = kv * 8 + head_local % 8
    x = tl.load(Q + (row * 16 + head) * 256 + dim, row < R, other=0)
    tl.store(QQ + (p * 256 + kv * 128 + part * 64) * 256 + idx, x)
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
):
    kv = tl.program_id(1)
    d = tl.arange(0, 256)
    t = tl.arange(0, 4)
    h = tl.arange(0, 8)
    req = tl.load(REQ).to(tl.int64)
    for row in range(tl.program_id(0), R, GRID):
        head = kv * 8 + h
        visible = BASE + row.to(tl.int32) + 1
        rem = ((visible % 4) + 4) % 4
        pos = visible - rem + t
        valid = (t < rem) & (pos >= 0) & (pos < LENGTH)
        safe_pos = tl.where(valid, pos, 0)
        slot = tl.load(T + req * STRIDE + safe_pos, valid, other=-1)
        valid = valid & (slot >= 0) & (slot < CAP)
        slot = tl.where(valid, slot, 0)
        q = tl.load(Q + (row * 16 + head[:, None]) * 256 + d[None, :]).to(tl.float32)
        k = tl.load(
            K + (slot[:, None] * 2 + kv) * 256 + d[None, :], valid[:, None], other=0
        ).to(tl.float32)
        logits = tl.sum(k[None, :, :] * q[:, None, :], 2) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        pair_head = kv * 128 + (row % 16) * 8 + h
        off = (row // 16) * 256 + pair_head
        lse = tl.load(L + off)
        peak = tl.maximum(lse, tl.max(logits, 1))
        peak = tl.where(peak == -float("inf"), 0.0, peak)
        w = tl.exp(logits - peak[:, None])
        bw = tl.exp(lse - peak)
        denominator = bw + tl.sum(w, 1)
        denominator = tl.where(denominator > 0, denominator, 1.0)
        value = tl.load(
            V + (slot[:, None] * 2 + kv) * 256 + d[None, :], valid[:, None], other=0
        ).to(tl.float32)
        base_out = tl.load(O + off[:, None] * 256 + d[None, :]).to(tl.float32)
        out = (
            bw[:, None] * base_out + tl.sum(w[:, :, None] * value[None, :, :], 1)
        ) / denominator[:, None]
        tl.store(Y + (row * 16 + head[:, None]) * 256 + d[None, :], out)


def qsa_prefill(
    q, k, v, blocks, table, req, length, base, scale=1 / 16, return_details=False
):
    """BF16 D256 ratio-4 prefill over 512 selected logical blocks per query.

    Q is [R,16,256], K/V [capacity,2,256]. Every nonnegative selected
    block expands into four token positions through the request table. Each
    query independently appends its causal remainder (0--3 tokens). Invalid
    blocks/physical slots are omitted; order and duplicates are preserved.
    request_row is a one-element int64 NPU tensor indexing a valid table row.
    length/base are CPU metadata; this API does no device-to-host reads.
    The function does not run the indexer or select Top-K.
    """
    if q.ndim != 3 or q.shape[1:] != (16, 256):
        raise ValueError("QSA prefill requires Q [R,16,256]")
    if k.ndim != 3 or k.shape[1:] != (2, 256) or v.shape != k.shape:
        raise ValueError("QSA prefill requires K,V [capacity,2,256]")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("QSA prefill requires BF16 Q/K/V")
    if q.device.type != "npu" or any(
        x.device != q.device for x in (k, v, blocks, table, req)
    ):
        raise ValueError("all inputs must share an NPU device")
    if any(not x.is_contiguous() for x in (q, k, v, blocks, req)):
        raise ValueError("Q/K/V, blocks and request row must be contiguous")
    if blocks.shape != (q.shape[0], 512) or blocks.dtype != torch.int32:
        raise ValueError("blocks must be int32 [R,512]")
    if (
        table.ndim != 2
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or table.shape[0] == 0
    ):
        raise ValueError("request table must be int32 with unit column stride")
    if req.shape != (1,) or req.dtype != torch.int64:
        raise ValueError("request row must be int64 [1]")
    if (
        not isinstance(length, int)
        or not isinstance(base, int)
        or not 0 <= length <= table.shape[1]
    ):
        raise ValueError("invalid CPU length/base metadata")
    if not -(2**31) <= base or base + q.shape[0] >= 2**31 or k.shape[0] >= 2**31:
        raise ValueError("metadata exceeds int32 indexing")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if length and not k.shape[0]:
        raise ValueError("nonempty sequences require a nonempty cache")
    r = q.shape[0]
    if length == 0 or r == 0:
        counts = torch.zeros((r * 16,), device=q.device, dtype=torch.int32)
        packed = torch.empty(((r + 15) // 16, 256, 256), device=q.device, dtype=q.dtype)
        if r:
            _pack[((r + 15) // 16, 2, 2)](q, blocks, counts, packed, r)
        out = torch.zeros_like(q)
        if return_details:
            return (
                out,
                counts,
                torch.zeros_like(packed),
                torch.full(
                    ((r + 15) // 16, 256),
                    -float("inf"),
                    device=q.device,
                    dtype=torch.float32,
                ),
            )
        return out
    counts = torch.empty((r * 16,), device=q.device, dtype=torch.int32)
    packed = torch.empty(((r + 15) // 16, 256, 256), device=q.device, dtype=q.dtype)
    _pack[((r + 15) // 16, 2, 2)](q, blocks, counts, packed, r)
    starts, runs, counts = torch.ops.npu.qsa_prefill_prepare(
        blocks, table, req, counts, length, base, k.shape[0]
    )
    base_out, lse = torch.ops.npu.qsa_prefill_runs(
        packed, k, v, starts, runs, counts, scale
    )
    out = torch.empty_like(q)
    _tail_merge[(256, 2)](
        q,
        k,
        v,
        table,
        req,
        base_out,
        lse,
        out,
        table.stride(0),
        length,
        base,
        k.shape[0],
        scale,
        r,
        256,
        enable_fp_fusion=False,
    )
    return (out, counts, base_out, lse) if return_details else out
