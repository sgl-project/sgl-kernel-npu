"""Triton implementation for sparse KV prefill plan metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_plan_kernel(
    requests,
    lengths,
    table,
    source,
    destination,
    valid,
    metadata,
    current_valid,
    current_src,
    current_dst,
    current_out_valid,
    host_mask,
    HAS_CURRENT: tl.constexpr,
    CURRENT_VALID_STRIDE: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    CONTEXT: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    LEN_STRIDE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    ids = tl.arange(0, BATCH_BLOCK)
    reqs = tl.load(requests + ids * REQ_STRIDE, ids < B, other=-1)
    lens = tl.load(lengths + ids * LEN_STRIDE, ids < B, other=0).to(tl.int64)
    lens = tl.where((reqs >= 0) & (lens > 0), lens, 0)
    start = tl.sum(tl.where(ids < r, lens, 0), 0)
    req = tl.load(requests + r * REQ_STRIDE).to(tl.int64)
    length = tl.load(lengths + r * LEN_STRIDE)
    pos = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = (req >= 0) & (length > 0) & (pos < length)
    safe_req = tl.minimum(tl.maximum(req, 0), ROWS - 1)
    safe_pos = tl.minimum(pos, CONTEXT - 1)
    if HAS_CURRENT:
        current_start = tl.load(metadata + r)
        prefix = tl.load(metadata + B + r)
        extend = tl.load(metadata + 2 * B + r)
        width = tl.load(metadata + 3 * B + r)
        block_start = tl.program_id(1) * BLOCK
        use_device = tl.full((BLOCK,), False, tl.int1)
        if (block_start < prefix + extend) & (block_start + BLOCK > prefix):
            in_current = (pos >= prefix) & (pos < prefix + extend)
            idx = current_start + pos - prefix
            current_live = (
                tl.load(
                    current_valid + idx * CURRENT_VALID_STRIDE,
                    live & in_current,
                    other=0,
                )
                != 0
            )
            use_device = live & in_current & (req > 0) & current_live
        tl.store(host_mask + start + pos, ~use_device, live)
        if block_start < width:
            local_valid = (
                tl.load(
                    current_valid + (current_start + pos) * CURRENT_VALID_STRIDE,
                    pos < width,
                    other=0,
                )
                != 0
            )
            local_live = (pos < extend) & (req > 0) & local_valid
            tl.store(
                current_src + current_start + pos, current_start + pos, pos < width
            )
            tl.store(
                current_dst + current_start + pos,
                tl.where(local_live, start + prefix + pos, 0),
                pos < width,
            )
            tl.store(current_out_valid + current_start + pos, local_live, pos < width)
    physical = tl.load(table + safe_req * S0 + safe_pos * S1, live, other=0)
    out = start + pos
    tl.store(source + out, physical.to(tl.int64), live)
    tl.store(destination + out, out, live)
    tl.store(valid + out, True, live)


def prefill_plan(
    requests,
    lengths,
    table,
    total,
    max_length,
    request_rows,
    context_len,
    *,
    current_metadata=None,
    current_valid=None,
    max_current_width=0,
):
    """Build host and optional current-device row mappings in one launch."""
    source = torch.empty(total, dtype=torch.long, device=table.device)
    destination = torch.empty_like(source)
    valid = torch.empty(total, dtype=torch.bool, device=table.device)
    has_current = current_metadata is not None
    if has_current != (current_valid is not None):
        raise ValueError("current metadata and validity must be supplied together")
    mapping = None
    if has_current:
        n = current_valid.numel()
        current_src = torch.empty(n, dtype=torch.long, device=table.device)
        current_dst = torch.empty_like(current_src)
        current_out_valid = torch.empty(n, dtype=torch.bool, device=table.device)
        host_mask = torch.empty(total, dtype=torch.bool, device=table.device)
        mapping = (current_src, current_dst, current_out_valid, host_mask)
    else:
        current_src = current_dst = current_out_valid = host_mask = None
    width = max(max_length, max_current_width)
    if width and requests.numel():
        b = requests.numel()
        _prefill_plan_kernel[(b, triton.cdiv(width, 256))](
            requests,
            lengths,
            table,
            source,
            destination,
            valid,
            current_metadata,
            current_valid,
            current_src,
            current_dst,
            current_out_valid,
            host_mask,
            has_current,
            current_valid.stride(0) if has_current else 0,
            b,
            request_rows,
            context_len,
            table.stride(0),
            table.stride(1),
            requests.stride(0),
            lengths.stride(0),
            triton.next_power_of_2(b),
            256,
        )
    return source, destination, valid, mapping
