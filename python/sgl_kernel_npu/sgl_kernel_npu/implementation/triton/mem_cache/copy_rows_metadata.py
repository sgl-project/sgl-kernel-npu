"""Triton implementation for sparse KV copy rows metadata metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows_metadata_kernel(
    source_rows,
    destination_rows,
    valid,
    src,
    dst,
    lengths,
    count,
    source_base_0,
    source_base_1,
    destination_base_0,
    destination_base_1,
    WIDTH_0: tl.constexpr,
    WIDTH_1: tl.constexpr,
    N: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    VALID_STRIDE: tl.constexpr,
    COMPONENT_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if COMPONENT_GRID:
        component = tl.program_id(1)
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        i = component * N + row
        mask = row < N
        first = component == 0
    else:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = i % N
        mask = i < 2 * N
        first = i < N
    live = tl.load(valid + row * VALID_STRIDE, mask, other=0) != 0
    s = tl.load(source_rows + row * SRC_STRIDE, mask, other=0).to(tl.int64)
    d = tl.load(destination_rows + row * DST_STRIDE, mask, other=0).to(tl.int64)
    s, d = tl.where(live, s, 0), tl.where(live, d, 0)
    width = tl.where(first, WIDTH_0, WIDTH_1)
    tl.store(src + i, tl.where(first, source_base_0, source_base_1) + s * width, mask)
    tl.store(
        dst + i,
        tl.where(first, destination_base_0, destination_base_1) + d * width,
        mask,
    )
    tl.store(lengths + i, tl.where(live, width, 0), mask)
    if tl.program_id(0) == 0:
        if COMPONENT_GRID:
            if tl.program_id(1) == 0:
                tl.store(count, N * 2)
        else:
            tl.store(count, N * 2)


def copy_rows_metadata(sources, destinations, source_rows, destination_rows, valid):
    """Build split CKV/KPE byte descriptors without touching host KV."""
    if len(sources) != 2 or len(destinations) != 2:
        raise ValueError("Fused MF descriptors require two split KV components")
    n = source_rows.numel()
    device = source_rows.device
    src = torch.empty(n * len(sources), dtype=torch.int64, device=device)
    dst = torch.empty_like(src)
    lengths = torch.empty(src.shape, dtype=torch.int32, device=device)
    count = torch.empty(1, dtype=torch.int32, device=device)
    # Keep component boundaries aligned to the 32-byte int32 output cache line.
    component_grid = n % 8 == 0
    grid = (triton.cdiv(n, 32), 2) if component_grid else (triton.cdiv(2 * n, 32),)
    _copy_rows_metadata_kernel[grid](
        source_rows,
        destination_rows,
        valid,
        src,
        dst,
        lengths,
        count,
        sources[0].data_ptr(),
        sources[1].data_ptr(),
        destinations[0].data_ptr(),
        destinations[1].data_ptr(),
        sources[0].shape[-1] * sources[0].element_size(),
        sources[1].shape[-1] * sources[1].element_size(),
        n,
        source_rows.stride(0),
        destination_rows.stride(0),
        valid.stride(0),
        component_grid,
        32,
    )
    return src, dst, lengths, count
