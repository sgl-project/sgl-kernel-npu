"""Snapshot-safe relocation of 16-bit NPU cache rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_rows(
    a,
    b,
    scratch,
    target,
    source,
    COUNT: tl.constexpr,
    A_WIDTH: tl.constexpr,
    B_WIDTH: tl.constexpr,
    LAYERS: tl.constexpr,
    A_LAYER: tl.constexpr,
    A_ROW: tl.constexpr,
    B_LAYER: tl.constexpr,
    B_ROW: tl.constexpr,
    TARGET_STRIDE: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SCATTER: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    layer = tl.program_id(2).to(tl.int64)
    rows = tl.program_id(1)
    cols = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    ac = tl.where(cols < A_WIDTH, cols, 0)
    bc = tl.where(cols < B_WIDTH, cols, 0)
    am = cols < A_WIDTH
    bm = cols < B_WIDTH
    original = tl.load(source + rows * SOURCE_STRIDE).to(tl.int64)
    if SCATTER:
        index = tl.load(target + rows * TARGET_STRIDE).to(tl.int64)
    else:
        index = original
    ao = layer * A_LAYER + index * A_ROW + ac
    bo = layer * B_LAYER + index * B_ROW + bc
    ax = (layer * COUNT + rows.to(tl.int64)) * A_WIDTH + ac
    bx = (layer * COUNT + rows.to(tl.int64)) * B_WIDTH + bc
    b_offset = LAYERS * COUNT * A_WIDTH
    if SCATTER:
        av = tl.load(scratch + ax, am, other=0)
        bv = tl.load(scratch + b_offset + bx, bm, other=0)
        tl.store(a + ao, av, am & (index != original))
        tl.store(b + bo, bv, bm & (index != original))
    else:
        av = tl.load(a + ao, am, other=0)
        bv = tl.load(b + bo, bm, other=0)
        tl.store(scratch + ax, av, am)
        tl.store(scratch + b_offset + bx, bv, bm)


def _move_cache_rows(a, b, tgt_loc, src_loc, count):
    """Snapshot all sources; non-identity targets are unique, self-copies may repeat."""
    a, b = a.view(torch.int16), b.view(torch.int16)
    layers, _, aw = a.shape
    bw = b.shape[-1]
    scratch = torch.empty(
        layers * count * (aw + bw), dtype=torch.int16, device=a.device
    )
    col_block = min(triton.next_power_of_2(max(aw, bw)), 512)
    grid = (
        triton.cdiv(max(aw, bw), col_block),
        count,
        layers,
    )
    common = dict(
        COUNT=count,
        A_WIDTH=aw,
        B_WIDTH=bw,
        LAYERS=layers,
        A_LAYER=a.stride(0),
        A_ROW=a.stride(1),
        B_LAYER=b.stride(0),
        B_ROW=b.stride(1),
        TARGET_STRIDE=tgt_loc.stride(0),
        SOURCE_STRIDE=src_loc.stride(0),
        COL_BLOCK=col_block,
    )
    _copy_rows[grid](
        a,
        b,
        scratch,
        tgt_loc,
        src_loc,
        SCATTER=False,
        **common,
    )
    _copy_rows[grid](
        a,
        b,
        scratch,
        tgt_loc,
        src_loc,
        SCATTER=True,
        **common,
    )


def _page_rows(cache):
    return cache.as_strided(
        (
            cache.shape[0],
            cache.shape[1] * cache.shape[2],
            cache.shape[3] * cache.shape[4],
        ),
        (cache.stride(0), cache.stride(2), 1),
    )


def move_kv_cache_rows(keys, values, tgt_loc, src_loc):
    """Move page-major MHA rows while preserving their 16-bit payloads."""
    count = src_loc.numel()
    if count == 0:
        return
    _move_cache_rows(_page_rows(keys), _page_rows(values), tgt_loc, src_loc, count)
