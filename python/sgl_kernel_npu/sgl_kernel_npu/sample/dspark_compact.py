"""Compact DSPark verify indexing, including graph bucket padding."""

import msgspec
import torch
import triton
import triton.language as tl
from sgl_kernel_npu.sample.dspark_ragged import BuildQoIndptr
from sgl_kernel_npu.utils.triton_utils import get_device_properties


class RaggedVerifyWindow(msgspec.Struct, frozen=True):
    positions: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_ids: torch.Tensor
    window_index: torch.Tensor


@triton.jit
def _compact_kernel(
    indptr,
    req_out,
    within_out,
    valid_out,
    n,
    bs,
    BLOCK: tl.constexpr,
    NBITS: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        x = tile * BLOCK + tl.arange(0, BLOCK)
        lo = tl.zeros((BLOCK,), tl.int32)
        hi = tl.full((BLOCK,), bs, tl.int32)
        for _ in range(NBITS):
            mid = (lo + hi) // 2
            val = tl.load(indptr + tl.minimum(mid + 1, bs), x < n, other=0)
            active = lo < hi
            right = val <= x
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        valid = x < tl.load(indptr + bs)
        start = tl.load(indptr + tl.minimum(lo, bs), x < n, other=0)
        tl.store(req_out + x, tl.where(valid, lo, bs).to(tl.int64), x < n)
        tl.store(within_out + x, tl.where(valid, x - start, 0).to(tl.int64), x < n)
        tl.store(valid_out + x, valid, x < n)


def compact_row_index_triton(*, verify_lens, padded_total, device):
    lens = verify_lens.to(device=device).contiguous()
    bs = lens.numel()
    assert bs > 0
    indptr = BuildQoIndptr.execute(verify_lens=lens).qo_indptr
    req = torch.empty(padded_total, dtype=torch.int64, device=device)
    within = torch.empty_like(req)
    valid = torch.empty(padded_total, dtype=torch.bool, device=device)
    _compact_kernel[
        (max(1, min(get_device_properties()[1], triton.cdiv(padded_total, 256))),)
    ](
        indptr,
        req,
        within,
        valid,
        padded_total,
        bs,
        BLOCK=256,
        NBITS=bs.bit_length() + 1,
        num_warps=1,
    )
    return req, within, valid


@triton.jit
def _ids_kernel(
    req, within, anchors, drafts, ids, n, bs, gamma, a0, a1, d0, d1, BLOCK: tl.constexpr
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        x = tile * BLOCK + tl.arange(0, BLOCK)
        row = tl.load(req + x, x < n, other=bs)
        col = tl.load(within + x, x < n, other=0)
        valid = (x < n) & (row < bs)
        anchor = tl.load(anchors + row * a0, valid & (col == 0), other=0)
        draft = tl.load(
            drafts + row * d0 + (col - 1) * d1,
            valid & (col > 0) & (col <= gamma),
            other=0,
        )
        # Select integer bits directly to preserve token ids on older toolchains.
        select = -(col == 0).to(tl.int64)
        value = (anchor.to(tl.int64) & select) | (draft.to(tl.int64) & ~select)
        tl.store(ids + x, value, x < n)


def compact_verify_ids_triton(*, draft_block_ids, draft_tokens, layout, device):
    req, within, _ = compact_row_index_triton(
        verify_lens=layout.verify_lens,
        padded_total=layout.graph_num_tokens,
        device=device,
    )
    n = layout.graph_num_tokens
    out = torch.empty(n, dtype=torch.int64, device=device)
    _ids_kernel[(max(1, min(get_device_properties()[1], triton.cdiv(n, 256))),)](
        req,
        within,
        draft_block_ids,
        draft_tokens,
        out,
        n,
        layout.verify_lens.numel(),
        draft_tokens.shape[1],
        draft_block_ids.stride(0),
        draft_block_ids.stride(1),
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        BLOCK=256,
        num_warps=1,
    )
    return out


@triton.jit
def _window_kernel(
    req,
    within,
    prefix,
    pool,
    table,
    pos,
    cache,
    index,
    n,
    bs,
    stride,
    table_stride,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        x = tile * BLOCK + tl.arange(0, BLOCK)
        row = tl.load(req + x, x < n, other=bs)
        col = tl.load(within + x, x < n, other=0)
        valid = (x < n) & (row < bs)
        start = tl.load(prefix + row, valid, other=0).to(tl.int64)
        slot = tl.load(pool + row, valid, other=0).to(tl.int64)
        location = tl.load(table + slot * table_stride + start + col, valid, other=0)
        tl.store(pos + x, tl.where(valid, start + col, 0), x < n)
        tl.store(cache + x, location, x < n)
        tl.store(index + x, tl.where(valid, row * stride + col, -1), x < n)


def build_ragged_verify_window_triton(
    *,
    batch,
    layout,
    draft_block_ids,
    draft_tokens,
    bs,
    device,
    verify_num_draft_tokens,
    model_runner
):
    n = layout.graph_num_tokens
    req, within, _ = compact_row_index_triton(
        verify_lens=layout.verify_lens, padded_total=n, device=device
    )
    positions = torch.empty(n, dtype=torch.int64, device=device)
    # Preserve the NPU assign_extend_cache_locs_func result dtype.
    cache = torch.empty(n, dtype=torch.int32, device=device)
    index = torch.empty_like(positions)
    table = model_runner.req_to_token_pool.req_to_token
    _window_kernel[(max(1, min(get_device_properties()[1], triton.cdiv(n, 256))),)](
        req,
        within,
        batch.seq_lens.contiguous(),
        batch.req_pool_indices.contiguous(),
        table,
        positions,
        cache,
        index,
        n,
        bs,
        verify_num_draft_tokens,
        table.stride(0),
        BLOCK=256,
        num_warps=1,
    )
    ids = compact_verify_ids_triton(
        draft_block_ids=draft_block_ids,
        draft_tokens=draft_tokens,
        layout=layout,
        device=device,
    )
    return RaggedVerifyWindow(
        positions=positions, verify_cache_loc=cache, verify_ids=ids, window_index=index
    )


class CompactRowIndex:
    execute = staticmethod(compact_row_index_triton)
    triton = staticmethod(compact_row_index_triton)


class CompactVerifyIds:
    execute = staticmethod(compact_verify_ids_triton)
    triton = staticmethod(compact_verify_ids_triton)


class BuildRaggedVerifyWindow:
    execute = staticmethod(build_ragged_verify_window_triton)
    triton = staticmethod(build_ragged_verify_window_triton)
