"""DSPark per-request ring metadata on Ascend vector cores."""

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _ring_indices(
    seq,
    req,
    floor,
    out,
    lengths,
    n,
    block_size: tl.constexpr,
    window: tl.constexpr,
    ring: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(n, BLOCK), tl.num_programs(0)):
        x = tile * BLOCK + tl.arange(0, BLOCK)
        q = x // width
        row = q // block_size
        col = x % width
        slot = tl.load(req + row * block_size, x < n, other=0).to(tl.int64)
        prefix = tl.load(seq + row * block_size, x < n, other=1).to(tl.int64) - 1
        valid_from = tl.load(floor + slot, x < n, other=0).to(tl.int64)
        start = tl.maximum(tl.maximum(prefix - window, valid_from), 0)
        context = tl.maximum(prefix - start, 0)
        pos = tl.where(col < context, start + col, prefix + col - context)
        value = slot * ring + pos % ring
        value = tl.where(col < context + block_size, value, -1)
        tl.store(out + x, value.to(tl.int32), x < n)
        tl.store(lengths + q, (context + block_size).to(tl.int32), (x < n) & (col == 0))


def build_dspark_swa_ring_indices(
    *,
    seq_lens_casual,
    req_pool_indices_repeated,
    valid_from,
    ring_size,
    block_size,
    swa_window,
    page_index_aligned_size
):
    count = seq_lens_casual.numel()
    assert count % block_size == 0
    width = (
        triton.cdiv(swa_window + block_size, page_index_aligned_size)
        * page_index_aligned_size
    )
    out = torch.empty((count, width), dtype=torch.int32, device=seq_lens_casual.device)
    lengths = torch.empty(count, dtype=torch.int32, device=seq_lens_casual.device)
    _ring_indices[
        (max(1, min(get_device_properties()[1], triton.cdiv(count * width, 256))),)
    ](
        seq_lens_casual.contiguous(),
        req_pool_indices_repeated.contiguous(),
        valid_from.contiguous(),
        out,
        lengths,
        count * width,
        block_size,
        swa_window,
        ring_size,
        width,
        BLOCK=256,
        num_warps=1,
    )
    return out, lengths


class BuildDsparkSwaRingIndices:
    execute = staticmethod(build_dspark_swa_ring_indices)
    triton = staticmethod(build_dspark_swa_ring_indices)
