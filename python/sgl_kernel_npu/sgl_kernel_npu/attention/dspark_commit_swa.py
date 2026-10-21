"""DSPark Triton-Ascend kernels.

Ported from sgl-project/sglang at 7036f4b0ad. GPU references stay in SGLang.
Launches use persistent vector-core grids; no device-dependent torch fallback.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _committed_swa_locations(
    LOC,
    MAP,
    LENS,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    _grid_0: tl.constexpr,
):
    for _job in range(tl.program_id(0), _grid_0, tl.num_programs(0)):
        i = _job * BLOCK + tl.arange(0, BLOCK)
        length = tl.load(LENS + i // WIDTH, i < N, other=0)
        committed = (i < N) & (i % WIDTH < length)
        loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
        loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
        swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
        tl.store(OUT + i, swa, i < N)


def committed_swa_locations(cache_loc, full_to_swa_mapping, commit_lens, width):
    assert cache_loc.numel() == commit_lens.numel() * width
    out = torch.empty_like(cache_loc, dtype=torch.int32)
    if cache_loc.numel():
        _committed_swa_locations[
            max(
                1, min(get_device_properties()[1], triton.cdiv(cache_loc.numel(), 256))
            ),
        ](
            cache_loc,
            full_to_swa_mapping,
            commit_lens,
            out,
            cache_loc.numel(),
            width,
            full_to_swa_mapping.numel(),
            256,
            _grid_0=triton.cdiv(cache_loc.numel(), 256),
            num_warps=1,
        )
    return out
