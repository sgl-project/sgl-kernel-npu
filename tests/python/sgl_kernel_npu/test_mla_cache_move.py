"""Bit-exact MLA relocation, including overlapping ND/NZ cache slots."""

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.mla_cache_move import (
    _mla_move_indices_kernel,
    move_mla_kv_cache,
)


def pack(logical, use_nz, page):
    layers, slots, dim = logical.shape
    if use_nz:
        return (
            logical.reshape(layers, -1, page, dim // 16, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(layers, -1, page, 1, dim)
        )
    return logical.reshape(layers, -1, page, 1, dim)


@pytest.mark.parametrize("use_nz", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 17, 64, 129])
def test_move_mla_overlapping_snapshots(use_nz, dtype, count):
    layers, slots, page = 3, 512, 128
    generator = torch.Generator().manual_seed(42 + count)
    logical = [
        torch.randint(-32768, 32768, (layers, slots, dim), generator=generator).short()
        for dim in (512, 64)
    ]
    targets = torch.randperm(slots, generator=generator)[:count]
    sources = targets.roll(1)
    if count > 3:
        sources[0] = sources[1]  # repeated sources are legal
    buffers = [pack(x, use_nz, page).npu().view(dtype) for x in logical]
    expected = [x.clone() for x in logical]
    for x in expected:
        x.index_copy_(1, targets, x.index_select(1, sources))
    # Strided int32 indices avoid relying on a .long() or .contiguous() launch.
    src_loc = torch.zeros(count * 2, dtype=torch.int32, device="npu")
    dst_loc = torch.zeros_like(src_loc)
    src_loc[::2] = sources.to(torch.int32).npu()
    dst_loc[::2] = targets.to(torch.int32).npu()
    move_mla_kv_cache(*buffers, dst_loc[::2], src_loc[::2], page, use_nz)
    for actual, reference in zip(buffers, expected):
        torch.testing.assert_close(
            actual.view(torch.int16).cpu(),
            pack(reference, use_nz, page),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("use_nz", [False, True])
def test_move_mla_cross_page_cycle(use_nz):
    layers, slots, page = 2, 512, 128
    logical = [
        (torch.arange(layers * slots * dim).reshape(layers, slots, dim) % 997).short()
        for dim in (512, 64)
    ]
    source = torch.tensor([127, 128, 130])
    target = torch.tensor([128, 130, 127])
    buffers = [pack(x, use_nz, page).npu().view(torch.float16) for x in logical]
    for x in logical:
        x.index_copy_(1, target, x.index_select(1, source))
    move_mla_kv_cache(*buffers, target.npu(), source.npu(), page, use_nz)
    for actual, reference in zip(buffers, logical):
        torch.testing.assert_close(
            actual.view(torch.int16).cpu(),
            pack(reference, use_nz, page),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("large_slots", [False, True])
def test_nz_indices_preserve_large_physical_rows(large_slots):
    # Even int32 logical slots can expand past int32 physical NZ row limits.
    locations = [2**31 - 129, 2**31 - 128, 2**31 - 1]
    if large_slots:
        locations = [value + 512 for value in locations]
    source = torch.tensor(locations, dtype=torch.int64, device="npu")
    target = source.roll(1)
    outputs = [
        torch.empty(n, dtype=torch.int64, device="npu") for n in (96, 96, 12, 12)
    ]
    _mla_move_indices_kernel[(1,)](
        source,
        target,
        *outputs,
        3,
        SRC_STRIDE=1,
        TGT_STRIDE=1,
        K_TILES=32,
        R_TILES=4,
        PAGE=128,
        PAGE_SHIFT=7,
        LARGE_SLOTS=large_slots,
        BLOCK=256,
    )
    for actual, slots, tiles in zip(
        outputs, [locations, locations[-1:] + locations[:-1]] * 2, [32, 32, 4, 4]
    ):
        expected = []
        for location in slots:
            page, slot = divmod(location, 128)
            expected.extend(
                page * tiles * 128 + tile * 128 + slot for tile in range(tiles)
            )
        torch.testing.assert_close(actual.cpu(), torch.tensor(expected), rtol=0, atol=0)
