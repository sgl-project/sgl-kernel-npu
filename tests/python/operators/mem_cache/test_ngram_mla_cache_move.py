"""Bit-exact MLA relocation, including overlapping ND/NZ cache slots."""

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.mem_cache import move_mla_kv_cache

from tests.python.operators.mem_cache.ngram_move_utils import check_move


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


@pytest.mark.parametrize("use_nz", [False, True], ids=["layout-ND", "layout-NZ"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "count",
    [
        pytest.param(0, id="empty"),
        pytest.param(12, id="deepseek-serving-d12"),
        *[
            pytest.param(n, id=f"cycle-{n}-slots", marks=pytest.mark.extra_checks)
            for n in (1, 3, 4, 5, 17, 64, 129)
        ],
    ],
)
def test_move_mla_overlapping_snapshots(use_nz, dtype, count):
    # DeepSeek-V2-Lite uses 27 layers and 512/64 latent/RoPE widths.
    layers, slots, page = 27, 512, 128
    generator = torch.Generator().manual_seed(42 + count)
    logical = [
        torch.randint(
            -32768, 32768, (layers, slots, dim), generator=generator, device="cpu"
        ).short()
        for dim in (512, 64)
    ]
    targets = torch.randperm(slots, generator=generator, device="cpu")[:count]
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
    check_move(
        move_mla_kv_cache,
        buffers,
        (dst_loc[::2], src_loc[::2]),
        (page, use_nz),
        [pack(value, use_nz, page) for value in expected],
        perf=count == 12 and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize("use_nz", [False, True], ids=["layout-ND", "layout-NZ"])
def test_move_mla_cross_page_cycle(use_nz):
    layers, slots, page = 2, 512, 128
    logical = [
        (
            torch.arange(layers * slots * dim, device="cpu").reshape(layers, slots, dim)
            % 997
        ).short()
        for dim in (512, 64)
    ]
    source = torch.tensor([127, 128, 130], device="cpu")
    target = torch.tensor([128, 130, 127], device="cpu")
    buffers = [pack(x, use_nz, page).npu().view(torch.float16) for x in logical]
    for x in logical:
        x.index_copy_(1, target, x.index_select(1, source))
    check_move(
        move_mla_kv_cache,
        buffers,
        (target.npu(), source.npu()),
        (page, use_nz),
        [pack(value, use_nz, page) for value in logical],
    )
