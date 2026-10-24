import math

import pytest
import torch
import torch_npu  # noqa: F401

from sgl_kernel_npu.mem_cache.allocator import alloc_extend_kernel


def _next_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _reference(prefix_lens, seq_lens, last_locs, free_pages, page_size):
    output = []
    free_page_index = 0
    for prefix, seq, last_loc in zip(prefix_lens, seq_lens, last_locs):
        first_page_end = min(seq, math.ceil(prefix / page_size) * page_size)
        num_part1 = first_page_end - prefix
        output.extend(range(last_loc + 1, last_loc + 1 + num_part1))

        if prefix + num_part1 == seq:
            continue

        full_page_end = seq // page_size * page_size
        num_part2 = full_page_end - math.ceil(prefix / page_size) * page_size
        for _ in range(num_part2 // page_size):
            page = free_pages[free_page_index]
            free_page_index += 1
            output.extend(range(page * page_size, (page + 1) * page_size))

        if prefix + num_part1 + num_part2 == seq:
            continue

        num_part3 = seq - full_page_end
        page = free_pages[free_page_index]
        free_page_index += 1
        output.extend(range(page * page_size, page * page_size + num_part3))

    return torch.tensor(output, dtype=torch.int64)


def _run_case(batch_size, page_size, unaligned_free_pages):
    prefix_lens = [(i * 53) % (page_size * 4 + 1) for i in range(batch_size)]
    extend_lens = [1 + (i * 131) % (page_size * 7 + 1) for i in range(batch_size)]
    if batch_size == 1:
        prefix_lens = [page_size]
        extend_lens = [page_size * 32 + 1]
    seq_lens = [prefix + extend for prefix, extend in zip(prefix_lens, extend_lens)]
    last_locs = [
        -1
        if prefix == 0
        else (10_000 + i) * page_size + (prefix - 1) % page_size
        for i, prefix in enumerate(prefix_lens)
    ]

    free_page_base = torch.arange(20_000, 40_001, dtype=torch.int64, device="npu")
    free_pages = free_page_base[1:] if unaligned_free_pages else free_page_base
    out = torch.empty(sum(extend_lens), dtype=torch.int64, device="npu")

    alloc_extend_kernel[(batch_size,)](
        torch.tensor(prefix_lens, dtype=torch.int64, device="npu"),
        torch.tensor(seq_lens, dtype=torch.int64, device="npu"),
        torch.tensor(last_locs, dtype=torch.int64, device="npu"),
        free_pages,
        out,
        _next_power_of_2(batch_size),
        page_size,
        _next_power_of_2(out.numel()),
    )
    torch.npu.synchronize()

    expected = _reference(
        prefix_lens,
        seq_lens,
        last_locs,
        free_pages.cpu().tolist(),
        page_size,
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("unaligned_free_pages", [False, True])
def test_alloc_extend_runtime_sizes(batch_size, unaligned_free_pages):
    _run_case(batch_size, page_size=128, unaligned_free_pages=unaligned_free_pages)


@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
def test_alloc_extend_page_sizes(page_size):
    _run_case(batch_size=5, page_size=page_size, unaligned_free_pages=True)
