"""Compare NPU row relocation with an independent CPU bit-pattern snapshot."""

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.operators.mem_cache import move_kv_cache_rows, move_mla_kv_cache

from tests.python.operators.mem_cache.ngram_move_utils import check_move


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["strided-i32", "strided-i64"]
)
@pytest.mark.parametrize(
    "count",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single-cross-page", marks=pytest.mark.extra_checks),
        pytest.param(5, id="cycle-repeated-source"),
    ],
)
@pytest.mark.parametrize("combined", [False, True], ids=["mha-split", "mla-combined"])
def test_page_rows_snapshot(dtype, index_dtype, count, combined):
    layers, pages, page = 3, 3, 4
    heads, widths = (1, (7, 5)) if combined else (3, (7, 5))
    generator = torch.Generator().manual_seed(123)
    if combined:
        storage = torch.randint(
            -32768,
            32768,
            (layers, pages, page, 1, sum(widths)),
            generator=generator,
            dtype=torch.int16,
            device="cpu",
        )
        cpu = [storage[..., : widths[0]], storage[..., widths[0] :]]
        device_storage = storage.npu().view(dtype)
        actual = [device_storage[..., : widths[0]], device_storage[..., widths[0] :]]
    else:
        cpu = [
            torch.randint(
                -32768,
                32768,
                (layers, pages, page, heads, width),
                generator=generator,
                dtype=torch.int16,
                device="cpu",
            )
            for width in widths
        ]
        actual = [x.npu().view(dtype) for x in cpu]
    # Cross-page cycle, with a repeated source. Targets are unique.
    source = torch.tensor([3, 4, 8, 3, 10][:count], dtype=torch.long, device="cpu")
    target = torch.tensor([4, 8, 3, 9, 11][:count], dtype=torch.long, device="cpu")
    expected = [x.clone() for x in cpu]
    for result, original in zip(expected, cpu):
        for dst, src in zip(target.tolist(), source.tolist()):
            result[:, dst // page, dst % page] = original[:, src // page, src % page]
    src_storage = torch.zeros(count * 2, dtype=index_dtype, device="npu")
    dst_storage = torch.zeros_like(src_storage)
    src_storage[::2] = source.npu()
    dst_storage[::2] = target.npu()
    check_move(
        move_mla_kv_cache if combined else move_kv_cache_rows,
        actual,
        (dst_storage[::2], src_storage[::2]),
        (page,) if combined else (),
        expected,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "count",
    [
        pytest.param(17, id="tile-tail", marks=pytest.mark.extra_checks),
        pytest.param(72, id="qwen-serving-b6-d12"),
    ],
)
def test_mha_qwen_width_across_blocks(dtype, count):
    # Qwen3-30B-A3B dimensions and six requests * 12 slots; reduced page count.
    layers, pages, page, heads, dim = 48, 3, 128, 4, 128
    generator = torch.Generator().manual_seed(2026)
    inputs = [
        torch.randint(
            -32768,
            32768,
            (layers, pages, page, heads, dim),
            generator=generator,
            dtype=torch.int16,
            device="cpu",
        )
        for _ in range(2)
    ]
    source = torch.arange(count, dtype=torch.int64, device="cpu") + page - 1
    target = source.roll(1)
    source[0] = source[1]
    expected = [value.clone() for value in inputs]
    for value, reference in zip(inputs, expected):
        for src, dst in zip(source.tolist(), target.tolist()):
            reference[:, dst // page, dst % page] = value[:, src // page, src % page]
    buffers = [value.npu().view(dtype) for value in inputs]
    check_move(
        move_kv_cache_rows,
        buffers,
        (target.npu(), source.npu()),
        (),
        expected,
        perf=count == 72 and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["mha_nd", "mla_nd", "mla_nz"])
def test_padded_self_copies_with_cycles(dtype, layout):
    layers, pages, page = 3, 3, 128
    heads, widths = (4, (128, 128)) if layout == "mha_nd" else (1, (512, 64))
    generator = torch.Generator().manual_seed(981)
    logical = [
        torch.randint(
            -32768,
            32768,
            (layers, pages * page, heads, width),
            generator=generator,
            dtype=torch.int16,
            device="cpu",
        )
        for width in widths
    ]
    # Four real moves, including a cross-page cycle and repeated source, plus
    # the repeated self-copy padding produced by a 6-by-12 acceptance array.
    source = torch.tensor(
        [128, 129, 127, 128] + [0] * 68, dtype=torch.int64, device="cpu"
    )
    target = torch.tensor(
        [127, 128, 129, 130] + [0] * 68, dtype=torch.int64, device="cpu"
    )
    expected = [value.clone() for value in logical]
    for original, reference in zip(logical, expected):
        for src, dst in zip(source.tolist(), target.tolist()):
            if src != dst:
                reference[:, dst] = original[:, src]

    def pack(value):
        if layout == "mla_nz":
            return (
                value.reshape(layers, pages, page, -1, 16)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(layers, pages, page, 1, -1)
            )
        return value.reshape(layers, pages, page, heads, -1)

    buffers = [pack(value).npu().view(dtype) for value in logical]
    check_move(
        move_kv_cache_rows if layout == "mha_nd" else move_mla_kv_cache,
        buffers,
        (target.npu(), source.npu()),
        () if layout == "mha_nd" else (page, layout == "mla_nz"),
        [pack(value) for value in expected],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["mha_nd", "mla_nd", "mla_nz"])
def test_all_rejected_padding_repeated_calls(dtype, layout):
    layers, pages, page = 2, 2, 128
    heads, widths = (4, (128, 128)) if layout == "mha_nd" else (1, (512, 64))
    generator = torch.Generator().manual_seed(561)
    original = [
        torch.randint(
            -32768,
            32768,
            (layers, pages, page, heads, width),
            generator=generator,
            dtype=torch.int16,
            device="cpu",
        )
        for width in widths
    ]
    buffers = [value.npu().view(dtype) for value in original]
    locations = torch.zeros(64, dtype=torch.int64, device="npu")
    check_move(
        move_kv_cache_rows if layout == "mha_nd" else move_mla_kv_cache,
        buffers,
        (locations, locations),
        () if layout == "mha_nd" else (page, layout == "mla_nz"),
        original,
        repeats=3,
    )
