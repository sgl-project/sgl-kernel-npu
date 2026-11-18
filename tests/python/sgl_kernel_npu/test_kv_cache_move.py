"""Compare NPU row relocation with an independent CPU bit-pattern snapshot."""

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.mem_cache.kv_cache_move import move_kv_cache_rows
from sgl_kernel_npu.mem_cache.mla_cache_move import move_mla_kv_cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("count", [0, 1, 5])
@pytest.mark.parametrize("combined", [False, True])
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
            )
            for width in widths
        ]
        actual = [x.npu().view(dtype) for x in cpu]
    # Cross-page cycle, with a repeated source. Targets are unique.
    source = torch.tensor([3, 4, 8, 3, 10][:count], dtype=torch.long)
    target = torch.tensor([4, 8, 3, 9, 11][:count], dtype=torch.long)
    expected = [x.clone() for x in cpu]
    for result, original in zip(expected, cpu):
        for dst, src in zip(target.tolist(), source.tolist()):
            result[:, dst // page, dst % page] = original[:, src // page, src % page]
    src_storage = torch.zeros(count * 2, dtype=index_dtype, device="npu")
    dst_storage = torch.zeros_like(src_storage)
    src_storage[::2] = source.npu()
    dst_storage[::2] = target.npu()
    if combined:
        move_mla_kv_cache(*actual, dst_storage[::2], src_storage[::2], page)
    else:
        move_kv_cache_rows(*actual, dst_storage[::2], src_storage[::2])
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result.view(torch.int16).cpu(), reference, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("count", [17, 72])
def test_mha_qwen_width_across_blocks(dtype, count):
    layers, pages, page, heads, dim = 3, 3, 128, 4, 128
    generator = torch.Generator().manual_seed(2026)
    inputs = [
        torch.randint(
            -32768,
            32768,
            (layers, pages, page, heads, dim),
            generator=generator,
            dtype=torch.int16,
        )
        for _ in range(2)
    ]
    source = torch.arange(count, dtype=torch.int64) + page - 1
    target = source.roll(1)
    source[0] = source[1]
    expected = [value.clone() for value in inputs]
    for value, reference in zip(inputs, expected):
        for src, dst in zip(source.tolist(), target.tolist()):
            reference[:, dst // page, dst % page] = value[:, src // page, src % page]
    buffers = [value.npu().view(dtype) for value in inputs]
    move_kv_cache_rows(*buffers, target.npu(), source.npu())
    for actual, reference in zip(buffers, expected):
        torch.testing.assert_close(
            actual.view(torch.int16).cpu(), reference, rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
        )
        for width in widths
    ]
    # Four real moves, including a cross-page cycle and repeated source, plus
    # the repeated self-copy padding produced by a 6-by-12 acceptance array.
    source = torch.tensor([128, 129, 127, 128] + [0] * 68, dtype=torch.int64)
    target = torch.tensor([127, 128, 129, 130] + [0] * 68, dtype=torch.int64)
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
    if layout == "mha_nd":
        move_kv_cache_rows(*buffers, target.npu(), source.npu())
    else:
        move_mla_kv_cache(
            *buffers, target.npu(), source.npu(), page, layout == "mla_nz"
        )
    for actual, reference, original in zip(buffers, expected, logical):
        bits = actual.view(torch.int16).cpu()
        torch.testing.assert_close(bits, pack(reference), rtol=0, atol=0)
        # This also independently checks the physical row containing logical slot 0.
        if layout == "mla_nz":
            row_zero = bits.reshape(layers, pages, -1, page, 16)[:, 0, :, 0, :]
            row_zero = row_zero.reshape(layers, 1, -1)
        else:
            row_zero = bits[:, 0, 0]
        torch.testing.assert_close(row_zero, original[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
        )
        for width in widths
    ]
    buffers = [value.npu().view(dtype) for value in original]
    locations = torch.zeros(64, dtype=torch.int64, device="npu")
    for _ in range(3):
        if layout == "mha_nd":
            move_kv_cache_rows(*buffers, locations, locations)
        else:
            move_mla_kv_cache(*buffers, locations, locations, page, layout == "mla_nz")
    for actual, reference in zip(buffers, original):
        torch.testing.assert_close(
            actual.view(torch.int16).cpu(), reference, rtol=0, atol=0
        )
