"""Public checksum tests with an independent zlib reference.

The raw-pointer API cannot recover owning tensors for a torch_native backend.
Tests retain the owners and compare their selected bytes with zlib instead.
The flow is generate_data -> check_descriptors -> check_performance. --impl
selects the public backend; the reference is always zlib (--impl-ref is unused).
"""

import zlib

import pytest
import torch
from tests.python.utils.common import get_test_impls, make_executor
from tests.python.utils.performance import check_performance

# BF16 Qwen3-32B TP2 K/V page descriptors: 128 tokens * 4 heads * 128 * 2 bytes.
# Two descriptors represent one layer's K and V; 8 pages cover 1024 tokens.
perf_shape = [(2, 1, 131072), (2, 8, 131072)]


def generate_data(shape, dtype=torch.uint8):
    descriptors, rows, width = shape
    torch.npu.manual_seed(42)
    if dtype == torch.uint8:
        tensors = [
            torch.randint(0, 256, (rows, width), dtype=dtype)
            for _ in range(descriptors)
        ]
    else:
        tensors = [torch.randn(rows, width, dtype=dtype) for _ in range(descriptors)]
    indices = [torch.arange(rows - 1, -1, -1, dtype=torch.int64) for _ in tensors]
    return tensors, [tensor[0].nbytes for tensor in tensors], indices


def reference(tensors, strides, indices):
    value = 1
    for tensor, stride, idx in zip(tensors, strides, indices):
        raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for row in idx.cpu().tolist():
            value = zlib.adler32(raw[row * stride : (row + 1) * stride], value)
    return value


def check_descriptors(tensors, strides, indices, *, enable_perf=False):
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    args = ([tensor.data_ptr() for tensor in tensors], strides, indices)
    executor = make_executor(adler32_strided_checksum, args=args, impl_ref="zlib")
    expected = reference(tensors, strides, indices)
    actual = executor()
    assert actual == expected
    executor.accuracy_checked = True
    check_performance(executor, enable_perf=enable_perf)
    return actual


@pytest.mark.parametrize("case", ["no_descriptors", "empty_indices", "zero_stride"])
def test_empty(case):
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    impl = get_test_impls()[0]
    if case == "no_descriptors":
        args = ([], [], [])
    elif case == "empty_indices":
        args = ([0], [64], [torch.empty(0, dtype=torch.int64)])
    else:
        args = ([0], [0], [torch.tensor([0], dtype=torch.int32)])
    assert adler32_strided_checksum(*args, impl=impl) == 1


@pytest.mark.parametrize("stride", [1, 31, 1023, 1024, 1025, 65521, 70001])
def test_byte_tails_order_and_duplicates(stride):
    tensors, strides, _ = generate_data((1, 8, stride))
    indices = [torch.tensor([7, 0, 3, 3, 1], dtype=torch.int64)]
    check_descriptors(tensors, strides, indices)


def test_multiple_dtypes_and_empty_component():
    tensors = [
        generate_data((1, 9, 35), dtype)[0][0]
        for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ]
    indices = [
        torch.tensor(rows, dtype=torch.int32) for rows in ([8, 1, 4], [], [2, 0])
    ]
    check_descriptors(tensors, [tensor[0].nbytes for tensor in tensors], indices)


def test_modular_overflow():
    tensors, strides, indices = generate_data((1, 257, 4097))
    tensors[0].fill_(255)
    check_descriptors(tensors, strides, indices)


def test_selection_and_corruption():
    data = torch.arange(256, dtype=torch.int32).view(8, 32)
    indices = [torch.tensor([5, 1], dtype=torch.int64)]
    first = check_descriptors([data], [128], indices)
    data[0, 0] += 1
    assert first == check_descriptors([data], [128], indices)
    data[5, 0] += 1
    changed = check_descriptors([data], [128], indices)
    assert first != changed
    assert changed != check_descriptors([data], [128], [indices[0].flip(0)])


@pytest.mark.parametrize("invalid", ["lengths", "rank", "dtype", "cpu"])
def test_invalid_descriptors(invalid):
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    tensors, strides, indices = generate_data((1, 2, 64))
    if invalid == "lengths":
        strides = []
    elif invalid == "rank":
        indices = [indices[0].view(1, 2)]
    elif invalid == "dtype":
        indices = [indices[0].to(torch.float32)]
    else:
        indices = [indices[0].cpu()]
    with pytest.raises(ValueError):
        adler32_strided_checksum(
            [tensors[0].data_ptr()], strides, indices, impl=get_test_impls()[0]
        )


@pytest.mark.parametrize(
    "shape", perf_shape, ids=["qwen3-tp2-128-tokens", "qwen3-tp2-1024-tokens"]
)
def test_page_descriptors(shape):
    tensors, strides, indices = generate_data(shape)
    check_descriptors(tensors, strides, indices, enable_perf=shape in perf_shape)


def test_public_implementations():
    from sgl_kernel_npu.operators import get_impl, list_impls
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    assert list_impls(adler32_strided_checksum) == ["auto", "triton"]
    assert get_impl(adler32_strided_checksum, "auto") is get_impl(
        adler32_strided_checksum, "triton"
    )
    with pytest.raises(ValueError, match="Unknown impl"):
        adler32_strided_checksum([], [], [], impl="missing")
