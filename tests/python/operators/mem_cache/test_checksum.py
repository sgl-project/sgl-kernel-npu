"""Public checksum accuracy and performance tests against torch_native."""

import pytest
import torch

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance

# BF16 Qwen3-32B TP2 K/V page descriptors: 128 tokens * 4 heads * 128 * 2 bytes.
# Two descriptors represent one layer's K and V; 8 pages cover 1024 tokens.
perf_shape = [(2, 1, 131072), (2, 8, 131072)]


def generate_data(shape, dtypes=(torch.uint8,), mode="normal", index_dtype=torch.int64):
    descriptors, rows, width = shape
    torch.npu.manual_seed(42)
    tensors = []
    for i in range(descriptors):
        dtype = dtypes[i % len(dtypes)]
        if mode == "selection":
            tensor = torch.arange(rows * width, dtype=dtype).reshape(rows, width)
        elif dtype == torch.uint8:
            tensor = torch.randint(0, 256, (rows, width), dtype=dtype)
        else:
            tensor = torch.randn(rows, width, dtype=dtype)
        if mode == "overflow":
            tensor.fill_(255)
        tensors.append(tensor)
    strides = [tensor[0].nbytes for tensor in tensors]
    indices = [torch.arange(rows - 1, -1, -1, dtype=index_dtype) for _ in tensors]
    if mode == "duplicates":
        indices = [torch.tensor([7, 0, 3, 3, 1], dtype=index_dtype)]
    elif mode == "mixed_empty":
        indices = [
            torch.tensor(selected, dtype=index_dtype)
            for selected in ([8, 1, 4], [], [2, 0])
        ]
    elif mode == "empty_indices":
        indices = [torch.empty(0, dtype=index_dtype)]
    elif mode == "zero_stride":
        strides = [0]
    elif mode == "selection":
        indices = [torch.tensor([5, 1], dtype=index_dtype)]
    return tensors, strides, indices


@pytest.mark.parametrize(
    "shape,dtypes,mode,index_dtype",
    [
        pytest.param(
            perf_shape[0],
            (torch.uint8,),
            "normal",
            torch.int64,
            id="qwen3-tp2-128-tokens",
        ),
        pytest.param(
            perf_shape[1],
            (torch.uint8,),
            "normal",
            torch.int64,
            id="qwen3-tp2-1024-tokens",
        ),
        *[
            pytest.param(
                (1, 8, stride),
                (torch.uint8,),
                "duplicates",
                torch.int64,
                id=f"byte-tail-{stride}",
            )
            for stride in (1, 31, 1023, 1024, 1025, 4095, 4096, 4097, 65521, 70001)
        ],
        pytest.param(
            (3, 9, 35),
            (torch.float16, torch.bfloat16, torch.float32),
            "mixed_empty",
            torch.int32,
            id="mixed-dtypes-empty-component",
        ),
        pytest.param(
            (1, 257, 4097),
            (torch.uint8,),
            "overflow",
            torch.int64,
            id="modular-overflow",
        ),
        pytest.param(
            (0, 0, 0), (torch.uint8,), "normal", torch.int64, id="no-descriptors"
        ),
        pytest.param(
            (1, 1, 64), (torch.uint8,), "empty_indices", torch.int64, id="empty-indices"
        ),
        pytest.param(
            (1, 1, 1), (torch.uint8,), "zero_stride", torch.int32, id="zero-stride"
        ),
    ],
)
def test_checksum(shape, dtypes, mode, index_dtype):
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    tensors, strides, indices = generate_data(shape, dtypes, mode, index_dtype)
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, indices)
    )
    check_performance(executor, enable_perf=shape in perf_shape)


# Specialized mutation, legacy pointer-input and invalid-input checks follow.
@pytest.mark.extra_checks
def test_selection_and_corruption():
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    tensors, strides, indices = generate_data((1, 8, 32), (torch.int32,), "selection")
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, indices)
    )
    first = executor()
    tensors[0][0, 0] += 1
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, indices)
    )
    assert executor() == first
    tensors[0][5, 0] += 1
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, indices)
    )
    changed = executor()
    assert changed != first
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, [indices[0].flip(0)])
    )
    assert executor() != changed


@pytest.mark.extra_checks
def test_integer_addresses():
    from sgl_kernel_npu.operators.mem_cache import adler32_strided_checksum

    tensors, strides, indices = generate_data((2, 8, 35))
    executor = check_accuracy(
        adler32_strided_checksum, args=(tensors, strides, indices)
    )
    expected = executor()
    addresses = [tensor.data_ptr() for tensor in tensors]
    assert adler32_strided_checksum(addresses, strides, indices) == expected
    assert (
        adler32_strided_checksum(addresses, strides, indices, impl="triton") == expected
    )
    with pytest.raises(ValueError, match="requires Tensor"):
        adler32_strided_checksum(addresses, strides, indices, impl="torch_native")
    assert adler32_strided_checksum([], [], []) == 1
    assert adler32_strided_checksum([0], [64], [torch.empty(0, dtype=torch.int64)]) == 1
    assert (
        adler32_strided_checksum([0], [0], [torch.tensor([0], dtype=torch.int32)]) == 1
    )


@pytest.mark.extra_checks
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
        adler32_strided_checksum(tensors, strides, indices, impl=get_test_impls()[0])
