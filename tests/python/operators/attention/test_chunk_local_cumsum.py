"""Check scalar chunk sums against an independent torch implementation."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("length", [64, 129])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_chunk_local_cumsum(length, reverse, varlen, dtype):
    from sgl_kernel_npu.operators.attention import chunk_local_cumsum

    g = torch.randn(1, length, 8, dtype=dtype)
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    executor = check_accuracy(
        chunk_local_cumsum,
        args=(g, 64),
        # The existing Triton store requires matching input/output element types.
        kwargs={
            "reverse": reverse,
            "scale": 0.5,
            "cu_seqlens": offsets,
            "output_dtype": None,
        },
    )
    check_performance(
        executor, enable_perf=(length == 129 and not reverse and not varlen)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["none", "explicit"])
def test_matching_output_dtype(dtype, mode):
    from sgl_kernel_npu.operators.attention import chunk_local_cumsum

    g = torch.randn(1, 64, 8, dtype=dtype)
    output_dtype = None if mode == "none" else dtype
    result = chunk_local_cumsum(g, 64, output_dtype=output_dtype, impl="triton")
    assert result.dtype == dtype


@pytest.mark.parametrize("impl", ["auto", "triton"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rejects_default_mismatched_output(dtype, impl):
    from sgl_kernel_npu.operators.attention import chunk_local_cumsum

    g = torch.randn(1, 64, 8, dtype=dtype)
    with pytest.raises(ValueError, match="output_dtype must match g.dtype"):
        chunk_local_cumsum(g, 64, impl=impl)


def test_default_output_dtype_for_float32():
    from sgl_kernel_npu.operators.attention import chunk_local_cumsum

    g = torch.randn(1, 64, 8, dtype=torch.float32)
    assert chunk_local_cumsum(g, 64, impl="triton").dtype == torch.float32


def test_rejects_explicit_mismatched_output():
    from sgl_kernel_npu.operators.attention import chunk_local_cumsum

    g = torch.randn(1, 64, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="output_dtype must match g.dtype"):
        chunk_local_cumsum(g, 64, output_dtype=torch.float16, impl="triton")
