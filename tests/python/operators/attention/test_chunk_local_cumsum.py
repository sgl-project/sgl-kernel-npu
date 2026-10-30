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
        kwargs={"reverse": reverse, "scale": 0.5, "cu_seqlens": offsets},
    )
    check_performance(
        executor, enable_perf=(length == 129 and not reverse and not varlen)
    )
