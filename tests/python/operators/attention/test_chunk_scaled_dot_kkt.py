"""Accuracy of chunked grouped-head key products and sequence tails."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("length", [64, 129])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_scaled_dot_kkt(length, varlen, dtype):
    from sgl_kernel_npu.operators.attention import chunk_scaled_dot_kkt

    k = torch.randn(1, length, 2, 128, dtype=dtype) * 0.05
    beta = torch.rand(1, length, 4, dtype=dtype)
    gate = -torch.rand(1, length, 4).cumsum(dim=1) * 0.01
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    executor = check_accuracy(
        chunk_scaled_dot_kkt, args=(k, beta, gate), kwargs={"cu_seqlens": offsets}
    )
    check_performance(executor, enable_perf=(length == 129 and not varlen))
