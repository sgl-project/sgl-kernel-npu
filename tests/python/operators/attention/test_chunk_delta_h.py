"""Test chunk states, updated values and final state against torch."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("length", [64, 129])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_delta_h(length, varlen, dtype):
    from sgl_kernel_npu.operators.attention import chunk_delta_h

    k = torch.randn(1, length, 2, 64, dtype=dtype) * 0.05
    w = torch.randn(1, length, 4, 64, dtype=dtype) * 0.05
    u = torch.randn(1, length, 4, 64, dtype=dtype) * 0.05
    g = -torch.rand(1, length, 4).cumsum(dim=1) * 0.01
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    state = torch.randn(2 if varlen else 1, 4, 64, 64, dtype=torch.float32) * 0.05
    executor = check_accuracy(
        chunk_delta_h,
        args=(k, w, u, g, state),
        kwargs={
            "cu_seqlens": offsets,
            "output_final_state": True,
        },
    )
    check_performance(executor, enable_perf=(length == 129 and not varlen))
