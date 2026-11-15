"""Accuracy of chunked grouped-head key products and sequence tails."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Qwen3-Next TP4: Hg=4, H=8, K=128. T=2048 is synthetic prefill.
perf_shape = [(1, 2048, 4, 8, 128)]


def generate_test_data(length, varlen, dtype):
    k = torch.randn(1, length, 4, 128, dtype=dtype) * 0.05
    beta = torch.rand(1, length, 8, dtype=dtype)
    gate = -torch.rand(1, length, 8).cumsum(dim=1) * 0.01
    offsets = torch.tensor([0, 31, length], dtype=torch.int32) if varlen else None
    return k, beta, gate, offsets


@pytest.mark.parametrize(
    "length", [64, 129, 2048], ids=["single-chunk", "chunk-tail", "prefill-2048"]
)
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_scaled_dot_kkt(length, varlen, dtype):
    from sgl_kernel_npu.operators.attention.gdn import chunk_scaled_dot_kkt

    k, beta, gate, offsets = generate_test_data(length, varlen, dtype)

    executor = check_accuracy(
        chunk_scaled_dot_kkt, args=(k, beta, gate), kwargs={"cu_seqlens": offsets}
    )
    check_performance(
        executor,
        enable_perf=(
            (1, length, 4, 8, 128) in perf_shape
            and dtype == torch.bfloat16
            and not varlen
        ),
    )
