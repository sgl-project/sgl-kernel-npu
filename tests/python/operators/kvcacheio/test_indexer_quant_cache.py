"""Indexer FP8 cache-write accuracy, including slot zero and the -1 sentinel."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(8, 128)]


@pytest.mark.parametrize("tokens", [1, 8], ids=["synthetic-decode", "synthetic-batch"])
def test_indexer_cache(tokens):
    from sgl_kernel_npu.operators.kvcacheio import indexer_quant_cache

    torch.manual_seed(16)
    x = torch.randn(tokens, 128, dtype=torch.bfloat16)
    slots = torch.arange(tokens, dtype=torch.int32)
    if tokens > 1:
        slots[-1] = -1
    keys = torch.zeros(1, 128, 1, 128, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(1, 128, 1, 1, dtype=torch.float32)
    executor = check_accuracy(
        indexer_quant_cache,
        args=(keys, scales, x, slots),
        args_ref=(keys.clone(), scales.clone(), x, slots),
        kwargs=dict(quant_mode="fp8"),
        standard={
            "torch.float8_e4m3fn": AccuracyStandard(atol=16, rtol=0.06),
            "torch.float32": AccuracyStandard(atol=1e-5, rtol=0.03),
        },
    )
    check_performance(executor, enable_perf=(tokens, 128) in perf_shape)
