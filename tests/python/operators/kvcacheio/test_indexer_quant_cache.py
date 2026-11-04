"""Indexer FP8 cache-write accuracy, including slot zero and the -1 sentinel."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
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
    # CANN 9.1 may not support generic FP8 initialization/copy/conversion.
    keys = torch.zeros(1, 128, 1, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = torch.zeros(1, 128, 1, 1, dtype=torch.float32)
    # Match the operator default: round FP8 scales up to a power of two.
    kwargs = dict(quant_mode="fp8", round_scale=True)
    executor = make_executor(
        indexer_quant_cache, args=(keys, scales, x, slots), kwargs=kwargs
    )
    if get_skip_acc_check() is True:
        require_test_impl(indexer_quant_cache, executor.impl)
    else:
        require_distinct_impls(indexer_quant_cache, executor.impl, executor.impl_ref)
        # Run independent reference math on CPU, where FP8 casts are available.
        # Copy cache bytes rather than requesting an NPU FP8 copy/conversion.
        with torch.device("cpu"):
            expected = indexer_quant_cache(
                keys.view(torch.uint8).cpu().clone().view(torch.float8_e4m3fn),
                scales.cpu().clone(),
                x.cpu(),
                slots.cpu(),
                impl=executor.impl_ref,
                **kwargs,
            )
        actual = executor()
        assert actual[0].dtype == expected[0].dtype == torch.float8_e4m3fn
        # Widen on CPU; scale accuracy retains its own tight tolerance.
        assert_accuracy(
            actual[0].view(torch.uint8).cpu().view(torch.float8_e4m3fn).float(),
            expected[0].float(),
            {"torch.float32": AccuracyStandard(atol=16, rtol=0.06)},
        )
        assert_accuracy(
            actual[1].cpu(),
            expected[1],
            {"torch.float32": AccuracyStandard(atol=1e-5, rtol=0.03)},
        )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=(tokens, 128) in perf_shape)
