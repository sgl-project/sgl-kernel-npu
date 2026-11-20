"""Verify dtype conversion, dense format conversion and the prefetch contract."""

import pytest

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

torch = pytest.importorskip("torch")
_EXACT_FP16 = {"torch.float16": AccuracyStandard(atol=0, rtol=0)}


def generate_data(shape):
    torch.npu.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32)


@pytest.mark.parametrize("shape", [pytest.param((1, 4096), id="linear-decode-dtype")])
def test_dtype_cast(shape):
    from sgl_kernel_npu.operators.data_move import dtype_cast

    executor = check_accuracy(
        dtype_cast,
        (generate_data(shape), torch.float16),
        standard=_EXACT_FP16,
    )
    check_performance(executor)


@pytest.mark.parametrize("shape", [pytest.param((16, 128), id="linear-weight-nd")])
def test_format_cast_to_nd(shape):
    import torch_npu
    from sgl_kernel_npu.operators.data_move import format_cast

    executor = check_accuracy(
        format_cast, (generate_data(shape).half(), 2), standard=_EXACT_FP16
    )
    if executor.accuracy_checked:
        assert torch_npu.get_npu_format(executor()) == 2
    check_performance(executor)


@pytest.mark.parametrize(
    "shape", [pytest.param((16, 128), id="arch22-weight-prefetch")]
)
def test_prefetch_arch22(shape):
    from sgl_kernel_npu.operators.data_move import prefetch

    weights = generate_data(shape).half()
    executor = make_executor(
        prefetch, (weights, weights, weights.numel() * weights.element_size())
    )
    require_test_impl(prefetch, executor.impl)
    if get_skip_acc_check() is not True:
        before = weights.clone()
        assert executor() is None
        torch.npu.synchronize()
        assert_accuracy(weights, before, _EXACT_FP16)
        executor.accuracy_checked = True
    check_performance(executor)
