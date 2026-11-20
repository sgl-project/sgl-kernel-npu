"""Compare the public TorchNPU activation operators against PyTorch formulas."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(shape, dtype):
    torch.npu.manual_seed(42)
    return torch.randn(shape, dtype=dtype) * 0.5


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 3072), id="clip-mlp-width-sample"),
    ],
)
def test_fast_gelu(shape):
    from sgl_kernel_npu.operators.activation import fast_gelu

    executor = check_accuracy(
        fast_gelu, args=(generate_test_data(shape, torch.float16),)
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "shape, dim, approximate, activate_left",
    [
        # SGLang GeluAndMul fast path activates the left half.
        pytest.param((16, 6144), -1, 1, True, id="gelu-and-mul"),
    ],
)
def test_geglu(shape, dim, approximate, activate_left):
    from sgl_kernel_npu.operators.activation import geglu

    x = generate_test_data(shape, torch.float16)
    executor = check_accuracy(geglu, args=(x, dim, approximate, activate_left))
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "shape, dim",
    [
        # SGLang SiLUAndMul splits packed gate and up projections along dim=-1.
        pytest.param((16, 4096), -1, id="packed-gate-up"),
    ],
)
def test_swiglu(shape, dim):
    from sgl_kernel_npu.operators.activation import swiglu

    x = generate_test_data(shape, torch.float16)
    executor = check_accuracy(swiglu, args=(x, dim))
    check_performance(executor, enable_perf=True)
