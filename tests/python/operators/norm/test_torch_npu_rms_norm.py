"""Check both RMSNorm APIs, including statistics and the residual output."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")

PERF_SHAPES = [(16, 7168)]


def generate_test_data(shape, gamma_shape, dtype, zero):
    torch.npu.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    gamma = torch.rand(gamma_shape, dtype=dtype) + 0.5
    return x.zero_() if zero else x, gamma


@pytest.mark.parametrize(
    "shape, gamma_shape, zero",
    [
        pytest.param((16, 7168), (7168,), False, id="deepseek-v3-hidden-width-sample"),
        pytest.param((3, 65), (65,), True, id="zero-epsilon-and-tail"),
        pytest.param((2, 3, 64), (3, 64), False, id="two-normalization-axes"),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5])
def test_rms_norm(shape, gamma_shape, zero, epsilon):
    from sgl_kernel_npu.operators.norm import rms_norm

    x, gamma = generate_test_data(shape, gamma_shape, torch.float32, zero)
    executor = check_accuracy(rms_norm, args=(x, gamma, epsilon))
    check_performance(executor, enable_perf=shape in PERF_SHAPES and epsilon == 1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 7168), id="hidden-width-sample"),
        pytest.param((1, 65), id="tail"),
    ],
)
def test_add_rms_norm(shape):
    from sgl_kernel_npu.operators.norm import add_rms_norm

    x, gamma = generate_test_data(shape, (shape[-1],), torch.float32, False)
    residual = torch.randn_like(x)
    before_x, before_residual = x.clone(), residual.clone()
    executor = check_accuracy(add_rms_norm, args=(x, residual, gamma))
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, before_residual, rtol=0, atol=0)
    check_performance(executor, enable_perf=shape in PERF_SHAPES)
