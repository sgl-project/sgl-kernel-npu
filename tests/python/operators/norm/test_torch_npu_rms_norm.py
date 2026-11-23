"""Check both RMSNorm APIs, including statistics and the residual output."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(shape, dtype):
    torch.npu.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    gamma = torch.rand(shape[-1], dtype=dtype) + 0.5
    return x, gamma


@pytest.mark.parametrize(
    "shape, epsilon",
    [pytest.param((16, 4096), 1e-6, id="qwen3-8b-first-layer-attn-input-tp1")],
)
def test_rms_norm(shape, epsilon):
    from sgl_kernel_npu.operators.norm import rms_norm

    x, gamma = generate_test_data(shape, torch.bfloat16)
    executor = check_accuracy(rms_norm, args=(x, gamma, epsilon))
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "shape, epsilon",
    [
        pytest.param((16, 4096), 1e-6, id="qwen3-8b-post-attention-residual-tp1"),
    ],
)
def test_add_rms_norm(shape, epsilon):
    from sgl_kernel_npu.operators.norm import add_rms_norm

    x, gamma = generate_test_data(shape, torch.bfloat16)
    residual = torch.randn_like(x)
    before_x, before_residual = x.clone(), residual.clone()
    executor = check_accuracy(add_rms_norm, args=(residual, x, gamma, epsilon))
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, before_residual, rtol=0, atol=0)
    check_performance(executor, enable_perf=True)
