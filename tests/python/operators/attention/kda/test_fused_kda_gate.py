import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(32, 12, 128), (128, 8, 128)]


def generate_data(shape, dtype, has_bias):
    tokens, heads, dim = shape
    torch.manual_seed(42)
    return (
        torch.randn(tokens, heads * dim, dtype=dtype),
        torch.randn(heads, dtype=torch.float32) * 0.1,
        torch.randn(heads * dim, dtype=torch.float32) * 0.1 if has_bias else None,
    )


@pytest.mark.parametrize(
    "shape,dtype,has_bias,lower_bound",
    [
        pytest.param(
            (128, 8, 128),
            torch.bfloat16,
            True,
            None,
            id="kimi-linear-48b-a3b-tp4-prefill",
        ),
        pytest.param(
            (32, 12, 128), torch.bfloat16, True, -5.0, id="kimi-k3.1-pro-tp8-verify"
        ),
    ],
)
def test_kda_gate(shape, dtype, has_bias, lower_bound):
    from sgl_kernel_npu.operators.attention.kda import fused_kda_gate

    x, a, bias = generate_data(shape, dtype, has_bias)
    executor = check_accuracy(
        fused_kda_gate,
        args=(x, a, shape[-1]),
        kwargs={"gate_bias": bias, "lower_bound": lower_bound},
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and has_bias and lower_bound is None
    )
