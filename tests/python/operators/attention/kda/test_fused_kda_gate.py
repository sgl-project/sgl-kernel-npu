import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(32, 12, 128), (128, 12, 128)]


def generate_data(shape, dtype, has_bias):
    tokens, heads, dim = shape
    torch.manual_seed(42)
    return (
        torch.randn(tokens, heads * dim, dtype=dtype),
        torch.randn(heads, dtype=torch.float32) * 0.1,
        torch.randn(heads * dim, dtype=torch.float32) * 0.1 if has_bias else None,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 12, 128), id="kimi-kda-verify"),
        pytest.param((128, 12, 128), id="kimi-kda-prefill"),
        pytest.param((1, 2, 16), id="minimum"),
        pytest.param((33, 2, 64), id="tile-tail"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
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
