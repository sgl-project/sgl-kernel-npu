import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 5760), (128, 5760), (128, 1440)]


def generate_data(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.bfloat16) * 4


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 5760), id="gpt-oss-moe-tp1-single-row"),
        pytest.param((128, 5760), id="gpt-oss-moe-tp1-128-rows"),
        pytest.param((128, 1440), id="gpt-oss-moe-tp4-128-rows"),
    ],
)
def test_swiglu_oai(shape):
    from sgl_kernel_npu.operators.activation import swiglu_oai

    x = generate_data(shape)
    executor = check_accuracy(swiglu_oai, args=(x, shape[-1], 1.702, 7.0))
    check_performance(executor, enable_perf=shape in perf_shape)
