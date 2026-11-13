import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 5760), (128, 5760)]


def generate_data(shape, dtype):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=dtype) * 4


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 5760), id="gpt-oss-120b-decode"),
        pytest.param((128, 5760), id="gpt-oss-120b-prefill"),
    ],
)
def test_swiglu_oai(shape):
    dtype = torch.bfloat16
    from sgl_kernel_npu.operators.activation import swiglu_oai

    x = generate_data(shape, dtype)
    executor = check_accuracy(swiglu_oai, args=(x, shape[-1], 1.702, 7.0))
    check_performance(executor, enable_perf=shape in perf_shape)
