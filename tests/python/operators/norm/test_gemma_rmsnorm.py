import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 2304), (16, 3840)]


def generate_data(shape, dtype):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=dtype), torch.randn(shape[-1], dtype=dtype)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 2304), id="gemma-2-2b-decode"),
        pytest.param((16, 3840), id="gemma-3-4b-prefill"),
        pytest.param((2, 3, 128), id="batched-input"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gemma_rmsnorm(shape, dtype):
    from sgl_kernel_npu.operators.norm import gemma_rmsnorm

    executor = check_accuracy(gemma_rmsnorm, args=generate_data(shape, dtype))
    check_performance(executor, enable_perf=shape in perf_shape)
