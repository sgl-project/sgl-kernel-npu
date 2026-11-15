"""One fixed AddExample scenario, using the shared accuracy/performance flow."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# This integration example has no default performance benchmark.
perf_shape = []
ADD_EXAMPLE_STANDARD = {
    str(dtype): AccuracyStandard(atol=0, rtol=0)
    for dtype in (torch.float32, torch.int32)
}


def generate_data(shape, dtype):
    x1 = torch.tensor([2, 5], dtype=dtype, device="npu").reshape(shape)
    x2 = torch.tensor([3, 3], dtype=dtype, device=x1.device).reshape(shape)
    return x1, x2


@pytest.mark.parametrize("shape", [pytest.param((2,), id="same-shape-add")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_add_example(shape, dtype):
    from sgl_kernel_npu.operators.examples import add_example

    x1, x2 = generate_data(shape, dtype)
    executor = check_accuracy(add_example, args=(x1, x2), standard=ADD_EXAMPLE_STANDARD)
    check_performance(executor, enable_perf=shape in perf_shape)
