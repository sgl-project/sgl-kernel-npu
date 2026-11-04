"""One fixed AddExample scenario, using the shared accuracy/performance flow."""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")

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
    """Construct the confirmed [2, 5] + [3, 3] inputs on the current NPU."""
    x1 = torch.tensor([2, 5], dtype=dtype, device="npu").reshape(shape)
    x2 = torch.tensor([3, 3], dtype=dtype, device=x1.device).reshape(shape)
    return x1, x2


@pytest.mark.parametrize("shape", [pytest.param((2,), id="same-shape-add")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_add_example(shape, dtype):
    from sgl_kernel_npu.operators.examples import add_example

    x1, x2 = generate_data(shape, dtype)
    before1, before2 = x1.clone(), x2.clone()
    executor = check_accuracy(add_example, args=(x1, x2), standard=ADD_EXAMPLE_STANDARD)
    result = executor()
    assert result.is_contiguous() and torch_npu.get_npu_format(result) == 2
    if executor.accuracy_checked:
        expected = torch.tensor([5, 8], dtype=dtype, device=x1.device)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert result.data_ptr() not in (x1.data_ptr(), x2.data_ptr())
    assert torch.equal(x1, before1)
    assert torch.equal(x2, before2)
    check_performance(executor, enable_perf=shape in perf_shape)
