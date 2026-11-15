"""Public recipes hc_post smoke tests for both supported input layouts."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


def generate_test_data(shape, dtype, weight_dtype):
    torch.npu.manual_seed(42)
    leading = shape[:-1]
    return (
        torch.randn(shape, dtype=dtype),
        torch.randn((*leading, 4, shape[-1]), dtype=dtype),
        torch.randn((*leading, 4), dtype=weight_dtype),
        torch.randn((*leading, 4, 4), dtype=weight_dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [
        # Minimum nonempty token count with a supported hidden size.
        pytest.param((1, 4096), id="edge-single-token-1x4096"),
        # Keep 3D coverage until the target V4.1 model dimensions are confirmed.
        pytest.param((2, 2, 7168), id="smoke-3d-2x2x7168"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32])
def test_hc_post(shape, dtype, weight_dtype):
    from sgl_kernel_npu.operators.hyper_conn import hc_post

    inputs = generate_test_data(shape, dtype, weight_dtype)
    executor = check_accuracy(hc_post, args=inputs)
    check_performance(executor, enable_perf=False)
