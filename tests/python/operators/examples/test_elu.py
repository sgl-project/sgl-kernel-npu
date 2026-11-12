"""ELU: one production shape and one unaligned tail case."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Only the production shape is measured; the edge case checks correctness.
perf_shape = [(1, 4096)]


def generate_data(shape, dtype):
    # Mean 0, unit variance values exercise both ELU branches.
    x = torch.randn(shape, dtype=dtype, device="npu")
    return (x,)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        # Borrowed LLM activation width (1 x 4096), not a traced SGLang call;
        # the aligned shape exercises the full-tile fast path.
        pytest.param((1, 4096), torch.float32, id="llm-activation-width-4096"),
        # 7 x 1025 leaves a partial tile in the last iteration.
        pytest.param((7, 1025), torch.float16, id="unaligned-tail-1025"),
    ],
)
def test_elu(shape, dtype):
    from sgl_kernel_npu.operators.examples import elu

    (x,) = generate_data(shape, dtype)
    executor = check_accuracy(elu, args=(x,), kwargs={"alpha": 1.0})
    check_performance(executor, enable_perf=shape in perf_shape)
