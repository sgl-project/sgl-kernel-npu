import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(8, 129280)]


def generate_test_data(rows, vocab):
    x = torch.randn(rows, vocab, dtype=torch.float32)
    if rows:
        x[:, 0] = 100
        x[:, -1] = 100
    return x


@pytest.mark.parametrize(
    "rows,vocab",
    [
        pytest.param(0, 17, id="empty-batch"),
        pytest.param(1, 8193, id="single-row-cross-tile"),
        pytest.param(8, 4097, id="batched-odd-vocab"),
        # DeepSeek-V4-Flash DSpark vocabulary; batch is an example workload.
        pytest.param(8, 129280, id="deepseek-v4-flash-vocab-b8"),
    ],
)
def test_row_argmax(rows, vocab):
    from sgl_kernel_npu.operators.sample import row_argmax

    x = generate_test_data(rows, vocab)
    executor = check_accuracy(row_argmax, args=(x,))
    check_performance(executor, enable_perf=(rows, vocab) in perf_shape)
