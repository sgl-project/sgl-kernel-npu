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


@pytest.mark.extra_checks
def test_row_argmax_reuses_compilation_across_batch_sizes():
    from sgl_kernel_npu.implementation.triton.sample.row_argmax import (
        _row_argmax_kernel,
    )
    from sgl_kernel_npu.operators.sample import row_argmax

    device = torch.get_default_device()
    vocab = 4099

    def _compiled_count(kernel):
        if hasattr(kernel, "device_caches"):
            return sum(len(entry[0]) for entry in kernel.device_caches.values())
        return sum(len(entry) for entry in kernel.cache.values())

    # Prime compilation for this vocab width once.
    x0 = generate_test_data(1, vocab).to(device)
    assert torch.equal(row_argmax(x0, impl="triton"), torch.argmax(x0, dim=-1))
    compiled_after_first = _compiled_count(_row_argmax_kernel)

    for rows in (2, 3, 7, 16, 32):
        x = generate_test_data(rows, vocab).to(device)
        actual = row_argmax(x, impl="triton")
        expected = torch.argmax(x, dim=-1)
        assert torch.equal(actual, expected)

    compiled_after_all = _compiled_count(_row_argmax_kernel)
    assert compiled_after_all == compiled_after_first
