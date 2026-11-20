"""Compare autoregressive TopK/TopP filtering through the public API."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(vocab_size, k, p):
    logits = -0.2 * torch.arange(vocab_size, dtype=torch.float32)
    # Break sorted order to verify that the retained logits land at the original indices.
    logits = logits.roll(2).unsqueeze(0)
    return (
        logits,
        torch.tensor([p], dtype=logits.dtype),
        torch.tensor([k], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "vocab_size, k, p",
    [pytest.param(8192, 32, 0.85, id="autoregressive-sampling")],
)
def test_top_k_top_p(vocab_size, k, p):
    from sgl_kernel_npu.operators.sampler import top_k_top_p

    executor = check_accuracy(top_k_top_p, args=generate_test_data(vocab_size, k, p))
    check_performance(executor, enable_perf=True)
