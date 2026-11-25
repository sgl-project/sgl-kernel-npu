"""Compare autoregressive TopK/TopP filtering through the public API."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(batch_size, vocab_size, k, p, temperature):
    logits = torch.linspace(2, -8, vocab_size, dtype=torch.float32)
    # Use distinct scores around both thresholds and move the tail's high scores
    # to different vocabulary offsets to exercise the scatter back to token IDs.
    logits[: k + 1] = 6 - 0.15 * torch.arange(k + 1, dtype=logits.dtype)
    scaled_logits = torch.stack(
        [logits.roll(137 * row + 2) for row in range(batch_size)]
    ).div_(temperature)
    return (
        scaled_logits,
        torch.full((batch_size,), p, dtype=logits.dtype),
        torch.full((batch_size,), k, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "batch_size, vocab_size, k, p, temperature",
    [pytest.param(4, 151936, 20, 0.95, 0.6, id="qwen3-8b-thinking-decode")],
)
def test_top_k_top_p(batch_size, vocab_size, k, p, temperature):
    from sgl_kernel_npu.operators.sampler import top_k_top_p

    executor = check_accuracy(
        top_k_top_p, args=generate_test_data(batch_size, vocab_size, k, p, temperature)
    )
    check_performance(executor, enable_perf=True)
