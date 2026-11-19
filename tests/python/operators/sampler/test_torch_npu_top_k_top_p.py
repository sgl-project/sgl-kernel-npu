"""Cover top-k ties, top-p boundaries and vocabulary order restoration."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(vocab_size, k, p, tie):
    logits = -0.2 * torch.arange(vocab_size, dtype=torch.float32)
    if tie:
        logits[k] = logits[k - 1]
    # Break sorted order to verify that the retained logits land at the original indices.
    logits = logits.roll(2).unsqueeze(0)
    return (
        logits,
        torch.tensor([p], dtype=logits.dtype),
        torch.tensor([k], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "vocab_size, k, p, tie",
    [
        pytest.param(8192, 32, 0.85, False, id="autoregressive-sampling-shape"),
        pytest.param(17, 3, 1.0, True, id="topk-retains-cutoff-ties"),
        pytest.param(17, 3, 0.0, False, id="topp-retains-one-at-zero"),
    ],
)
def test_top_k_top_p(vocab_size, k, p, tie):
    from sgl_kernel_npu.operators.sampler import top_k_top_p

    executor = check_accuracy(
        top_k_top_p, args=generate_test_data(vocab_size, k, p, tie)
    )
    check_performance(executor, enable_perf=vocab_size == 8192)
