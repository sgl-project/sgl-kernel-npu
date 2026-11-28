import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 129280, 256)]


def generate_test_data(rows, vocab, hidden, dtype):
    return {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "prev_embeds": torch.randn(rows, hidden, dtype=dtype),
        "w2_weight": torch.randn(vocab, hidden, dtype=dtype),
    }


@pytest.mark.parametrize(
    "rows,vocab,hidden,dtype",
    [
        pytest.param(1, 513, 33, torch.float32, id="single-odd-tile-and-rank"),
        pytest.param(8, 4097, 32, torch.float32, id="batched-cross-tile"),
        # Model vocabulary and Markov rank; batch is an example workload.
        pytest.param(1, 129280, 256, torch.bfloat16, id="deepseek-v4-flash-b1"),
    ],
)
def test_markov_greedy(rows, vocab, hidden, dtype):
    from sgl_kernel_npu.operators.sample import markov_greedy_step_triton

    kwargs = generate_test_data(rows, vocab, hidden, dtype)
    executor = check_accuracy(markov_greedy_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, vocab, hidden) in perf_shape)


@pytest.mark.extra_checks
def test_markov_greedy_reuses_compilation_across_batch_sizes():
    from sgl_kernel_npu.implementation.triton.sample.markov_greedy import (
        _markov_greedy_add_argmax_kernel,
        _markov_greedy_combine_kernel,
        _markov_greedy_row_kernel,
    )
    from sgl_kernel_npu.operators.sample import markov_greedy_step_triton

    device = torch.get_default_device()
    vocab, hidden, dtype = 4099, 32, torch.bfloat16
    w2 = torch.randn(vocab, hidden, dtype=dtype, device=device)

    def _cache_size(kernel):
        if hasattr(kernel, "device_caches"):
            return sum(len(entry[0]) for entry in kernel.device_caches.values())
        return sum(len(entry) for entry in kernel.cache.values())

    def _run(rows):
        base = torch.randn(rows, vocab, dtype=dtype, device=device)
        prev = torch.randn(rows, hidden, dtype=dtype, device=device)
        actual = markov_greedy_step_triton(
            base_logits=base, prev_embeds=prev, w2_weight=w2, impl="triton"
        )
        expected = markov_greedy_step_triton(
            base_logits=base, prev_embeds=prev, w2_weight=w2, impl="torch_native"
        )
        assert torch.equal(actual, expected)

    # Small-batch split-vocab path (bs < 16)
    _run(1)
    small_add = _cache_size(_markov_greedy_add_argmax_kernel)
    small_comb = _cache_size(_markov_greedy_combine_kernel)
    for rows in (2, 3, 7, 15):
        _run(rows)
    assert _cache_size(_markov_greedy_add_argmax_kernel) == small_add
    assert _cache_size(_markov_greedy_combine_kernel) == small_comb

    # Large-batch row-scan path (bs >= 32)
    _run(32)
    large_row = _cache_size(_markov_greedy_row_kernel)
    for rows in (33, 40, 48):
        _run(rows)
    assert _cache_size(_markov_greedy_row_kernel) == large_row
