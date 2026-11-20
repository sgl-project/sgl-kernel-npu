from unittest.mock import Mock

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


def test_markov_greedy_keyword_compatibility(monkeypatch):
    from sgl_kernel_npu.operators.sample import markov_greedy

    backend = Mock()
    select_backend = Mock(return_value=backend)
    monkeypatch.setattr(markov_greedy, "get_impl_by_name", select_backend)
    kwargs = generate_test_data(1, 8, 4, torch.float32)

    result = markov_greedy.markov_greedy_step_triton(**kwargs, future_option=None)
    assert result is backend.return_value
    select_backend.assert_called_once_with("sample.markov_greedy_step_triton", "auto")
    backend.assert_called_once_with(**kwargs)

    select_backend.reset_mock()
    with pytest.raises(TypeError, match="future_option"):
        markov_greedy.markov_greedy_step_triton(**kwargs, future_option=False)
    select_backend.assert_not_called()


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
