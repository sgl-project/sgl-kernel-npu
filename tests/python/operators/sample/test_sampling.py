"""Exercise sampling registration and backend parity on Ascend."""

import pytest
import torch

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("rows,vocab", [(0, 17), (1, 8193), (8, 4097)])
def test_row_argmax(rows, vocab):
    from sgl_kernel_npu.operators.sample import row_argmax

    x = torch.randn(rows, vocab, dtype=torch.float32)
    if rows:
        x[:, 0] = 100
        x[:, -1] = 100
    executor = check_accuracy(row_argmax, args=(x,))
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("rows,vocab,hidden", [(1, 513, 33), (8, 4097, 32)])
def test_markov_greedy(rows, vocab, hidden):
    from sgl_kernel_npu.operators.sample import markov_greedy

    kwargs = {
        "base_logits": torch.randn(rows, vocab),
        "prev_embeds": torch.randn(rows, hidden),
        "w2_weight": torch.randn(vocab, hidden),
    }
    executor = check_accuracy(markov_greedy, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("gamma", [0, 3, 17])
def test_build_out_tokens(gamma):
    from sgl_kernel_npu.operators.sample import build_out_tokens

    rows = gamma + 1
    kwargs = {
        "draft_tokens": torch.arange(rows * gamma).reshape(rows, gamma),
        "correct_len": torch.arange(rows),
        "bonus": torch.arange(rows) + 50,
        "verify_num_draft_tokens": gamma + 1,
        "gamma": gamma,
    }
    executor = check_accuracy(build_out_tokens, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,vocab", [(1, 513), (8, 4097)])
def test_markov_sample(rows, vocab, dtype):
    from sgl_kernel_npu.operators.sample import markov_sample

    from tests.python.sgl_kernel_npu.test_markov_sample import (
        reference_markov_sample_step,
    )

    kwargs = {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "bias": torch.randn(rows, vocab - 3, dtype=dtype),
        "temperatures": torch.ones(rows, dtype=torch.float32),
        "key": 1,
        "offset": torch.zeros(1, dtype=torch.int64),
        "step": 0,
        "step_dtype": dtype,
        "exp_noise": torch.rand(rows, vocab, dtype=torch.float32) + 0.1,
    }
    kwargs["temperatures"][::2] = 0
    expected = reference_markov_sample_step(**kwargs)
    actual = markov_sample(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Sampling has no registered torch_native backend; its CPU reference is
    # checked above and cannot be used by the cross-backend executor.
    executor = check_accuracy(markov_sample, kwargs=kwargs, skip_acc_check=True)
    check_performance(executor, enable_perf=False)
