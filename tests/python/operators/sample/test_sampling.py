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
@pytest.mark.parametrize("precomputed_noise", [True, False])
def test_markov_sample(rows, vocab, dtype, precomputed_noise):
    from sgl_kernel_npu.operators.sample import markov_sample

    kwargs = {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "bias": torch.randn(rows, vocab - 3, dtype=dtype),
        "temperatures": torch.ones(rows, dtype=torch.float32),
        "key": 1,
        "offset": torch.zeros(1, dtype=torch.int64),
        "step": 0,
        "step_dtype": dtype,
        "exp_noise": (
            torch.rand(rows, vocab, dtype=torch.float32) + 0.1
            if precomputed_noise
            else None
        ),
    }
    kwargs["temperatures"][::2] = 0
    executor = check_accuracy(markov_sample, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


def test_markov_sample_kept_logits():
    from sgl_kernel_npu.operators.sample import markov_sample

    rows, vocab = 3, 513
    kwargs = {
        "base_logits": torch.randn(rows, vocab, dtype=torch.float16),
        "bias": torch.randn(rows, vocab - 3, dtype=torch.float16),
        "temperatures": torch.tensor([0.0, 0.7, 1.0], dtype=torch.float32),
        "key": 12345,
        "offset": torch.tensor([7], dtype=torch.int64),
        "step": 2,
        "step_dtype": torch.float16,
    }
    outputs = []
    for impl in ("torch_native", "triton"):
        logits = torch.empty_like(kwargs["base_logits"])
        tokens = markov_sample(**kwargs, logits_out=logits, impl=impl)
        outputs.append((tokens, logits))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
