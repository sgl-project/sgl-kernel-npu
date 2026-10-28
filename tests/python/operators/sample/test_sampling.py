"""Exercise sampling registration and backend parity on Ascend."""

import pytest
import torch

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance


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

    x = torch.randn(rows, vocab, dtype=torch.float32)
    if rows:
        x[:, 0] = 100
        x[:, -1] = 100
    executor = check_accuracy(row_argmax, args=(x,))
    check_performance(executor, enable_perf=(rows, vocab) == (8, 129280))


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

    kwargs = {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "prev_embeds": torch.randn(rows, hidden, dtype=dtype),
        "w2_weight": torch.randn(vocab, hidden, dtype=dtype),
    }
    executor = check_accuracy(markov_greedy_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, vocab, hidden) == (1, 129280, 256))


@pytest.mark.parametrize(
    "rows,gamma",
    [
        pytest.param(1, 0, id="edge-zero-draft"),
        pytest.param(4, 3, id="all-accept-lengths"),
        pytest.param(18, 17, id="odd-width-all-accept-lengths"),
        # DSpark draft block size 6, verify length 7; batch is an example.
        pytest.param(8, 6, id="dspark-draft6-verify7-b8"),
    ],
)
def test_build_out_tokens(rows, gamma):
    from sgl_kernel_npu.operators.sample import build_out_tokens_triton

    kwargs = {
        "draft_tokens": torch.arange(rows * gamma).reshape(rows, gamma),
        "correct_len": torch.arange(rows) % (gamma + 1),
        "bonus": torch.arange(rows) + 50,
        "verify_num_draft_tokens": gamma + 1,
        "gamma": gamma,
    }
    executor = check_accuracy(build_out_tokens_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, gamma) == (8, 6))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,vocab",
    [
        pytest.param(1, 513, id="single-partial-tile"),
        pytest.param(8, 4097, id="batched-cross-tile"),
    ],
)
@pytest.mark.parametrize(
    "precomputed_noise", [True, False], ids=["external-noise", "philox"]
)
def test_markov_sample(rows, vocab, dtype, precomputed_noise):
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

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
    if rows > 1:
        kwargs["temperatures"][::2] = 0
    executor = check_accuracy(markov_sample_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "rows,vocab,dtype",
    [pytest.param(8, 129280, torch.bfloat16, id="deepseek-v4-flash-b8")],
)
def test_markov_sample_deepseek_v4_flash(rows, vocab, dtype):
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    # Full model vocabulary, BF16 logits and batch 8 as an example workload.
    kwargs = {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "bias": torch.randn(rows, vocab, dtype=dtype),
        "temperatures": torch.ones(rows, dtype=torch.float32),
        "key": 12345,
        "offset": torch.zeros(1, dtype=torch.int64),
        "step": 0,
        "step_dtype": dtype,
    }
    executor = check_accuracy(markov_sample_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=True)


def test_markov_sample_kept_logits():
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

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
        tokens = markov_sample_step_triton(**kwargs, logits_out=logits, impl=impl)
        outputs.append((tokens, logits))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
