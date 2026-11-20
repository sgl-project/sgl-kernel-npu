import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(8, 129280)]


def generate_test_data(rows, vocab, dtype, precomputed_noise, full_bias=False):
    temperatures = torch.ones(rows, dtype=torch.float32)
    if rows > 1 and not full_bias:
        temperatures[::2] = 0
    return {
        "base_logits": torch.randn(rows, vocab, dtype=dtype),
        "bias": torch.randn(rows, vocab if full_bias else vocab - 3, dtype=dtype),
        "temperatures": temperatures,
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

    kwargs = generate_test_data(rows, vocab, dtype, precomputed_noise)
    executor = check_accuracy(markov_sample_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=False)


@pytest.mark.parametrize(
    "rows,vocab,dtype",
    [pytest.param(8, 129280, torch.bfloat16, id="deepseek-v4-flash-b8")],
)
def test_markov_sample_deepseek_v4_flash(rows, vocab, dtype):
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    # Full model vocabulary, BF16 logits and batch 8 as an example workload.
    kwargs = generate_test_data(rows, vocab, dtype, False, full_bias=True)
    executor = check_accuracy(markov_sample_step_triton, kwargs=kwargs)
    check_performance(executor, enable_perf=(rows, vocab) in perf_shape)
