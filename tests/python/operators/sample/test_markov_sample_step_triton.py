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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "softcap,vocab_limit",
    [
        pytest.param(30.0, None, id="softcap-only"),
        pytest.param(None, 3000, id="vocab-limit-only"),
        pytest.param(30.0, 3000, id="softcap-and-vocab-limit"),
    ],
)
@pytest.mark.parametrize(
    "precomputed_noise", [True, False], ids=["external-noise", "philox"]
)
def test_markov_sample_softcap_and_vocab_limit(
    dtype, softcap, vocab_limit, precomputed_noise
):
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    device = torch.get_default_device()
    rows, vocab = 6, 4097
    kwargs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in generate_test_data(rows, vocab, dtype, precomputed_noise).items()
    }
    # Place a dominant spike past vocab_limit to verify masking excludes it,
    # and large magnitudes inside valid columns to exercise softcap saturation.
    kwargs["base_logits"] = kwargs["base_logits"] * 40.0
    if vocab_limit is not None:
        kwargs["base_logits"][:, vocab_limit:] = 1e4
    kwargs["softcap"] = softcap
    kwargs["vocab_limit"] = vocab_limit

    logits_triton = torch.empty((rows, vocab), dtype=dtype, device=device)
    logits_ref = torch.empty((rows, vocab), dtype=dtype, device=device)

    actual = markov_sample_step_triton(
        **kwargs, logits_out=logits_triton, impl="triton"
    )
    expected = markov_sample_step_triton(
        **kwargs, logits_out=logits_ref, impl="torch_native"
    )

    assert torch.equal(actual, expected)
    if vocab_limit is not None:
        assert (actual < vocab_limit).all()
        assert torch.isneginf(logits_triton[:, vocab_limit:]).all()
    if softcap is not None:
        finite = torch.isfinite(logits_triton)
        assert (logits_triton[finite].abs() <= softcap + 1e-2).all()
        atol = 1e-5 if dtype == torch.float32 else 1e-1
        rtol = 1e-5 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(
            logits_triton[finite].float(),
            logits_ref[finite].float(),
            atol=atol,
            rtol=rtol,
        )
    else:
        torch.testing.assert_close(logits_triton, logits_ref, atol=0, rtol=0)


@pytest.mark.extra_checks
def test_markov_sample_sharded_vocab_limit_and_softcap():
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    device = torch.get_default_device()
    rows, shard_width, vocab_start, vocab_limit = 4, 1024, 1024, 1500
    dtype = torch.bfloat16
    base = torch.randn(rows, shard_width, dtype=dtype, device=device) * 30.0
    base[:, vocab_limit - vocab_start :] = 1e4
    bias = torch.randn(rows, shard_width, dtype=dtype, device=device)
    temps = torch.zeros(rows, dtype=torch.float32, device=device)
    offset = torch.zeros(1, dtype=torch.int64, device=device)

    logits_triton = torch.empty((rows, shard_width), dtype=dtype, device=device)
    logits_ref = torch.empty((rows, shard_width), dtype=dtype, device=device)
    actual = markov_sample_step_triton(
        base_logits=base,
        bias=bias,
        temperatures=temps,
        key=7,
        offset=offset,
        step=2,
        step_dtype=dtype,
        vocab_start=vocab_start,
        gather=lambda t: t,
        gather_logits=lambda t: t,
        logits_out=logits_triton,
        has_sampled_rows=False,
        softcap=30.0,
        vocab_limit=vocab_limit,
        impl="triton",
    )
    expected = markov_sample_step_triton(
        base_logits=base,
        bias=bias,
        temperatures=temps,
        key=7,
        offset=offset,
        step=2,
        step_dtype=dtype,
        vocab_start=vocab_start,
        gather=lambda t: t,
        gather_logits=lambda t: t,
        logits_out=logits_ref,
        has_sampled_rows=False,
        softcap=30.0,
        vocab_limit=vocab_limit,
        impl="torch_native",
    )
    assert torch.equal(actual, expected)
    assert torch.isneginf(logits_triton[:, vocab_limit - vocab_start :]).all()


@pytest.mark.extra_checks
def test_markov_sample_reuses_compilation_across_batch_and_step():
    from sgl_kernel_npu.implementation.triton.sample.markov_sample import (
        _markov_sample_finish_kernel,
        _markov_sample_partial_kernel,
    )
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    device = torch.get_default_device()
    vocab, dtype = 2051, torch.bfloat16

    def _cache_size(kernel):
        if hasattr(kernel, "device_caches"):
            return sum(len(entry[0]) for entry in kernel.device_caches.values())
        return sum(len(entry) for entry in kernel.cache.values())

    def _run(rows, step, key):
        kwargs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in generate_test_data(rows, vocab, dtype, False).items()
        }
        kwargs["step"] = step
        kwargs["key"] = key
        actual = markov_sample_step_triton(**kwargs, impl="triton")
        expected = markov_sample_step_triton(**kwargs, impl="torch_native")
        assert torch.equal(actual, expected)

    _run(1, 0, 1)
    partial_first = _cache_size(_markov_sample_partial_kernel)
    finish_first = _cache_size(_markov_sample_finish_kernel)

    for rows, step, key in ((2, 1, 2), (3, 2, 16), (8, 3, 17), (16, 4, 1024)):
        _run(rows, step, key)

    assert _cache_size(_markov_sample_partial_kernel) == partial_first
    assert _cache_size(_markov_sample_finish_kernel) == finish_first
