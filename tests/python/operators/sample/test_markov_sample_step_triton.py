import pytest
import triton
import triton.language as tl
from sgl_kernel_npu.implementation.triton.sample.markov_sample import (
    _negative_log1p_negative,
)

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(8, 129280)]


def generate_test_data(
    rows, vocab, dtype, precomputed_noise, full_bias=False, seed=None
):
    def logits(width):
        if generator is None:
            return torch.randn(rows, width, dtype=dtype)
        return torch.randn(
            rows, width, dtype=dtype, device="cpu", generator=generator
        ).to(temperatures.device)

    generator = (
        None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    )
    temperatures = torch.ones(rows, dtype=torch.float32)
    if rows > 1 and not full_bias:
        temperatures[::2] = 0
    return {
        "base_logits": logits(vocab),
        "bias": logits(vocab if full_bias else vocab - 3),
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


@triton.jit
def _philox_probe(OFFSET, RAW, PACKED, STAGES):
    q = tl.arange(0, 256)
    lane = tl.zeros((256,), tl.int64)
    w0, w1, w2, w3 = tl.philox(
        1,
        q.to(tl.uint32),
        lane.to(tl.uint32),
        lane.to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        10,
    )
    tl.store(RAW + q, w0.to(tl.int32, bitcast=True))
    tl.store(RAW + 256 + q, w1.to(tl.int32, bitcast=True))
    tl.store(RAW + 512 + q, w2.to(tl.int32, bitcast=True))
    tl.store(RAW + 768 + q, w3.to(tl.int32, bitcast=True))
    words = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    i = tl.arange(0, 1024)
    tl.store(PACKED + i, words.to(tl.int32, bitcast=True))
    unsigned = words.to(tl.int64) & 0xFFFFFFFF
    uniform = (unsigned.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    noise = _negative_log1p_negative(uniform)
    tl.store(STAGES + i, uniform)
    tl.store(STAGES + 1024 + i, noise)
    tl.store(STAGES + 2048 + i, -tl.log(noise))


def first_philox_difference(offset, width):
    from sgl_kernel_npu.implementation.torch_native.sample.markov_sample import _philox

    raw = torch.empty(1024, dtype=torch.int32, device=offset.device)
    packed = torch.empty_like(raw)
    stages = torch.empty((3, 1024), dtype=torch.float32, device=offset.device)
    _philox_probe[(1,)](offset, raw, packed, stages, num_warps=1)

    q = torch.arange(256, dtype=torch.int64, device="cpu")
    zero = torch.zeros_like(q)
    words = torch.stack(_philox(1, q, zero, zero, zero), dim=-1).flatten()
    uniform = (words.float() + 0.5) * 2**-32
    noise = -torch.log1p(-uniform)
    actual_uniform, actual_noise, actual_score = stages.cpu()
    stages_to_check = (
        ("raw Philox", raw.cpu().view(4, 256).T.reshape(-1).long() & 0xFFFFFFFF, words),
        ("interleave", packed.cpu().long() & 0xFFFFFFFF, words),
        ("uniform", actual_uniform, uniform),
        ("exponential noise", actual_noise, noise),
        ("Gumbel score", actual_score, -torch.log(noise)),
    )
    for name, actual, expected in stages_to_check:
        if name in ("exponential noise", "Gumbel score"):
            matches = torch.isclose(
                actual[:width], expected[:width], rtol=2e-5, atol=1e-6
            )
        else:
            matches = actual[:width] == expected[:width]
        differences = (~matches).nonzero()
        if differences.numel():
            i = differences[0].item()
            return f"{name}, token {i}, Triton={actual[i].item()}, CPU={expected[i].item()}"
    return (
        f"all intermediate values match the CPU reference across {width} valid tokens"
    )


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
def test_markov_sample(rows, vocab, dtype, precomputed_noise, capsys):
    from sgl_kernel_npu.operators.sample import markov_sample_step_triton

    probe_philox = (rows, vocab, dtype, precomputed_noise) == (
        1,
        513,
        torch.float32,
        False,
    )
    kwargs = generate_test_data(
        rows, vocab, dtype, precomputed_noise, seed=0 if probe_philox else None
    )
    if probe_philox:
        message = first_philox_difference(kwargs["offset"], kwargs["bias"].shape[1])
        with capsys.disabled():
            print(f"Philox probe: {message}", flush=True)
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
