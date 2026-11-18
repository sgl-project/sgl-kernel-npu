import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance

# Gumbel-max draft sampling over the DeepSeek-V4 vocabulary; element count
# above the SGLang size gate (numel >= 1 << 23).
perf_shape = [(128, 129280)]

_TINY = torch.finfo(torch.float32).tiny


def generate_data(rows, vocab, mode):
    torch.manual_seed(42)
    x = torch.randn(rows, vocab, dtype=torch.float32)
    # Divisor between 0.5 and 1.5 keeps quotients near one.
    divisor = torch.rand(rows, vocab, dtype=torch.float32) + 0.5
    if rows and vocab:
        if mode == "nan_quotient":
            # A NaN divisor yields a NaN quotient; a NaN logit does too.
            divisor[:, 0] = float("nan")
            x[:, 1] = float("nan")
        elif mode == "tiny_divisor":
            # Subnormal divisors without the clamp would overflow the quotient.
            divisor.mul_(1e-40)
        elif mode != "normal":
            raise ValueError(f"Unknown mode: {mode}")
    return x, divisor


@pytest.mark.parametrize(
    "rows,vocab",
    [
        pytest.param(0, 17, id="empty-batch"),
        pytest.param(1, 8193, id="single-row-cross-tile"),
        pytest.param(8, 4097, id="batched-odd-vocab"),
        # Rows above the 256 threshold switch the block size to 8192.
        pytest.param(300, 4096, id="wide-batch-block-8192"),
        pytest.param(128, 129280, id="deepseek-v4-gumbel-b128"),
    ],
)
@pytest.mark.parametrize(
    "mode,divisor_min,ignore_nan",
    [
        pytest.param("normal", None, False, id="plain"),
        # The production Gumbel call: noise clamped to tiny, NaN ignored.
        pytest.param("normal", _TINY, True, id="gumbel-production"),
        # torch.argmax picks the NaN index but the kernel never does, so NaN
        # data is only compared with ignore_nan=True.
        pytest.param("nan_quotient", None, True, id="nan-ignored"),
        pytest.param("tiny_divisor", _TINY, False, id="clamped"),
    ],
)
def test_div_argmax(rows, vocab, mode, divisor_min, ignore_nan):
    from sgl_kernel_npu.operators.sample import div_argmax

    x, divisor = generate_data(rows, vocab, mode)
    executor = check_accuracy(
        div_argmax,
        args=(x, divisor),
        kwargs={"divisor_min": divisor_min, "ignore_nan": ignore_nan},
    )
    check_performance(
        executor,
        enable_perf=(
            (rows, vocab) in perf_shape and mode == "normal" and divisor_min == _TINY
        ),
    )


@pytest.mark.extra_checks
def test_row_strided_input():
    from sgl_kernel_npu.operators.sample import div_argmax

    torch.manual_seed(42)
    x = torch.randn(3, 129 + 7, dtype=torch.float32)[:, :129]
    divisor = torch.rand(3, 129 + 7, dtype=torch.float32)[:, :129]
    assert x.stride(0) == 136 and x.stride(1) == 1
    executor = check_accuracy(div_argmax, args=(x, divisor))
    check_performance(executor, enable_perf=False)


@pytest.mark.extra_checks
def test_validation():
    from sgl_kernel_npu.operators.sample import div_argmax

    # Registered for arch35 (A5) only.
    impl, _ = get_test_impls()
    require_test_impl(div_argmax, impl)

    x = torch.ones(2, 4, device="npu")
    with pytest.raises(ValueError, match="shape"):
        div_argmax(x, x[:0], impl="auto")
    with pytest.raises(ValueError, match="float32"):
        div_argmax(x, x.bfloat16(), impl="auto")
    with pytest.raises(ValueError, match="NPU"):
        div_argmax(x.cpu(), x.cpu(), impl="auto")
    with pytest.raises(ValueError, match="vocab_size"):
        div_argmax(x[:, :0], x[:, :0], impl="auto")
    strided = torch.randn(2, 8, device="npu")[:, ::2]
    with pytest.raises(ValueError, match="stride"):
        div_argmax(strided, strided, impl="auto")
