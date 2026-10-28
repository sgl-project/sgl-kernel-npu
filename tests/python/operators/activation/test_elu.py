"""ELU operator tests through the public API.

ELU is element-wise, so the interesting cases are dtype, ``alpha``, the
flatten/pad/restore path for unaligned element counts, and numerical extremes.

The shared ``tests/python/utils`` accuracy/performance helpers (and their
``conftest.py``) land with the test-framework refactor. When they are absent
from the current checkout, this module falls back to a self-contained
comparison so the operator is still exercised instead of failing at import.
"""

import pytest

torch = pytest.importorskip("torch")

try:  # Shared helpers land with the test-framework refactor.
    from tests.python.utils.accuracy import check_accuracy
    from tests.python.utils.performance import check_performance
except ImportError:  # pragma: no cover - depends on the checkout
    check_accuracy = check_performance = None

# Shapes selected for performance collection (production shapes below).
perf_shape = [
    (1, 4096),
    (16, 7168),
]

# Per-dtype tolerance, matching the shared accuracy standards.
#
# fp16 note: the kernel evaluates ``exp(min(x,0)) - 1`` directly in fp16, so
# the *relative* error grows near x -> 0- (measured ~2.3% at x = -0.001). The
# *absolute* error there is only ~2.3e-5, and ``assert_close`` compares against
# ``atol + rtol * |ref|``, so the standard 1e-3 still holds. Use fp32 when
# relative accuracy in that narrow band matters.
_ATOL_RTOL = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
}


def generate_test_data(shape, dtype, mode, alpha):
    """Build NPU inputs using the device context supplied by conftest.

    normal: random values around zero exercise both ELU branches.
    wide:   -30..30 sweeps exp underflow on the negative side, so fp16/fp32
            must not overflow and the negative branch saturates correctly.
    tail:   a one-element tensor is shorter than a single tile and forces the
            host-side pad-to-one-tile path.
    alpha:  a non-unit coefficient combined with unaligned element counts.
    empty:  a zero-element tensor must short-circuit without launching.
    """
    torch.npu.manual_seed(42)
    if mode == "wide":
        x = torch.linspace(-30.0, 30.0, shape[0], dtype=torch.float32)
        return x.to(dtype).npu(), alpha
    if mode == "tail":
        return torch.randn(1, dtype=dtype).npu(), alpha
    if mode == "alpha":
        return (torch.randn(shape, dtype=dtype) * 2.0).npu(), alpha
    if mode == "empty":
        return torch.empty(0, dtype=dtype).npu(), alpha
    if mode == "normal":
        return (torch.randn(shape, dtype=dtype) * 2.0).npu(), alpha
    raise ValueError(f"Unknown data mode: {mode}")


def _assert_elu(x, alpha, dtype):
    """Compare the Ascend C result against the independent torch_native one."""
    from sgl_kernel_npu.operators.activation import elu

    ref = elu(x, alpha, impl="torch_native")
    out = elu(x, alpha, impl="ascendc")
    assert tuple(out.shape) == tuple(x.shape), (out.shape, x.shape)
    assert out.dtype == x.dtype, (out.dtype, x.dtype)
    # auto is a declared alias of the Ascend C entry point, not a fallback, so
    # both must dispatch to the same kernel and agree bit for bit.
    auto = elu(x, alpha)
    assert torch.equal(auto, out), "auto must dispatch to the ascendc entry point"
    atol, rtol = _ATOL_RTOL[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-activation-width"),
        pytest.param((16, 7168), id="deepseek-v3-activation-width"),
        pytest.param((777,), id="edge-unaligned-flat"),  # 1-D, odd element count.
        pytest.param((5, 7, 2), id="edge-unaligned-3d"),  # Multi-dim + padding.
        pytest.param((7, 1025), id="edge-tail"),  # One-element tile tail.
    ],
)
@pytest.mark.parametrize("mode", ["normal", "alpha"])
def test_elu(shape, dtype, mode):
    from sgl_kernel_npu.operators.activation import elu

    alpha = 1.67326 if mode == "alpha" else 1.0
    x, _ = generate_test_data(shape, dtype, mode, alpha)

    if check_accuracy is None:
        _assert_elu(x, alpha, dtype)
        return

    executor = check_accuracy(elu, args=(x,), kwargs={"alpha": alpha})
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and dtype == torch.float16 and mode == "normal"),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape, mode",
    [
        pytest.param((2048,), "wide", id="wide-range-flat"),
        pytest.param((1,), "tail", id="edge-smallest"),
        pytest.param((0,), "empty", id="edge-empty"),
    ],
)
def test_elu_edge_cases(shape, dtype, mode):
    from sgl_kernel_npu.operators.activation import elu

    x, alpha = generate_test_data(shape, dtype, mode, 1.0)

    if check_accuracy is None:
        if x.numel() == 0:
            assert elu(x, alpha, impl="ascendc").numel() == 0
        else:
            _assert_elu(x, alpha, dtype)
        return

    check_accuracy(elu, args=(x,), kwargs={"alpha": alpha})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_elu_near_zero_negative_branch(dtype):
    """Pin the exp(min(x,0)) - 1 cancellation behaviour close to zero.

    Random inputs almost never land in this narrow band, so the generic cases
    cannot cover it. The fp16 relative error is large here while the absolute
    error stays around 2e-5; the shared tolerance still holds because
    ``assert_close`` adds an atol term.
    """
    from sgl_kernel_npu.operators.activation import elu

    x = torch.tensor([-0.001, -0.002, -0.005, -0.01, -0.02], dtype=dtype, device="npu")
    ref = torch.where(x > 0, x, x.float().expm1()).to(dtype)
    torch.testing.assert_close(elu(x, impl="ascendc"), ref, atol=1e-3, rtol=1e-3)


def test_elu_alpha_zero_is_identity_on_positives():
    """alpha = 0 collapses the negative branch to zero and keeps x on x > 0."""
    from sgl_kernel_npu.operators.activation import elu

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device="npu")
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    torch.testing.assert_close(elu(x, 0.0, impl="ascendc").cpu(), expected, atol=0, rtol=0)


def test_elu_rejects_unsupported_dtype():
    from sgl_kernel_npu.operators.activation import elu

    x = torch.ones(8, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError):
        elu(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int32, torch.float64])
def test_elu_rejects_unsupported_dtypes(dtype):
    """bfloat16 is the dtype a user is most likely to hit by accident."""
    from sgl_kernel_npu.operators.activation import elu

    x = torch.ones(64, dtype=torch.float32).to(dtype) if dtype == torch.float64 else torch.ones(64, dtype=dtype)
    with pytest.raises(RuntimeError):
        elu(x.npu(), impl="ascendc")


def test_elu_rejects_zero_dim():
    """The kernel requires at least one dimension; a scalar must be rejected."""
    from sgl_kernel_npu.operators.activation import elu

    with pytest.raises(RuntimeError):
        elu(torch.tensor(1.0, device="npu"))


def test_elu_rejects_unknown_impl():
    from sgl_kernel_npu.operators.activation import elu

    with pytest.raises(ValueError):
        elu(torch.randn(8, device="npu"), impl="bogus")


def test_elu_non_contiguous():
    """ascendc contracts contiguity; torch_native materialises and still works."""
    from sgl_kernel_npu.operators.activation import elu

    x = torch.randn(64, 128, dtype=torch.float32, device="npu")[:, ::2]
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError):
        elu(x, impl="ascendc")
    ref = torch.where(x > 0, x, x.float().expm1())
    torch.testing.assert_close(elu(x, impl="torch_native"), ref, atol=1e-5, rtol=1e-5)


def test_elu_does_not_modify_input():
    """elu returns a new tensor; the input must be left untouched."""
    from sgl_kernel_npu.operators.activation import elu

    x = torch.randn(1024, dtype=torch.float32, device="npu")
    before = x.clone()
    out = elu(x)
    assert out.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, before, atol=0, rtol=0)


def test_elu_repeated_calls_use_cached_backends():
    """The lazy torch_native cache must keep returning identical results."""
    from sgl_kernel_npu.operators.activation import elu

    x = torch.randn(256, dtype=torch.float32, device="npu")
    first = elu(x, impl="torch_native")
    second = elu(x, impl="torch_native")
    torch.testing.assert_close(first, second, atol=0, rtol=0)

