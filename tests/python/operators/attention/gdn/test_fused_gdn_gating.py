import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# rows = extend-token count, heads = value heads after TP sharding;
# A_log/dt_bias are FP32 parameters, as in the SGLang GDN layer.
perf_shape = [(8192, 4), (2048, 32)]


def generate_test_data(shape, dtype, mode, beta=1.0, threshold=20.0, b_dtype=None):
    """normal tests the log branch; large tests thresholded linear pass-through."""
    torch.npu.manual_seed(42)
    heads = shape[1]
    A_log = torch.randn(heads, dtype=torch.float32)
    dt_bias = torch.randn(heads, dtype=torch.float32)
    a = torch.randn(shape, dtype=dtype)
    if mode == "large":
        a.fill_(threshold / beta + 10.0)
    elif mode != "normal":
        raise ValueError(f"Unknown mode: {mode}")
    b = torch.randn(shape, dtype=dtype if b_dtype is None else b_dtype)
    return A_log, a, b, dt_bias


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((8192, 4), id="qwen3-next-tp8-extend"),  # 32 v-heads / TP8.
        pytest.param((2048, 32), id="qwen3-next-tp1-extend"),
        pytest.param((2048, 48), id="qwen3.5-27b-tp1-extend"),  # 48 v-heads.
        pytest.param((1, 1), id="edge-1x1"),
        pytest.param((1, 32), id="edge-single-token"),
        pytest.param((49, 32), id="edge-vectorcore-plus1"),  # arch22 has 48 cores.
        pytest.param((65, 33), id="edge-row-and-head-tail"),
        pytest.param((100, 7), id="edge-heads-below-blk"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["normal", "large"])
@pytest.mark.parametrize("beta,threshold", [(1.0, 20.0), (0.5, 4.0)])
def test_fused_gdn_gating(shape, dtype, mode, beta, threshold):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

    executor = check_accuracy(
        fused_gdn_gating,
        args=generate_test_data(shape, dtype, mode, beta, threshold),
        kwargs={"beta": beta, "threshold": threshold},
    )
    check_performance(
        executor,
        enable_perf=shape in perf_shape
        and dtype == torch.bfloat16
        and mode == "normal"
        and beta == 1.0,
    )


@pytest.mark.extra_checks
def test_mixed_input_dtypes():
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

    args = generate_test_data((17, 9), torch.float16, "normal", b_dtype=torch.bfloat16)
    check_accuracy(fused_gdn_gating, args=args)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case, match",
    [
        ("b_shape", "must match a shape"),
        ("a_stride", "must be contiguous"),
        ("A_log", "A_log must have"),
        ("dt_bias", "dt_bias must have"),
    ],
)
def test_rejects_unsupported_inputs(case, match):
    from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

    heads = 8
    A_log = torch.zeros(heads, dtype=torch.float32)
    dt_bias = torch.zeros(heads, dtype=torch.float32)
    a, b = torch.randn(4, heads), torch.randn(4, heads)
    if case == "b_shape":
        b = b[:, : heads - 1]
    elif case == "a_stride":
        a = torch.randn(4, 2 * heads)[:, ::2]
    elif case == "A_log":
        A_log = torch.zeros(heads - 1, dtype=torch.float32)
    else:
        dt_bias = torch.zeros(heads + 1, dtype=torch.float32)
    with pytest.raises(AssertionError, match=match):
        fused_gdn_gating(A_log, a, b, dt_bias, impl="triton")
