import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shapes selected for performance collection.
perf_variance_shape = [
    (1, 8190, 512),
]
perf_rsqrt_mul_shape = [
    (1, 8190, 2560),
]

# Source: sglang/.../multimodal_gen/runtime/layers/layernorm.py runs
# tensor_parallel_rms_norm as fused_variance -> all_reduce(AVG) ->
# fused_rsqrt_mul, after casting x and the sharded weight to float32. C is the
# per-rank width (local heads * head_dim); TP-sharded Wan widths are multiples
# of head_dim=128. Token counts are illustrative, not traced production values.
# float32 is the only dtype this path uses, so other dtypes are not covered.
# fused_variance reduces with tl.arange(0, C), so its shapes use power-of-two
# widths; fused_rsqrt_mul masks tiles and also covers unaligned widths.


def generate_variance_data(shape, dtype, mode, seed=42):
    """Create NPU inputs using the device context supplied by conftest.

    normal: random input exercises the general reduction path.
    zero: zero input exercises eps handling; the variance must stay zero.
    constant: uniform magnitude checks the mean-square scale per row.
    """
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    if mode == "zero":
        x.zero_()
    elif mode == "constant":
        x.fill_(1.5)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return x


def generate_rsqrt_mul_data(shape, dtype, mode, variance_layout, seed=42):
    """Create inputs for the apply phase of the split RMSNorm.

    normal: random positive variance, matching the output of fused_variance.
    zero: zero input and variance exercise eps handling.
    constant: uniform input and variance check the normalization scale.
    """
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[2], dtype=dtype)
    if mode == "normal":
        variance = torch.randn(shape[0], shape[1], 1, dtype=dtype).abs() + 0.1
    elif mode == "zero":
        x.zero_()
        variance = torch.zeros(shape[0], shape[1], 1, dtype=dtype)
    elif mode == "constant":
        x.fill_(1.5)
        variance = torch.full((shape[0], shape[1], 1), 2.25, dtype=dtype)
    else:
        raise ValueError(f"Unknown data mode: {mode}")
    if variance_layout == "flat":
        variance = variance.reshape(-1)
    return x, variance, weight


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1024, 512), id="legacy-1x1024x512"),  # Old test shape.
        pytest.param((1, 8190, 512), id="legacy-1x8190x512"),  # Long sequence.
        pytest.param((2, 4096, 2048), id="tp2-2x4096x2048"),  # Wider TP shard.
        pytest.param((1, 1, 512), id="edge-single-token"),  # Minimum token count.
        pytest.param((1, 65, 1024), id="edge-65-tokens"),  # One token past 64.
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # General random reduction.
        "zero",  # Zero input must stay finite.
        "constant",  # Uniform rows check the mean-square scale.
    ],
)
def test_fused_variance(shape, dtype, mode):
    from sgl_kernel_npu.operators.norm import fused_variance

    x = generate_variance_data(shape, dtype, mode)
    executor = check_accuracy(fused_variance, args=(x,))
    enable_perf = shape in perf_variance_shape and mode == "normal"
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 8190, 2560), id="legacy-1x8190x2560"),  # Old test shape.
        pytest.param((1, 1024, 512), id="legacy-1x1024x512"),  # Short sequence.
        pytest.param((2, 4096, 640), id="wan-tp8-2x4096x640"),  # 5 heads * 128.
        pytest.param((1, 1, 128), id="edge-single-token"),  # Minimum token count.
        pytest.param((3, 17, 100), id="edge-3x17x100"),  # Unaligned width and rows.
    ],
)
@pytest.mark.parametrize("variance_layout", ["row", "flat"])
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # Random positive variance, as produced by fused_variance.
        "zero",  # Zero variance exercises eps handling.
        "constant",  # Uniform values check the normalization scale.
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_fused_rsqrt_mul(shape, dtype, mode, variance_layout, eps):
    from sgl_kernel_npu.operators.norm import fused_rsqrt_mul

    x, variance, weight = generate_rsqrt_mul_data(shape, dtype, mode, variance_layout)
    executor = check_accuracy(fused_rsqrt_mul, args=(x, variance, weight, eps))
    enable_perf = (
        shape in perf_rsqrt_mul_shape
        and mode == "normal"
        and variance_layout == "row"
        and eps == 1e-6
    )
    check_performance(executor, enable_perf=enable_perf)
