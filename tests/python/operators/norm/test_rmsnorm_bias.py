import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shapes selected for performance collection.
perf_shape = [
    (1, 4096),
    (16, 7168),
]


def generate_test_data(shape, dtype, mode, seed=42):
    """Create NPU inputs using the device context supplied by conftest.

    normal: random input/weight/bias exercises general affine RMSNorm.
    zero: zero variance exercises eps and should leave only the bias.
    offset_only: offset without scale must not enable quantization.
    quantized: signed values exercise truncation and int8 saturation, including
        masked tails when H is not a multiple of the 1024-element output tile.
    """
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[1], dtype=dtype)
    bias = torch.randn(shape[1], dtype=dtype)
    scale = offset = None
    if mode == "zero":
        x.zero_()
    elif mode == "offset_only":
        offset = torch.full((shape[1],), 3.0, dtype=torch.float32)
    elif mode == "quantized":
        # +/-1 inputs and quarter offsets avoid ambiguous rounding
        # boundaries; also exercise truncation and both saturation endpoints.
        x.fill_(1)
        x[:, 1::2] = -1
        weight.fill_(1)
        bias.zero_()
        scale = (
            torch.tensor(
                [0.5, 2.0, 200.0, 200.0],
                dtype=torch.float32,
            )
            .repeat((shape[1] + 3) // 4)[: shape[1]]
            .contiguous()
        )
        offset = torch.full((shape[1],), 0.25, dtype=torch.float32)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return x, weight, bias, scale, offset


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-1x4096"),  # Single-token decode.
        pytest.param((16, 7168), id="deepseek-v3-16x7168"),  # Batched decode.
        pytest.param((1, 1), id="edge-1x1"),  # Minimum nonempty reduction.
        pytest.param(
            (7, 1025), id="edge-7x1025"
        ),  # Odd rows and a one-element tile tail.
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # General RMSNorm + nonzero affine bias.
        "zero",  # Zero variance must stay finite and preserve bias.
        "offset_only",  # No scale means offset is ignored, output keeps dtype.
        "quantized",  # Int8 truncation, saturation and tiled-output tail mask.
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import rmsnorm_bias

    x, weight, bias, scale, offset = generate_test_data(shape, dtype, mode)
    executor = check_accuracy(
        rmsnorm_bias,
        args=(x, weight, bias, eps, scale, offset),
    )
    enable_perf = shape in perf_shape and dtype == torch.float16 and mode == "quantized"
    check_performance(executor, enable_perf=enable_perf)
