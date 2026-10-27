import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shapes selected for performance collection.
perf_shape = [
    (1, 4096),
    (16, 7168),
]

# Source: sglang/.../quantization/modelslim/modelslim.py calls add_rmsnorm_bias
# with x and residual reshaped to [num_tokens, hidden_size]. The first two cases
# borrow Qwen3-8B and DeepSeek-V3 widths; token counts are illustrative decode
# token counts rather than traced production values.


def generate_test_data(shape, dtype, mode, seed=42):
    """Create NPU inputs using the device context supplied by conftest.

    normal: random input/residual/weight/bias exercises residual add + affine
        RMSNorm.
    zero: zero input and residual exercise eps handling and leave only the bias.
    offset_only: offset without scale must not enable quantization.
    quantized: symmetric values exercise truncation and int8 saturation, including
        masked tails when H is not a multiple of the 2048-element output tile.
    """
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[1], dtype=dtype)
    bias = torch.randn(shape[1], dtype=dtype)
    scale = offset = None
    if mode == "zero":
        x.zero_()
        residual.zero_()
    elif mode == "offset_only":
        offset = torch.full((shape[1],), 3.0, dtype=torch.float32)
    elif mode == "quantized":
        # Each input/residual pair sums to exactly +/-1, so normalization stays
        # deterministic. Quarter offsets also exercise truncation and both
        # saturation endpoints.
        x.fill_(0.25)
        x[:, 1::2] = -0.25
        residual.fill_(0.75)
        residual[:, 1::2] = -0.75
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
    return x, residual, weight, bias, scale, offset


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-1x4096"),  # ModelSlim single-token decode.
        pytest.param((16, 7168), id="deepseek-v3-16x7168"),  # 16-token decode.
        pytest.param((3, 6144), id="legacy-3x6144"),  # Width from the old test.
        pytest.param((1, 1), id="edge-1x1"),  # Minimum nonempty reduction.
        pytest.param(
            (7, 2049), id="edge-7x2049"
        ),  # Odd rows and a one-element tile tail.
        pytest.param(
            (128, 4096), id="big-batch-128x4096"
        ),  # Batch larger than the vector core count; rows loop over the grid.
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # Residual add plus nonzero affine RMSNorm.
        "zero",  # Zero variance must stay finite and preserve bias.
        "offset_only",  # No scale means offset is ignored, output keeps dtype.
        "quantized",  # Int8 truncation, saturation and tiled-output tail mask.
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_add_rmsnorm_bias(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import add_rmsnorm_bias

    x, residual, weight, bias, scale, offset = generate_test_data(shape, dtype, mode)
    executor = check_accuracy(
        add_rmsnorm_bias,
        args=(x, residual, weight, bias, eps, scale, offset),
    )
    enable_perf = shape in perf_shape and dtype == torch.float16 and mode == "normal"
    check_performance(executor, enable_perf=enable_perf)
