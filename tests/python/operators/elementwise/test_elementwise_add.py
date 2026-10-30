import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shapes selected for performance collection: decode-shaped activations.
perf_shape = [
    (1, 4096),
    (16, 7168),
]


def generate_data(shape, dtype, mode, seed=42):
    """Create NPU inputs using the device context supplied by conftest.

    normal: independent random addends.
    zeros: y is zero, so the result must equal x exactly.
    cancelling: equal and opposite addends; the fp16 path must not saturate.
    """
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    y = torch.randn(shape, dtype=dtype)
    if mode == "zeros":
        y.zero_()
    elif mode == "cancelling":
        x.fill_(1000.0)
        y.fill_(-1000.0)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return x, y


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096), id="qwen3-8b-1x4096"),  # Single-token decode.
        pytest.param((16, 7168), id="deepseek-v3-16x7168"),  # Batched decode.
        pytest.param((1, 1), id="edge-1x1"),  # Minimum nonempty input.
        pytest.param((7, 1025), id="edge-7x1025"),  # Odd numel, tile tail.
        pytest.param((4096,), id="edge-1d-4096"),  # 1-D input keeps its shape.
        pytest.param((2, 3, 5), id="edge-3d-2x3x5"),  # Rank-3 input.
        pytest.param((0, 128), id="edge-empty"),  # Empty input, no launch.
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # General add.
        "zeros",  # Identity case.
        "cancelling",  # Large equal/opposite operands.
    ],
)
def test_elementwise_add(shape, dtype, mode):
    from sgl_kernel_npu.operators.elementwise import elementwise_add

    x, y = generate_data(shape, dtype, mode)
    executor = check_accuracy(elementwise_add, args=(x, y))
    # An accuracy failure raises before performance collection.
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and dtype == torch.float16 and mode == "normal"),
    )


def test_elementwise_add_rejects_bfloat16():
    """bfloat16 is not implemented yet; the host launcher must reject it."""
    from sgl_kernel_npu.operators.elementwise import elementwise_add

    x = torch.randn((8, 8), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="float16 and float32"):
        elementwise_add(x, x, impl="ascendc")


def test_elementwise_add_rejects_non_contiguous():
    """Non-contiguous inputs are rejected instead of being silently copied."""
    from sgl_kernel_npu.operators.elementwise import elementwise_add

    x = torch.randn((16, 16), dtype=torch.float32).t()
    with pytest.raises(RuntimeError, match="contiguous"):
        elementwise_add(x, x, impl="ascendc")
