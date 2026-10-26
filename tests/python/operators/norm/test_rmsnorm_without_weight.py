import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Shapes selected for performance collection.
perf_shape = [
    (1, 130, 2048),
    (2, 1, 4096),
]

# Source: sglang/.../multimodal_gen/runtime/layers/layernorm.py routes
# RMSNormNoWeight.forward_npu to fused_rmsnorm_without_weight(x, eps) from the
# LTX-2 DiT blocks. Video width is 32*128=4096 and audio width is 32*64=2048;
# token counts below are illustrative decode-style values, not traced shapes.


def generate_test_data(shape, dtype, mode, seed=42):
    """Create NPU inputs using the device context supplied by conftest.

    normal: random input exercises the general reduction path.
    zero: zero input exercises eps handling and must produce a zero output.
    constant: uniform magnitude checks the normalization scale per row.
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 130, 2048), id="ltx2-audio-1x130x2048"),  # Old test shape.
        pytest.param((2, 16, 2048), id="ltx2-audio-2x16x2048"),  # Audio width.
        pytest.param((2, 1, 4096), id="ltx2-video-2x1x4096"),  # Video width.
        pytest.param((1, 1, 2048), id="edge-single-token"),  # Minimum token count.
        pytest.param((1, 65, 2048), id="edge-65-tokens"),  # One token past block_l=64.
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "normal",  # General random reduction.
        "zero",  # Zero variance must stay finite.
        "constant",  # Uniform rows check the normalization scale.
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_fused_rmsnorm_without_weight(
    shape,
    dtype,
    mode,
    eps,
):
    from sgl_kernel_npu.operators.norm import fused_rmsnorm_without_weight

    x = generate_test_data(shape, dtype, mode)
    executor = check_accuracy(
        fused_rmsnorm_without_weight,
        args=(x, eps),
    )
    enable_perf = shape in perf_shape and dtype == torch.bfloat16 and mode == "normal"
    check_performance(executor, enable_perf=enable_perf)
