import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shapes selected for performance collection.
perf_shape = [
    # Qwen-Image 2.1 block width at 1024x1024: 4096 image tokens.
    (1, 4096, 4096),
]

# Both eager operations round to BF16; the fused kernel must match exactly.
EXACT_BF16 = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=0.0)}


def generate_data(shape, gate_mode, seed=42):
    """Create NPU inputs for a full-shape or single-row gate.

    full: one gate element per update element (per-token modulation).
    row: a single broadcast row, as the per-timestep modulation uses.
    """
    torch.npu.manual_seed(seed)
    residual = torch.randn(shape, dtype=torch.bfloat16)
    update = torch.randn(shape, dtype=torch.bfloat16)
    if gate_mode == "row":
        gate = torch.randn((shape[-1],), dtype=torch.bfloat16)
    elif gate_mode == "full":
        gate = torch.randn(shape, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")
    return residual, update, gate


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096, 4096), id="qwen-image-2.1-block-width"),
        pytest.param((1, 16, 128), id="edge-minimum-tile"),
        pytest.param(
            (2, 64, 256),
            id="edge-two-batches",
            marks=pytest.mark.extra_checks,
        ),
    ],
)
@pytest.mark.parametrize("gate_mode", ["full", "row"])
def test_residual_gate_add(shape, gate_mode):
    from sgl_kernel_npu.operators.diffusion import residual_gate_add

    residual, update, gate = generate_data(shape, gate_mode)
    executor = check_accuracy(
        residual_gate_add,
        args=(residual, update, gate),
        standard=EXACT_BF16,
    )
    check_performance(
        executor,
        enable_perf=shape in perf_shape and gate_mode == "row",
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 16, 129), id="non-power-of-two-width"),
        pytest.param((1, 8, 128), id="fewer-than-minimum-rows"),
    ],
)
def test_residual_gate_add_rejects_unsupported_shape(shape):
    from sgl_kernel_npu.operators.diffusion import residual_gate_add

    residual, update, gate = generate_data(shape, "row")
    with pytest.raises(ValueError):
        residual_gate_add(residual, update, gate, impl="triton")
