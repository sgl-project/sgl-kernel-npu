import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shapes selected for performance collection.
perf_shape = [
    # Qwen-Image 2.1 FFN at 1024x1024: 4096 image tokens, 12288-wide gate/up.
    (1, 4096, 12288),
]

# Both eager operations round to BF16; the fused kernel must match exactly.
EXACT_BF16 = {"torch.bfloat16": AccuracyStandard(atol=0.0, rtol=0.0)}


def generate_data(shape, seed=42):
    torch.npu.manual_seed(seed)
    return (
        torch.randn(shape, dtype=torch.bfloat16),
        torch.randn(shape, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096, 12288), id="qwen-image-2.1-ffn-1024"),
        pytest.param((1, 1), id="edge-minimum"),
        pytest.param((7, 1025), id="edge-tail"),
    ],
)
def test_silu_mul(shape):
    from sgl_kernel_npu.operators.diffusion import silu_mul

    a, b = generate_data(shape)
    executor = check_accuracy(silu_mul, args=(a, b), standard=EXACT_BF16)
    # Accuracy failures raise before any profiling runs.
    check_performance(executor, enable_perf=shape in perf_shape)
