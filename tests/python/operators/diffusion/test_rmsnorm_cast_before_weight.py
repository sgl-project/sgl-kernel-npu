import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shapes selected for performance collection.
perf_shape = [
    # Qwen-Image 2.1 Q/K normalization at 1024x1024: 4096 tokens, 32 heads of 128.
    (1, 4096, 32, 128),
]


def generate_data(shape, dtype, seed=42):
    torch.npu.manual_seed(seed)
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    return x, weight


def exact_standard(dtype):
    return {str(dtype): AccuracyStandard(atol=0.0, rtol=0.0)}


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16", marks=pytest.mark.extra_checks),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4096, 32, 128), id="qwen-image-2.1-qk-1024"),
        pytest.param((16, 128), id="edge-minimum-rows"),
        pytest.param(
            (128, 512),
            id="edge-wide-row",
            marks=pytest.mark.extra_checks,
        ),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_cast_before_weight(shape, dtype, eps):
    from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

    x, weight = generate_data(shape, dtype)
    executor = check_accuracy(
        rmsnorm_cast_before_weight,
        args=(x, weight, eps),
        standard=exact_standard(dtype),
    )
    check_performance(
        executor,
        enable_perf=(shape in perf_shape and dtype is torch.bfloat16 and eps == 1e-6),
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((7, 1025), id="non-power-of-two-width"),
        pytest.param((1, 128), id="fewer-than-minimum-rows"),
    ],
)
def test_rmsnorm_cast_before_weight_rejects_unsupported_shape(shape):
    from sgl_kernel_npu.operators.diffusion import rmsnorm_cast_before_weight

    x, weight = generate_data(shape, torch.bfloat16)
    with pytest.raises(ValueError):
        rmsnorm_cast_before_weight(x, weight, 1e-6, impl="triton")
