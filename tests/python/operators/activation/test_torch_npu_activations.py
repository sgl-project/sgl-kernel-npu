"""Compare the public TorchNPU activation operators against PyTorch formulas."""

import pytest

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")


def generate_test_data(shape, dtype):
    torch.npu.manual_seed(42)
    return torch.randn(shape, dtype=dtype) * 0.5


@pytest.mark.parametrize(
    "shape",
    [
        # One 224x224 image before spatial merge: (224 / 14)^2 patches.
        # Qwen2-VL-7B vision MLP at TP1 has width 1280 * 4.
        pytest.param((256, 1, 5120), id="qwen2-vl-7b-vision-mlp-tp1"),
    ],
)
def test_fast_gelu(shape):
    from sgl_kernel_npu.operators.activation import fast_gelu

    executor = check_accuracy(
        fast_gelu, args=(generate_test_data(shape, torch.bfloat16),)
    )
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "shape, dim, approximate, activate_left",
    [
        # Gemma 2 2B packs local gate/up shards (9216 / 2 each) at TP2.
        pytest.param((16, 9216), -1, 1, True, id="gemma-2-2b-prefill-mlp-tp2"),
    ],
)
def test_geglu(shape, dim, approximate, activate_left):
    from sgl_kernel_npu.operators.activation import geglu

    x = generate_test_data(shape, torch.bfloat16)
    executor = check_accuracy(geglu, args=(x, dim, approximate, activate_left))
    check_performance(executor, enable_perf=True)


@pytest.mark.parametrize(
    "shape, dim",
    [
        # Qwen3-8B packs local gate/up shards (12288 / 4 each) at TP4.
        pytest.param((16, 6144), -1, id="qwen3-8b-prefill-mlp-tp4"),
    ],
)
def test_swiglu(shape, dim):
    from sgl_kernel_npu.operators.activation import swiglu

    x = generate_test_data(shape, torch.bfloat16)
    executor = check_accuracy(swiglu, args=(x, dim))
    check_performance(executor, enable_perf=True)
