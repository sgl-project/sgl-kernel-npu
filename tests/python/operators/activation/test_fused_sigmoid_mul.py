"""Public-interface tests for the Qwen3.5 attention output gate.

In SGLang (sglang/srt/models/qwen3_5.py),
Qwen3_5AttentionDecoderLayer.self_attention applies the gate before the
output projection on NPU. Qwen3.5-27B (dense) and Qwen3.5-35B-A3B (MoE)
provide the model-based widths: local query heads times head dimension.
Token counts are synthetic decode and prefill scenarios.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Preserve the legacy tolerances, including its stricter BF16 threshold.
STANDARDS = {
    "torch.float32": AccuracyStandard(atol=1e-5, rtol=1e-5),
    "torch.float16": AccuracyStandard(atol=1e-3, rtol=1e-3),
    "torch.bfloat16": AccuracyStandard(atol=1.6e-2, rtol=1.6e-2),
}
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
perf_shape = [
    (1, 4096),
    (32, 3584),
    (128, 5120),
    (1, 6144),
    (4096, 4096),
    (4096, 6144),
]


def generate_test_data(shape, dtype, gate_dtype=None):
    torch.npu.manual_seed(42)
    if gate_dtype is None:
        gate_dtype = dtype
    return torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=gate_dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        # Retain all five legacy synthetic shapes.
        pytest.param((1, 4096), id="legacy-1x4096"),
        pytest.param((16, 4096), id="legacy-16x4096"),
        pytest.param((32, 3584), id="legacy-32x3584"),
        pytest.param((128, 5120), id="legacy-128x5120"),
        pytest.param((4, 16, 256), id="legacy-3d"),
        # Scalars and the 2048-element block boundary, including masked tails.
        pytest.param((), id="scalar"),
        pytest.param((2048,), id="one-tile"),
        pytest.param((2049,), id="tile-plus-one"),
        pytest.param((7, 33), id="odd-rows-unaligned-width"),
        # Qwen3.5-35B-A3B: 16 attention heads * 256 = 4096 at TP1.
        pytest.param((4096, 4096), id="qwen35-35b-tp1-prefill-synthetic"),
        pytest.param((32, 2048), id="qwen35-35b-tp2-decode-synthetic"),
        # Qwen3.5-27B: 24 attention heads * 256 = 6144 at TP1.
        pytest.param((1, 6144), id="qwen35-27b-tp1-decode-synthetic"),
        pytest.param((4096, 6144), id="qwen35-27b-tp1-prefill-synthetic"),
        pytest.param((32, 3072), id="qwen35-27b-tp2-decode-synthetic"),
    ],
)
def test_fused_sigmoid_mul(shape, dtype):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    x, gate = generate_test_data(shape, dtype)
    executor = check_accuracy(fused_sigmoid_mul, args=(x, gate), standard=STANDARDS)
    check_performance(
        executor,
        enable_perf=shape in perf_shape and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate_value", [0.0, -20.0, 20.0])
def test_fused_sigmoid_mul_gate_values(dtype, gate_value):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    x, gate = generate_test_data((7, 33), dtype)
    gate.fill_(gate_value)
    check_accuracy(fused_sigmoid_mul, args=(x, gate), standard=STANDARDS)


@pytest.mark.parametrize(
    "x_dtype,gate_dtype",
    [
        (x_dtype, gate_dtype)
        for x_dtype in DTYPES
        for gate_dtype in DTYPES
        if x_dtype != gate_dtype
    ],
)
def test_fused_sigmoid_mul_mixed_dtypes(x_dtype, gate_dtype):
    """Preserve the legacy per-input FP32 promotion and x-dtype output."""
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    x, gate = generate_test_data((7, 2049), x_dtype, gate_dtype)
    check_accuracy(fused_sigmoid_mul, args=(x, gate), standard=STANDARDS)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_sigmoid_mul_empty(dtype):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    x, gate = generate_test_data((0, 4096), dtype)
    check_accuracy(fused_sigmoid_mul, args=(x, gate), standard=STANDARDS)
    # No profiling: an empty output launches no device task.


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_sigmoid_mul_non_contiguous(dtype):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    x, gate = generate_test_data((33, 7), dtype)
    check_accuracy(fused_sigmoid_mul, args=(x.T, gate.T), standard=STANDARDS)


@pytest.mark.extra_checks
def test_fused_sigmoid_mul_rejects_shape_mismatch():
    """Preserve the legacy same-shape requirement without testing new checks."""
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul

    impl, _ = get_test_impls()
    x = torch.ones((2, 3), dtype=torch.float32)
    gate = torch.ones((2, 1), dtype=x.dtype)
    with pytest.raises(ValueError, match="same shape"):
        fused_sigmoid_mul(x, gate, impl=impl)
