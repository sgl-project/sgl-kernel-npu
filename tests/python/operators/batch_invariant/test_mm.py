"""Batch-invariance tests for the public batch-invariant mm operator.

Each output row must be bitwise identical alone or inside a larger batch. The
operator has one external implementation, so cases compare it with itself at
different sizes; a float32 torch comparison only guards accuracy.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.operators.batch_invariant.batch_invariance import (
    assert_batch_invariant,
    assert_unchanged,
    require_batch_invariant_ops,
    snapshot,
)
from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_skip_acc_check, get_test_impls, make_executor
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# The kernel and torch.mm reduce in different orders, so fp32 results cannot
# meet the shared 1e-5 default; low-precision dtypes keep their defaults.
FP32_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3)}

# Decode-sized logits matmul; widths borrowed, not traced.
perf_shape = [
    (257, 7168, 16384),
]

# Rows, K, N.
shape_cases = [
    pytest.param((257, 7168, 16384), id="decode-257x7168x16384"),
    pytest.param((17, 128, 256), id="edge-17x128x256"),
    pytest.param((1, 1, 1), id="edge-1x1x1"),
    pytest.param((65, 1025, 513), id="edge-unaligned-65x1025x513"),
]


def generate_test_data(shape, dtype, seed=42):
    """Create an NPU left matrix and a strided, transposed right matrix."""
    torch.npu.manual_seed(seed)
    rows, k, n = shape
    mat1 = torch.randn(rows, k, dtype=dtype)
    mat2 = torch.randn(n, k, dtype=dtype).transpose(0, 1)
    return mat1, mat2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", shape_cases)
def test_mm_batch_invariant(shape, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    mat1, mat2 = generate_test_data(shape, dtype)
    if checked:
        snapshots = snapshot((mat1, mat2))
        rows = shape[0]
        full = mm_batch_invariant(mat1, mat2, impl=impl)
        for take in sorted({1, 2, 17, rows // 2}):
            if not 0 < take <= rows:
                continue
            # First, middle and last groups cover tile boundaries.
            for offset in sorted({0, rows // 2, rows - take}):
                result = mm_batch_invariant(
                    mat1[offset : offset + take], mat2, impl=impl
                )
                assert_batch_invariant(
                    full[offset : offset + take],
                    result,
                    path=f"rows[{offset}:{offset + take}]",
                )
        assert_unchanged((mat1, mat2), snapshots)
    executor = make_executor(
        mm_batch_invariant,
        args=(mat1, mat2),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and dtype == torch.float16
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", shape_cases[1:])
def test_mm_accuracy(shape, dtype):
    """Compare against a float32 torch.matmul within dtype tolerances."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    mat1, mat2 = generate_test_data(shape, dtype)
    result = mm_batch_invariant(mat1, mat2, impl=impl)
    expected = torch.mm(mat1.float(), mat2.float()).to(dtype)
    assert_accuracy(result, expected, standard=FP32_STANDARD)
