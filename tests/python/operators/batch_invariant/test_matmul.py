"""Batch-invariance tests for the public batch-invariant matmul operator.

Each matrix in the leading batch dimension must be bitwise identical alone or
inside a larger batch. Broadcast and batched right operands are both covered
because broadcasting must not change the reduction order.
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

# The kernel and torch.matmul reduce in different orders, so fp32 results
# cannot meet the shared 1e-5 default; low-precision dtypes keep theirs.
FP32_STANDARD = {"torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3)}

# Decode-sized batched matmul with a broadcast weight.
perf_shape = [
    (8, 1, 4096, 4096),
]

# Batch, rows, K, N and the right-operand mode.
matmul_cases = [
    pytest.param((8, 1, 4096, 4096), "broadcast", id="decode-8x1x4096x4096-broadcast"),
    pytest.param((17, 3, 64, 128), "broadcast", id="edge-odd-broadcast"),
    pytest.param((5, 7, 100, 33), "batched", id="edge-unaligned-batched"),
    pytest.param((16, 1, 128, 256), "batched", id="edge-batched-rhs"),
    pytest.param((1, 1, 1, 1), "broadcast", id="edge-1x1x1x1"),
]


def generate_test_data(shape, dtype, rhs_mode, seed=42):
    """Create NPU operands; broadcast weights are transposed and non-contiguous."""
    torch.npu.manual_seed(seed)
    batches, rows, k, n = shape
    mat1 = torch.randn(batches, rows, k, dtype=dtype)
    if rhs_mode == "broadcast":
        mat2 = torch.randn(n, k, dtype=dtype).transpose(0, 1)
    elif rhs_mode == "batched":
        mat2 = torch.randn(batches, k, n, dtype=dtype)
    else:
        raise ValueError(f"Unknown right-operand mode: {rhs_mode}")
    return mat1, mat2


def slice_rhs(mat2, offset, take, rhs_mode):
    return mat2[offset : offset + take] if rhs_mode == "batched" else mat2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,rhs_mode", matmul_cases)
def test_matmul_batch_invariant(shape, rhs_mode, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    mat1, mat2 = generate_test_data(shape, dtype, rhs_mode)
    if checked:
        snapshots = snapshot((mat1, mat2))
        batches = shape[0]
        full = matmul_batch_invariant(mat1, mat2, impl=impl)
        for take in sorted({1, 2, 7, batches // 2}):
            if not 0 < take <= batches:
                continue
            for offset in sorted({0, batches // 2, batches - take}):
                result = matmul_batch_invariant(
                    mat1[offset : offset + take],
                    slice_rhs(mat2, offset, take, rhs_mode),
                    impl=impl,
                )
                assert_batch_invariant(
                    full[offset : offset + take],
                    result,
                    path=f"batch[{offset}:{offset + take}]",
                )
        assert_unchanged((mat1, mat2), snapshots)
    executor = make_executor(
        matmul_batch_invariant,
        args=(mat1, mat2),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape and rhs_mode == "broadcast" and dtype == torch.float16
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,rhs_mode", matmul_cases[1:])
def test_matmul_accuracy(shape, rhs_mode, dtype):
    """Compare against a float32 torch.matmul within dtype tolerances."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    mat1, mat2 = generate_test_data(shape, dtype, rhs_mode)
    result = matmul_batch_invariant(mat1, mat2, impl=impl)
    expected = torch.matmul(mat1.float(), mat2.float()).to(dtype)
    assert_accuracy(result, expected, standard=FP32_STANDARD)
