"""Batch-invariance tests for the public batch-invariant log-softmax.

Each row must be bitwise identical alone or inside a larger batch.
Log-softmax over the batch dimension is not tested because it depends on size.
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

# Token logits with a local vocabulary shard.
perf_shape = [
    (257, 16384),
]

shape_cases = [
    pytest.param((257, 16384), id="decode-257x16384"),
    pytest.param((17, 1025), id="edge-17x1025"),
    pytest.param((1, 1), id="edge-1x1"),
    pytest.param((3, 4096), id="edge-3x4096"),
]


def generate_test_data(shape, dtype, mode, seed=42):
    """Create NPU logits; offset shifts by 100, equal repeats one value."""
    torch.npu.manual_seed(seed)
    tensor = torch.randn(shape, dtype=dtype)
    if mode == "offset":
        tensor.add_(100.0)
    elif mode == "equal":
        tensor.fill_(0.5)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return tensor


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "offset", "equal"])
@pytest.mark.parametrize("shape", shape_cases)
def test_log_softmax_batch_invariant(shape, mode, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import log_softmax_batch_invariant

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    tensor = generate_test_data(shape, dtype, mode)
    dim = -1
    if checked:
        snapshots = snapshot((tensor,))
        rows = shape[0]
        full = log_softmax_batch_invariant(tensor, dim, impl=impl)
        for offset in sorted({0, rows // 2, rows - 1}):
            result = log_softmax_batch_invariant(
                tensor[offset : offset + 1], dim, impl=impl
            )
            assert_batch_invariant(
                full[offset : offset + 1], result, path=f"rows[{offset}:{offset + 1}]"
            )
        assert_unchanged((tensor,), snapshots)
    executor = make_executor(
        log_softmax_batch_invariant,
        args=(tensor, dim),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(executor, enable_perf=shape in perf_shape and mode == "normal")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
# The aclnn kernel only reduces the last dimension, as production calls do.
@pytest.mark.parametrize("dim", [-1])
def test_log_softmax_accuracy(dim, dtype):
    """Compare against a float32 torch log_softmax within dtype tolerances."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import log_softmax_batch_invariant

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    tensor = generate_test_data((17, 1025), dtype, "normal")
    result = log_softmax_batch_invariant(tensor, dim, impl=impl)
    expected = torch.log_softmax(tensor.float(), dim=dim).to(dtype)
    assert_accuracy(result, expected)
