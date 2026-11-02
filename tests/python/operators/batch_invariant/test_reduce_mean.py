"""Batch-invariance tests for the public batch-invariant mean reduction.

Each row must be bitwise identical alone or inside a larger batch. Reducing
the batch dimension is not tested because it depends on the batch size.
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
    """Create NPU input; spread scales by 100 to amplify ordering effects."""
    torch.npu.manual_seed(seed)
    tensor = torch.randn(shape, dtype=dtype)
    if mode == "spread":
        tensor.mul_(100.0)
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return tensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "spread"])
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
@pytest.mark.parametrize("shape", shape_cases)
def test_reduce_mean_batch_invariant(shape, keepdim, mode, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import reduce_mean_batch_invariant

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    tensor = generate_test_data(shape, dtype, mode)
    dim = -1
    if checked:
        snapshots = snapshot((tensor,))
        rows = shape[0]
        full = reduce_mean_batch_invariant(tensor, dim, keepdim, impl=impl)
        for offset in sorted({0, rows // 2, rows - 1}):
            result = reduce_mean_batch_invariant(
                tensor[offset : offset + 1], dim, keepdim, impl=impl
            )
            assert_batch_invariant(
                full[offset : offset + 1], result, path=f"rows[{offset}:{offset + 1}]"
            )
        assert_unchanged((tensor,), snapshots)
    executor = make_executor(
        reduce_mean_batch_invariant,
        args=(tensor, dim, keepdim),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(
        executor, enable_perf=shape in perf_shape and mode == "normal" and not keepdim
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [1, -1])
@pytest.mark.parametrize("keepdim", [False, True], ids=["dropdim", "keepdim"])
def test_reduce_mean_accuracy(dim, keepdim, dtype):
    """Compare against a float32 torch mean within dtype tolerances."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import reduce_mean_batch_invariant

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    tensor = generate_test_data((17, 1025), dtype, "normal")
    result = reduce_mean_batch_invariant(tensor, dim, keepdim, impl=impl)
    expected = tensor.float().mean(dim=dim, keepdim=keepdim).to(dtype)
    assert_accuracy(result, expected)
