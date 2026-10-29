"""Batch-invariance tests for the public batch-invariant add+rms_norm.

Each row's normalized output, reciprocal standard deviation and residual sum
must be bitwise identical alone or inside a larger batch.
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

# Batched decode width borrowed from DeepSeek-V3, not a traced call.
perf_shape = [
    (16, 7168),
]

# Rows, hidden.
shape_cases = [
    pytest.param((16, 7168), id="decode-16x7168"),
    pytest.param((1, 4096), id="decode-1x4096"),
    pytest.param((7, 1025), id="edge-7x1025"),
    pytest.param((1, 1), id="edge-1x1"),
]


def generate_test_data(shape, dtype, mode, seed=42):
    """Create NPU inputs; cancel keeps large inputs with a small sum, zero
    exercises the eps path."""
    torch.npu.manual_seed(seed)
    hidden = shape[1]
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn(shape, dtype=dtype)
    weight = torch.randn(hidden, dtype=dtype)
    if mode == "cancel":
        x.add_(4.0)
        residual.fill_(-4.0)
    elif mode == "zero":
        x.zero_()
        residual.zero_()
    elif mode != "normal":
        raise ValueError(f"Unknown data mode: {mode}")
    return x, residual, weight


def torch_reference(x, residual, weight, eps):
    """Reference add + RMSNorm computed in float32 from the same inputs."""
    residual_out = x + residual
    summed = residual_out.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    out = (summed * rstd * weight.float()).to(x.dtype)
    return out, rstd, residual_out


def select_rows(outputs, offset, count):
    """Slice row-aligned outputs; None leaves pass through.

    SGLang's local split path returns None for rstd, so that slot may be
    undefined in the fused operator too.
    """
    return tuple(
        output if output is None else output[offset : offset + count]
        for output in outputs
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@pytest.mark.parametrize("mode", ["normal", "cancel", "zero"])
@pytest.mark.parametrize("shape", shape_cases)
def test_add_rms_norm_batch_invariant(shape, mode, eps, dtype):
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import add_rms_norm_batch_invariant

    impl, impl_ref = get_test_impls()
    checked = get_skip_acc_check() is not True
    x, residual, weight = generate_test_data(shape, dtype, mode)
    if checked:
        snapshots = snapshot((x, residual, weight))
        rows = shape[0]
        full = add_rms_norm_batch_invariant(x, residual, weight, eps, impl=impl)
        for offset in sorted({0, rows // 2, rows - 1}):
            result = add_rms_norm_batch_invariant(
                x[offset : offset + 1],
                residual[offset : offset + 1],
                weight,
                eps,
                impl=impl,
            )
            assert_batch_invariant(
                select_rows(full, offset, 1),
                result,
                path=f"rows[{offset}:{offset + 1}]",
            )
        assert_unchanged((x, residual, weight), snapshots)
    executor = make_executor(
        add_rms_norm_batch_invariant,
        args=(x, residual, weight, eps),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and mode == "normal"
            and eps == 1e-6
            and dtype == torch.bfloat16
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_add_rms_norm_accuracy(eps, dtype):
    """Compare the three outputs with a float32 torch reference."""
    require_batch_invariant_ops()
    from sgl_kernel_npu.operators.batch_invariant import add_rms_norm_batch_invariant

    if get_skip_acc_check() is True:
        pytest.skip("Accuracy checks are disabled by --skip-acc-check=true")

    impl, _ = get_test_impls()
    x, residual, weight = generate_test_data((16, 7168), dtype, "normal")
    result = add_rms_norm_batch_invariant(x, residual, weight, eps, impl=impl)
    out, rstd, residual_out = torch_reference(x, residual, weight, eps)
    expected = (out, None if result[1] is None else rstd, residual_out)
    # rstd is fp32, so it gets a stricter tolerance than the low-precision
    # outputs.
    assert_accuracy(
        result,
        expected,
        standard={
            "torch.float16": AccuracyStandard(atol=2e-3, rtol=2e-3),
            "torch.float32": AccuracyStandard(atol=1e-3, rtol=1e-3),
        },
    )
