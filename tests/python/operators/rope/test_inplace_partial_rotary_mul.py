"""Partial RoPE accuracy with independent in-place buffers and NPU profiling."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

# Install bundled vendor paths before the shared fixture initializes the NPU.
import sgl_kernel_npu  # noqa: F401

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(16, 1, 64, 512)]


# DeepSeek-V4-Flash attention Q: D=512, tail RoPE=[448,512].
def generate_data(shape, partial_slice):
    torch.manual_seed(15)
    x = torch.randn(shape, dtype=torch.bfloat16)
    width = partial_slice[1] - partial_slice[0]
    angles = torch.randn(shape[0], 1, 1, width // 2).repeat_interleave(2, -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    kwargs = dict(rotary_mode="interleave", partial_slice=partial_slice)
    return (x, cos, sin), kwargs


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 512), id="deepseek-v4-flash-decode-t1-n64-d512"),
        pytest.param((16, 1, 64, 512), id="deepseek-v4-flash-decode-t16-n64-d512"),
    ],
)
@pytest.mark.parametrize(
    "partial_slice",
    [
        pytest.param([448, 512], id="rope64"),
        pytest.param(
            [0, 512], id="synthetic-full-rope", marks=pytest.mark.extra_checks
        ),
    ],
)
def test_partial_rope(shape, partial_slice):
    from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

    args, kwargs = generate_data(shape, partial_slice)
    x, cos, sin = args
    executor = make_executor(
        inplace_partial_rotary_mul, args=(x, cos, sin), kwargs=kwargs
    )
    if get_skip_acc_check() is True:
        require_test_impl(inplace_partial_rotary_mul, executor.impl)
    else:
        require_distinct_impls(
            inplace_partial_rotary_mul, executor.impl, executor.impl_ref
        )
        before = x.clone()
        expected = inplace_partial_rotary_mul(
            x.clone(), cos, sin, impl=executor.impl_ref, **kwargs
        )
        actual = executor()
        assert_accuracy(
            actual,
            expected,
            {"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
        )
        torch.testing.assert_close(
            x[..., : partial_slice[0]], before[..., : partial_slice[0]], atol=0, rtol=0
        )
        executor.accuracy_checked = True
    # Repeated profiling rotates only the DUT buffer; the reference is never run.
    check_performance(
        executor, enable_perf=shape in perf_shape and partial_slice == [448, 512]
    )
