"""Partial RoPE accuracy with independent in-place buffers and NPU profiling."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_skip_acc_check
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(16, 1, 64, 512)]


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 512), id="synthetic-decode"),
        pytest.param((16, 1, 64, 512), id="synthetic-batch"),
    ],
)
@pytest.mark.parametrize("partial_slice", [[448, 512], [0, 512]], ids=["tail", "full"])
def test_partial_rope(shape, partial_slice):
    from sgl_kernel_npu.operators.rope import inplace_partial_rotary_mul

    torch.manual_seed(15)
    x = torch.randn(shape, dtype=torch.bfloat16)
    before = x.clone()
    width = partial_slice[1] - partial_slice[0]
    angles = torch.randn(shape[0], 1, 1, width // 2).repeat_interleave(2, -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    executor = check_accuracy(
        inplace_partial_rotary_mul,
        args=(x, cos, sin),
        args_ref=(x.clone(), cos, sin),
        kwargs=dict(rotary_mode="interleave", partial_slice=partial_slice),
        standard={"torch.bfloat16": AccuracyStandard(atol=0.03, rtol=0.03)},
    )
    if get_skip_acc_check() is not True:
        torch.testing.assert_close(
            x[..., : partial_slice[0]], before[..., : partial_slice[0]], atol=0, rtol=0
        )
    # Repeated profiling rotates only the DUT buffer; the reference is never run.
    check_performance(
        executor, enable_perf=shape in perf_shape and partial_slice == [448, 512]
    )
