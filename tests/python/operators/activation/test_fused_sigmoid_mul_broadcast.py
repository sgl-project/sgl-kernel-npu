"""Test row-wise gating used by SGLang's shared experts.

The NPU branch in qwen2_moe.py:_forward_shared_experts passes an output
of shape [tokens, hidden_size] and a gate of shape [tokens, 1].
Qwen/Qwen2-57B-A14B-Instruct has hidden_size=3584:
https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct/blob/main/config.json
The 1/32-row model cases represent single-token decode and a small prefill
or decode batch. They are constructed scenarios, not captured model calls.
The performance shapes come from the legacy tests and remain synthetic.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_skip_acc_check, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

STANDARD = {
    "torch.float32": AccuracyStandard(atol=1e-5, rtol=1e-5),
    "torch.float16": AccuracyStandard(atol=1e-3, rtol=1e-3),
    "torch.bfloat16": AccuracyStandard(atol=1.6e-2, rtol=1.6e-2),
}

# These sizes come from the legacy kernel test and are also used for profiling.
PERF_SHAPES = [(1, 4096), (32, 4096), (128, 5120)]
ACCURACY_CASES = [
    pytest.param(shape, True, False, id=f"{shape[0]}x{shape[1]}-column-gate")
    for shape in PERF_SHAPES
] + [
    pytest.param((1, 3584), True, False, id="qwen2-57b-a14b-single-token"),
    pytest.param((32, 3584), True, False, id="qwen2-57b-a14b-32-tokens"),
    pytest.param((1, 1), False, False, id="minimum-vector-gate"),
    # More rows than vector cores, a masked column tail, and two strided inputs.
    pytest.param((257, 33), False, True, id="strided-row-loop-and-tail"),
]


def generate_data(shape, column_gate, strided, dtype):
    torch.npu.manual_seed(42)
    rows, width = shape
    if strided:
        x = torch.randn((width, rows), dtype=dtype).T
        gate = torch.randn(rows * 2, dtype=dtype)[::2]
    else:
        x = torch.randn(shape, dtype=dtype)
        gate = torch.randn(rows, dtype=dtype)
    if column_gate:
        gate = gate[:, None]
    return x, gate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,column_gate,strided", ACCURACY_CASES)
def test_fused_sigmoid_mul_broadcast(shape, column_gate, strided, dtype):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

    x, gate = generate_data(shape, column_gate, strided, dtype)
    executor = check_accuracy(
        fused_sigmoid_mul_broadcast, args=(x, gate), standard=STANDARD
    )
    check_performance(
        executor, enable_perf=shape in PERF_SHAPES and dtype == torch.bfloat16
    )


def test_gate_broadcasts_per_row(impl_pair):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

    if get_skip_acc_check():
        pytest.skip("Accuracy checks are disabled")
    impl = impl_pair[0]
    require_test_impl(fused_sigmoid_mul_broadcast, impl)
    x = torch.ones((3, 3), dtype=torch.float32)
    gate = torch.tensor([0.0, 20.0, -20.0], dtype=x.dtype)
    # Square input would also accept a gate broadcast across columns.
    expected = torch.tensor(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=x.dtype
    )
    result = fused_sigmoid_mul_broadcast(x, gate, impl=impl)
    assert_accuracy(result, expected, STANDARD)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "x_shape,gate_shape,message",
    [
        ((3,), (3,), "x must be 2-D"),
        ((3, 5), (2,), "gate must be"),
        ((3, 5), (1, 3), "2-D gate must have shape"),
    ],
    ids=["x-rank", "gate-length", "gate-shape"],
)
def test_invalid_inputs(x_shape, gate_shape, message):
    from sgl_kernel_npu.operators.activation import fused_sigmoid_mul_broadcast

    x = torch.empty(x_shape)
    gate = torch.empty(gate_shape)
    with pytest.raises(ValueError, match=message):
        fused_sigmoid_mul_broadcast(x, gate)
