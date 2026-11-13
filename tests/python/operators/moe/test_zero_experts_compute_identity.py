import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

perf_shape = [(16, 6144, 8)]


def generate_data(shape, all_identity, dtype):
    rows, hidden, topk = shape
    torch.manual_seed(42)
    indices = torch.randint(0, 4, (rows, topk), dtype=torch.int32) + (
        16 if all_identity else 14
    )
    scales = torch.rand(rows, topk, dtype=torch.float32)
    x = torch.randn(rows, hidden, dtype=dtype)
    return indices, scales, 16, "identity", x, -1


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 6144, 8), id="longcat-flash-decode"),
        pytest.param((16, 6144, 8), id="longcat-flash-prefill"),
        pytest.param((3, 33, 2), id="dimension-tail"),
    ],
)
@pytest.mark.parametrize("all_identity", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_experts(shape, all_identity, dtype):
    from sgl_kernel_npu.operators.moe import zero_experts_compute_identity

    args = generate_data(shape, all_identity, dtype)
    impl, impl_ref = get_test_impls()
    executor = make_executor(zero_experts_compute_identity, args=args)
    if not get_skip_acc_check():
        require_distinct_impls(zero_experts_compute_identity, impl, impl_ref)
        reference = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in args)
        expected = zero_experts_compute_identity(*reference, impl=impl_ref)
        actual = executor()
        assert_accuracy(
            (actual, args[0], args[1]), (expected, reference[0], reference[1])
        )
        executor.accuracy_checked = True
    # Restore routing buffers before each measured invocation; repeated calls otherwise erase the identity contribution.
    initial = generate_data(shape, all_identity, dtype)

    def run():
        args[0].copy_(initial[0])
        args[1].copy_(initial[1])
        return executor()

    run.func = executor.func
    run.impl, run.impl_ref, run.accuracy_checked = (
        executor.impl,
        executor.impl_ref,
        executor.accuracy_checked,
    )
    check_performance(run, enable_perf=shape in perf_shape)
