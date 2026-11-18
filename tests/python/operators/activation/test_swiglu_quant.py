import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(4096, 3072), (128, 1536)]
quant_standard = {
    "torch.int8": AccuracyStandard(1, 0),
    "torch.float32": AccuracyStandard(0, 5e-3),
}


def generate_data(shape, full):
    torch.manual_seed(42)
    counts = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
    if full:
        counts = [shape[0]] + [0] * 15
    return torch.randn(shape, dtype=torch.bfloat16), torch.tensor(
        counts, dtype=torch.int64
    )


def _check_swiglu_int8_accuracy(operator, x, counts, do_limit):
    args = (x, counts, 1)
    executor = make_executor(operator, args=args, kwargs={"do_limit": do_limit})
    if get_skip_acc_check():
        require_test_impl(operator, executor.impl)
        return executor

    require_distinct_impls(operator, executor.impl, executor.impl_ref)
    actual, scale = executor()
    expected, scale_ref = operator(*args, do_limit=do_limit, impl=executor.impl_ref)
    rows = int(counts.sum())
    actual, expected = actual[:rows], expected[:rows]
    assert_accuracy(
        (actual, scale[:rows]), (expected, scale_ref[:rows]), standard=quant_standard
    )
    mean_error = (actual.float() - expected.float()).abs().mean()
    assert mean_error < 2e-2, f"Mean absolute INT8 error: {mean_error.item()}"
    executor.accuracy_checked = True
    return executor


@pytest.mark.parametrize(
    "shape,need_quant",
    [
        pytest.param((1, 1536), False, id="qwen3-30b-a3b-moe-decode"),
        pytest.param((128, 1536), True, id="qwen3-30b-a3b-moe-int8-prefill"),
    ],
)
def test_swiglu_quant(shape, need_quant):
    from sgl_kernel_npu.operators.activation import swiglu_quant

    x, counts = generate_data(shape, full=True)
    if need_quant:
        executor = _check_swiglu_int8_accuracy(swiglu_quant, x, counts, do_limit=False)
    else:
        args = (x, counts, 1)
        kwargs = {"need_quant": False, "do_limit": False}
        executor = make_executor(swiglu_quant, args=args, kwargs=kwargs)
        if not get_skip_acc_check():
            require_distinct_impls(swiglu_quant, executor.impl, executor.impl_ref)
            actual, _ = executor()
            expected, _ = swiglu_quant(*args, **kwargs, impl=executor.impl_ref)
            assert_accuracy(actual, expected)
            executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("do_limit", [False, True])
def test_swiglu_quant_partial_buffer(do_limit):
    from sgl_kernel_npu.operators.activation import swiglu_quant

    x, counts = generate_data((4096, 3072), full=False)
    executor = _check_swiglu_int8_accuracy(swiglu_quant, x, counts, do_limit)
    check_performance(executor, enable_perf=x.shape in perf_shape)
