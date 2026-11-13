import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(4096, 3072), (128, 6144), (128, 1536)]
quant_standard = {
    "torch.int8": AccuracyStandard(1, 0),
    "torch.float32": AccuracyStandard(0, 5e-3),
}


def generate_data(shape, dtype, full):
    torch.manual_seed(42)
    counts = [0, 32, 0, 0, 10, 0, 0, 0, 100, 0, 0, 5, 5, 5, 0, 0]
    if full:
        counts = [shape[0]] + [0] * 15
    return torch.randn(shape, dtype=dtype), torch.tensor(counts, dtype=torch.int64)


@pytest.mark.parametrize(
    "name,shape,need_quant,do_limit",
    [
        pytest.param("situ", (1, 6144), False, True, id="kimi-k3.1-pro-moe-decode"),
        pytest.param(
            "situ", (128, 6144), True, True, id="kimi-k3.1-pro-moe-int8-prefill"
        ),
        pytest.param(
            "swiglu_quant", (1, 1536), False, False, id="qwen3-30b-a3b-moe-decode"
        ),
        pytest.param(
            "swiglu_quant",
            (128, 1536),
            True,
            False,
            id="qwen3-30b-a3b-moe-int8-prefill",
        ),
    ],
)
def test_grouped_activation(name, shape, need_quant, do_limit):
    from sgl_kernel_npu.operators import activation

    operator = getattr(activation, name)
    x, counts = generate_data(shape, torch.bfloat16, full=True)
    kwargs = {"need_quant": need_quant}
    kwargs.update(
        {"linear_beta": 25.0 if do_limit else None}
        if name == "situ"
        else {"do_limit": do_limit}
    )
    if need_quant or name == "situ":
        executor = check_accuracy(
            operator, args=(x, counts, 1), kwargs=kwargs, standard=quant_standard
        )
    else:
        executor = make_executor(operator, args=(x, counts, 1), kwargs=kwargs)
        if not get_skip_acc_check():
            require_distinct_impls(operator, executor.impl, executor.impl_ref)
            actual, _ = executor()
            expected, _ = operator(x, counts, 1, **kwargs, impl=executor.impl_ref)
            assert_accuracy(actual, expected)
            executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("do_limit", [False, True])
def test_swiglu_quant_partial_buffer(do_limit):
    from sgl_kernel_npu.operators.activation import swiglu_quant

    x, counts = generate_data((4096, 3072), torch.bfloat16, full=False)
    executor = make_executor(
        swiglu_quant, args=(x, counts, 1), kwargs={"do_limit": do_limit}
    )
    if not get_skip_acc_check():
        require_distinct_impls(swiglu_quant, executor.impl, executor.impl_ref)
        actual, scale = executor()
        expected, scale_ref = swiglu_quant(
            x, counts, 1, do_limit=do_limit, impl=executor.impl_ref
        )
        rows = int(counts.sum())
        assert_accuracy(
            (actual[:rows], scale[:rows]),
            (expected[:rows], scale_ref[:rows]),
            standard=quant_standard,
        )
        assert (actual[:rows].float() - expected[:rows].float()).abs().mean() < 2e-2
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=x.shape in perf_shape)
