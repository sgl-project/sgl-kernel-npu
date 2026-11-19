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


@pytest.mark.parametrize("group_list_type,capacity", [(1, 3072), (0, 12288)])
def test_clamp_before_silu_valid_rows_replay(group_list_type, capacity):
    pytest.importorskip("torch_npu")
    from sgl_kernel_npu.operators.activation import swiglu_quant
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation != Generation.ARCH22:
        pytest.skip("A3 model shape coverage")

    torch.manual_seed(23)
    x = torch.full((capacity, 4608), float("nan"), device="npu", dtype=torch.bfloat16)
    values = (torch.randn(48, 4608) * 6).to(x).contiguous()
    values[0].zero_()
    group_list = torch.zeros(17, device="npu", dtype=torch.int64)

    def populate_groups(rows):
        counts = [rows // 2, 0, rows - rows // 2] + [0] * 14
        groups = torch.tensor(counts, dtype=group_list.dtype)
        if group_list_type == 0:
            groups = groups.cumsum(0)
        group_list.copy_(groups)

    def invoke(**kwargs):
        return swiglu_quant(
            x, group_list, group_list_type, limit=10.0, clamp_before_silu=True, **kwargs
        )

    x[:48].copy_(values)
    populate_groups(48)
    invoke()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        quant, scale = invoke()

    def check_rows(rows):
        qref, sref = invoke(impl="torch_native")
        torch.testing.assert_close(scale[:rows], sref[:rows], rtol=2e-5, atol=0)
        error = (quant[:rows].to(torch.int16) - qref[:rows]).abs()
        assert error.max().item() <= 1
        assert error.float().mean().item() < 0.01
        assert scale[0].item() == torch.tensor(1e-30).item()
        assert not quant[0].any().item()

    graph.replay()
    torch.npu.synchronize()
    check_rows(48)
    previous_quant = quant[:48].clone()
    previous_scale = scale[:48].clone()

    populate_groups(0)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(quant[:48], previous_quant, atol=0, rtol=0)
    torch.testing.assert_close(scale[:48], previous_scale, atol=0, rtol=0)

    populate_groups(7)
    x[1:7].mul_(-0.5)
    x[7:48].fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    check_rows(7)
    torch.testing.assert_close(quant[7:48], previous_quant[7:], atol=0, rtol=0)
    torch.testing.assert_close(scale[7:48], previous_scale[7:], atol=0, rtol=0)
