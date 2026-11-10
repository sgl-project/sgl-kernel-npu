"""Four independent 2560-wide Gemma norm groups in BF16 x[R,10240].

Independent CPU FP64 reference; atol=.005, rtol=.01 for BF16 outputs.
All row counts use the same tolerance, including R4096 and R8193.
"""

import pytest
import torch
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation != Generation.ARCH22,
    reason="These model operator contracts are validated only on arch22",
)

# Allow BF16 output rounding around FP32/FP64 computation differences.
STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.01)}
perf_shape = [1, 32, 4096]


def generate_data(rows):
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn(rows, 10240, generator=generator, device="cpu").to(torch.bfloat16)
    weight = (torch.randn(10240, generator=generator, device="cpu") * 0.1).to(
        torch.bfloat16
    )
    return x.to("npu"), weight.to("npu"), 2560, 1e-6


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(0, id="empty"),
        pytest.param(1, id="single-token"),
        pytest.param(32, id="decode-batch"),
        pytest.param(33, id="decode-row-tail"),
        pytest.param(4096, id="prefill"),
        pytest.param(8193, id="large-prefill-row-tail"),
    ],
)
def test_grouped_gemma_rmsnorm_accuracy(rows):
    # Each row contains four groups; an odd row count also checks the last tile.
    args = generate_data(rows)
    saved = [x.clone() for x in args[:2]]
    executor = check_accuracy(grouped_gemma_rmsnorm, args=args, standard=STANDARD)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_shape)


def test_grouped_gemma_rmsnorm_graph_updates():
    args = generate_data(8)
    _check_graph_updates(
        grouped_gemma_rmsnorm,
        args,
        [lambda: args[0].mul_(-0.5), lambda: args[1].add_(0.125)],
        standard=STANDARD,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "eps",
    [0, -1, float("nan"), float("inf"), True],
    ids=["zero", "negative", "nan", "infinite", "boolean"],
)
def test_grouped_gemma_rmsnorm_rejects_epsilon(eps):
    x, weight, group, _ = generate_data(0)
    with pytest.raises(ValueError):
        grouped_gemma_rmsnorm(x, weight, group, eps)


def _tensors(args):
    return [value for value in args if isinstance(value, torch.Tensor)]


def _assert_unchanged(args, saved):
    for value, before in zip(_tensors(args), saved):
        torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)


def _check_graph_updates(
    operator, args, updates, *, standard=None, kwargs=None, compare=None
):
    """Capture the public default; compare each replay to an independent reference.

    updates mutate contents at stable addresses. References run outside capture.
    compare can supply a semantic comparator for nonunique TopK outputs.
    """
    kwargs = dict(kwargs or {})
    inputs = _tensors(args)
    pointers = [x.data_ptr() for x in inputs]
    for _ in range(3):
        operator(*args, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operator(*args, **kwargs)
    for change in [None, *updates]:
        if change is not None:
            change()
        saved = [x.clone() for x in inputs]
        graph.replay()
        torch.npu.synchronize()
        if compare is None:
            reference = operator(*args, **kwargs, impl="torch_native")
            assert_accuracy(output, reference, standard)
        else:
            compare(output)
        assert [x.data_ptr() for x in inputs] == pointers
        _assert_unchanged(args, saved)
    return graph, output
