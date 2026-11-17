"""Four independent 2560-wide Gemma norm groups in BF16 x[R,10240].

Independent CPU FP64 reference; atol=.005, rtol=.01 for BF16 outputs.
All row counts use the same tolerance, including R4096 and R8193.
Qwen3.8-Flash-Next: H2560 * HC4 stays 10240 under TP, attention DP and EP.
Decode uses local B1/B8; verify B1/B8 * W4 gives R4/R32. Prefill T4096
can be split into T512 row views by attention TP8 (SP). PP keeps this width.
Shapes follow SGLang hyperconnection.py and the gated layer-boundary adapter;
inputs are synthetic, not captured tensors.
"""

import pytest
import torch
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

# Allow BF16 output rounding around FP32/FP64 computation differences.
STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.01)}
perf_shape = [1, 32, 4096]


def generate_data(rows, row_offset=0):
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn(rows + row_offset, 10240, generator=generator, device="cpu").to(
        torch.bfloat16
    )
    weight = (torch.randn(10240, generator=generator, device="cpu") * 0.1).to(
        torch.bfloat16
    )
    # Slice after transfer: SP rank 1 keeps a nonzero offset on the device.
    x = x.to("npu")[row_offset:]
    assert x.is_contiguous() and x.storage_offset() == row_offset * 10240
    return x, weight.to("npu"), 2560, 1e-6


@pytest.mark.parametrize(
    "rows,row_offset",
    [
        pytest.param(1, 0, id="qwen3-8-flash-next-decode-b1-h10240"),
        pytest.param(8, 0, id="qwen3-8-flash-next-decode-b8-h10240"),
        pytest.param(4, 0, id="qwen3-8-flash-next-verify-b1-w4-t4-h10240"),
        pytest.param(32, 0, id="qwen3-8-flash-next-verify-b8-w4-t32-h10240"),
        pytest.param(4096, 0, id="qwen3-8-flash-next-prefill-t4096-h10240"),
        pytest.param(
            512, 512, id="qwen3-8-flash-next-sp8-prefill-local-t512-offset512"
        ),
        pytest.param(0, 0, id="edge-empty"),
        pytest.param(33, 0, id="edge-row-tail-r33"),
        pytest.param(8193, 0, id="edge-large-row-tail-r8193"),
    ],
)
def test_grouped_gemma_rmsnorm_accuracy(rows, row_offset):
    # Each row contains four groups; an odd row count also checks the last tile.
    args = generate_data(rows, row_offset)
    saved = [x.clone() for x in args[:2]]
    executor = check_accuracy(grouped_gemma_rmsnorm, args=args, standard=STANDARD)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_shape)


@pytest.mark.parametrize(
    "rows", [pytest.param(8, id="qwen3-8-flash-next-decode-r8-h10240")]
)
def test_grouped_gemma_rmsnorm_graph_updates(rows):
    args = generate_data(rows)
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
