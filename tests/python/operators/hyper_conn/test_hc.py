"""HC4/H2560/L320 BF16 mix and combine, with FP32 production intermediates.

Independent CPU FP64 reference, BF16 output; atol=.005, rtol=.01.
Inputs and learned weights must remain unchanged; graph reads updated weights.
Qwen3.8-Flash-Next: H2560 * HC4 and low rank 320 are replicated under TP/EP.
Attention DP uses local token counts; verify B1/B8 * W4 gives T4/T32.
SGLang's gated layer-boundary adapter slices rows for SP, not hidden width;
T4096 / attention TP8 gives T512. PP also preserves the stream width.
Inputs are synthetic, not captured tensors.
"""

import pytest
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine, hc_mix
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.005, rtol=0.01)}
OPERATORS = {"mix": hc_mix, "combine": hc_combine}
perf_shape = [1, 32, 4096]


def _cpu_inputs(op, rows):
    generator = torch.Generator(device="cpu").manual_seed(73)

    def rand(shape, scale=1.0):
        return (torch.randn(shape, generator=generator, device="cpu") * scale).to(
            torch.bfloat16
        )

    x = rand((rows, 10240))
    if op == "mix":
        return x, rand((320, 10240), 0.02), rand((10240, 320), 0.02), 4, 2560
    return rand((rows, 2560)), x, rand((rows, 10240)), rand((4, 10240), 0.02), 4, 2560


def generate_data(op, rows, row_offset=0):
    args = list(
        x.to("npu") if isinstance(x, torch.Tensor) else x for x in _cpu_inputs(op, rows)
    )
    # SP rank 1's residual is a contiguous row view with a nonzero offset.
    for index in [0] if op == "mix" else [0, 1, 2]:
        x = args[index]
        if row_offset:
            backing = torch.zeros(
                (rows + row_offset, x.shape[1]), dtype=x.dtype, device=x.device
            )
            args[index] = backing[row_offset:].copy_(x)
            assert args[index].storage_offset() == row_offset * x.shape[1]
    return tuple(args)


@pytest.mark.parametrize("op", ["mix", "combine"])
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
        pytest.param(8193, 0, id="edge-chunk-tail-r8193"),
    ],
)
def test_hc_accuracy(op, rows, row_offset):
    # Mix reduces four groups; combine adds a gated block to each residual group.
    args = generate_data(op, rows, row_offset)
    saved = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    executor = check_accuracy(OPERATORS[op], args=args, standard=STANDARD)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=rows in perf_shape)


@pytest.mark.parametrize("op", ["mix", "combine"])
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param(8, id="qwen3-8-flash-next-decode-r8-h10240"),
        pytest.param(32, id="qwen3-8-flash-next-verify-b8-w4-t32-h10240"),
    ],
)
def test_hc_graph_input_and_weight_updates(op, rows):
    args = generate_data(op, rows)
    updates = [
        lambda x=x: x.mul_(-0.5).add_(0.03125)
        for x in args
        if isinstance(x, torch.Tensor)
    ]
    _check_graph_updates(OPERATORS[op], args, updates, standard=STANDARD)


@pytest.mark.extra_checks
@pytest.mark.parametrize("op", ["mix", "combine"])
@pytest.mark.parametrize("bad", ["dtype", "shape", "layout", "hc"])
def test_hc_rejects_metadata(op, bad):
    args = list(generate_data(op, 1))
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "shape":
        args[0] = args[0][:, :-1].contiguous()
    elif bad == "layout":
        args[0] = torch.stack((args[0], args[0]), -1)[..., 0]
    else:
        args[-2] = 5
    with pytest.raises(ValueError):
        OPERATORS[op](*args)


@pytest.mark.extra_checks
def test_hc_combine_cancellation():
    args = list(generate_data("combine", 1))
    block, _, normed, weight = args[:4]
    gate = 2 * torch.sigmoid(normed.cpu().double() @ weight.cpu().double().T / 4)
    args[1] = (
        (-(block.cpu().double()[:, None, :] * gate[:, :, None]))
        .flatten(1)
        .to(device=block.device, dtype=block.dtype)
    )
    check_accuracy(hc_combine, args=args, standard=STANDARD)


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
