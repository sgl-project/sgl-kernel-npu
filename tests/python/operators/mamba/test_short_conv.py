"""PLE convolution: [B,10240,9+W] -> [B,W,10240], W=1..4, dilation=3.

Reference: native NPU F.conv1d outside capture. BF16 atol=rtol=.02;
FP16 atol=rtol=.003. State writes and SiLU belong to framework tests.
"""

import pytest
import torch
from sgl_kernel_npu.operators.mamba import short_conv
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation != Generation.ARCH22,
    reason="These model operator contracts are validated only on arch22",
)

STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.02),
    "torch.float16": AccuracyStandard(atol=0.003, rtol=0.003),
}
perf_shape = [(1, 1), (8, 4), (32, 4)]


def generate_data(batch, width, dtype):
    torch.manual_seed(42)
    return (
        torch.randn(batch, 10240, 9 + width, dtype=dtype),
        torch.randn(10240, 1, 4, dtype=dtype),
        3,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "batch",
    [0, 1, 3, 8, 32, 65],
    ids=[
        "empty",
        "single-request",
        "batch-tail",
        "decode-batch",
        "large-batch",
        "multi-tile-tail",
    ],
)
@pytest.mark.parametrize(
    "width,single",
    [(1, True), (1, False), (2, False), (3, False), (4, False)],
    ids=["decode-flat-output", "decode-token-output", "verify2", "verify3", "verify4"],
)
def test_short_conv_accuracy(batch, width, single, dtype):
    # For W=4, 13 input positions produce four output tokens.
    args = generate_data(batch, width, dtype)
    saved = [x.clone() for x in args[:2]]
    executor = check_accuracy(
        short_conv, args=args, kwargs={"single_token": single}, standard=STANDARD
    )
    out = executor()
    assert out.shape == ((batch, 10240) if single else (batch, width, 10240))
    assert out.is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=(batch, width) in perf_shape and not single)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "batch,width",
    [(1, 1), (3, 2), (8, 3), (32, 4)],
    ids=["single-decode", "small-verify2", "batch-verify3", "large-verify4"],
)
def test_short_conv_graph_updates(batch, width, dtype):
    args = generate_data(batch, width, dtype)
    _check_graph_updates(
        short_conv,
        args,
        [lambda: args[0].mul_(-0.7), lambda: args[1].mul_(0.3)],
        standard=STANDARD,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "bad", ["dtype", "channels", "dilation", "width", "layout", "single"]
)
def test_short_conv_rejects_metadata(bad):
    x, weight, dilation = generate_data(1, 4, torch.bfloat16)
    kwargs = {}
    if bad == "dtype":
        weight = weight.float()
    elif bad == "channels":
        x = x[:, :-1].contiguous()
    elif bad == "dilation":
        dilation = 1
    elif bad == "width":
        x = x[:, :, :9].contiguous()
    elif bad == "layout":
        x = torch.stack((x, x), -1)[..., 0]
    elif bad == "single":
        kwargs["single_token"] = True
    with pytest.raises(ValueError):
        short_conv(x, weight, dilation, **kwargs)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "width", [1, 2, 3, 4], ids=["decode", "verify2", "verify3", "verify4"]
)
def test_short_conv_storage_offsets_and_padding(width):
    x, weight, dilation = generate_data(3, width, torch.bfloat16)
    backing = torch.full((x.numel() + 3,), 7, dtype=x.dtype)
    x = backing[3:].view_as(x).copy_(x)
    x[1, :, 9:] = 0
    x[2] = 0
    before = backing.clone()
    check_accuracy(short_conv, args=(x, weight, dilation), standard=STANDARD)
    torch.testing.assert_close(backing, before, atol=0, rtol=0)


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
