"""PLE convolution: [B,10240,9+W] -> [B,W,10240], W=1..4, dilation=3.

Reference: native NPU F.conv1d outside capture. BF16 atol=rtol=.02;
FP16 atol=rtol=.003. State writes and SiLU belong to framework tests.
Qwen3.8-Flash-Next uses BF16, decode W1 and four-token MTP verification.
B1/B8 follow the serving setup; inputs are synthetic, not captured tensors.
SGLang qwen4_exp.py uses local batches under attention DP; TP/EP do not
split the 10240 channels. Prefill uses a different convolution path.
"""

import pytest
import torch
from sgl_kernel_npu.operators.mamba import short_conv
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.02),
    "torch.float16": AccuracyStandard(atol=0.003, rtol=0.003),
}
perf_shape = [(1, 1, True), (8, 1, True), (8, 4, False)]


def generate_data(batch, width, dtype):
    torch.manual_seed(42)
    return (
        torch.randn(batch, 10240, 9 + width, dtype=dtype),
        torch.randn(10240, 1, 4, dtype=dtype),
        3,
    )


@pytest.mark.parametrize(
    "batch,width,single,dtype",
    [
        pytest.param(
            1, 1, True, torch.bfloat16, id="qwen3-8-flash-next-decode-b1-w1-bf16"
        ),
        pytest.param(
            8, 1, True, torch.bfloat16, id="qwen3-8-flash-next-decode-b8-w1-bf16"
        ),
        pytest.param(
            1, 4, False, torch.bfloat16, id="qwen3-8-flash-next-mtp-b1-w4-bf16"
        ),
        pytest.param(
            8, 4, False, torch.bfloat16, id="qwen3-8-flash-next-mtp-b8-w4-bf16"
        ),
        pytest.param(
            3,
            1,
            True,
            torch.bfloat16,
            id="qwen3-8-flash-next-attn-dp-uneven-local-b3-w1-bf16",
        ),
        pytest.param(0, 1, True, torch.bfloat16, id="edge-empty-flat-output"),
        pytest.param(0, 4, False, torch.bfloat16, id="edge-empty-token-output"),
        pytest.param(3, 1, False, torch.bfloat16, id="edge-w1-token-output"),
        pytest.param(3, 2, False, torch.bfloat16, id="edge-width2-bf16"),
        pytest.param(3, 3, False, torch.bfloat16, id="edge-width3-bf16"),
        pytest.param(65, 4, False, torch.bfloat16, id="edge-batch-tail-b65"),
        pytest.param(1, 1, True, torch.float16, id="edge-width1-fp16"),
        pytest.param(3, 2, False, torch.float16, id="edge-width2-fp16"),
        pytest.param(3, 3, False, torch.float16, id="edge-width3-fp16"),
        pytest.param(8, 4, False, torch.float16, id="edge-width4-fp16"),
    ],
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
    check_performance(
        executor,
        enable_perf=(batch, width, single) in perf_shape and dtype == torch.bfloat16,
    )


@pytest.mark.parametrize(
    "batch,width,single,dtype",
    [
        pytest.param(
            1, 1, True, torch.bfloat16, id="qwen3-8-flash-next-decode-b1-w1-bf16"
        ),
        pytest.param(
            8, 1, True, torch.bfloat16, id="qwen3-8-flash-next-decode-b8-w1-bf16"
        ),
        pytest.param(
            8, 4, False, torch.bfloat16, id="qwen3-8-flash-next-mtp-b8-w4-bf16"
        ),
        pytest.param(
            1, 4, False, torch.bfloat16, id="qwen3-8-flash-next-verify-b1-w4-bf16"
        ),
    ],
)
def test_short_conv_graph_updates(batch, width, single, dtype):
    args = generate_data(batch, width, dtype)
    _check_graph_updates(
        short_conv,
        args,
        [lambda: args[0].mul_(-0.7), lambda: args[1].mul_(0.3)],
        standard=STANDARD,
        kwargs={"single_token": single},
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
