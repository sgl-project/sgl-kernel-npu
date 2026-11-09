"""BF16 sparse GQA, D256, physical slots; output has Q shape and is contiguous.

Independent selected-key CPU FP64 attention, rounded directly to BF16.
atol=rtol=.02; all-padding rows must be exactly zero. No input may be modified.
"""

import pytest
import torch
from sgl_kernel_npu.operators.attention import qsa_sparse_attention

from tests.python.operators.qsa_test_utils import (
    ARCH22_ONLY,
    assert_unchanged,
    check_graph_updates,
)
from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = ARCH22_ONLY

STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.02)}
perf_shape = [(1, 3, 1), (32, 3, 1)]


def generate_data(rows=4, heads=3, kv_heads=1, width=2051, gaps=False):
    g = torch.Generator(device="cpu").manual_seed(73 + rows + heads)

    def rand(shape):
        return torch.randn(shape, generator=g, dtype=torch.bfloat16, device="cpu").to(
            "npu"
        )

    q = (
        rand((rows * 2 + 1, heads, 512))[1::2, :, :256]
        if gaps
        else rand((rows, heads, 256))
    )
    k, v = rand((73, kv_heads, 256))[1:], rand((73, kv_heads, 256))[1:]
    slots = torch.full((rows * 2 + 1, width + 7), -1, device="npu", dtype=torch.int32)[
        1::2, :width
    ]
    if rows:
        slots[:, :33] = torch.randint(
            1, 72, (rows, 33), generator=g, dtype=torch.int32, device="cpu"
        ).to("npu")
        slots[0] = -1
    k[0], v[0] = float("nan"), float("inf")
    return q, k, v, slots


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)])
@pytest.mark.parametrize("rows", [0, 1, 3, 8, 32, 33])
@pytest.mark.parametrize("width", [2051, 2055])
def test_qsa_sparse_attention_accuracy(rows, heads, kv_heads, width):
    args = generate_data(rows, heads, kv_heads, width, gaps=True)
    saved = [x.clone() for x in args]
    executor = check_accuracy(qsa_sparse_attention, args=args, standard=STANDARD)
    out = executor()
    assert out.is_contiguous()
    if rows:
        assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert_unchanged(args, saved)
    check_performance(
        executor, enable_perf=(rows, heads, kv_heads) in perf_shape and width == 2051
    )


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)])
def test_qsa_sparse_attention_graph_updates(heads, kv_heads):
    args = generate_data(4, heads, kv_heads, 2055, gaps=True)

    def update_slots():
        args[1][0] = 0.5
        args[2][0] = 0.5
        args[3].fill_(-1)
        args[3][1:, :17] = 0

    check_graph_updates(
        qsa_sparse_attention,
        args,
        [
            lambda: args[0].mul_(0.5),
            lambda: args[1].mul_(0.25),
            lambda: args[2].mul_(-0.5),
            update_slots,
            lambda: args[3].fill_(-1),
        ],
        standard=STANDARD,
    )


@pytest.mark.parametrize("scale", [None, 0, 0.3, -0.1])
def test_qsa_sparse_attention_scale(scale):
    check_accuracy(
        qsa_sparse_attention,
        args=generate_data(),
        kwargs={"softmax_scale": scale},
        standard=STANDARD,
    )


@pytest.mark.parametrize("bad", ["dtype", "heads", "slots_dtype", "kv_layout"])
def test_qsa_sparse_attention_rejects_metadata(bad):
    args = list(generate_data())
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "heads":
        args[0] = args[0][:, :2]
    elif bad == "slots_dtype":
        args[3] = args[3].long()
    else:
        args[1] = torch.stack((args[1], args[1]), -1)[..., 0]
    with pytest.raises(ValueError):
        qsa_sparse_attention(*args)


@pytest.mark.parametrize("rows", [32, 33])
@pytest.mark.parametrize("width", [1025, 2051, 2054, 2055, 8195])
def test_qsa_sparse_attention_scan_boundaries(rows, width):
    args = generate_data(rows, width=width, gaps=True)
    q, k, v, slots = args
    k[0], v[0] = 0.25, 0.5
    q[0] = float("nan")
    slots.fill_(-1)

    def place(last):
        slots.fill_(-1)
        if last is not None:
            for column in sorted({0, min(31, last), last}):
                slots[1:, column] = (column + 7) % 72

    check_graph_updates(
        qsa_sparse_attention,
        args,
        [
            lambda last=last: place(last)
            for last in [width - 1, 255, 256, 511, 512, 0, None]
        ],
        standard=STANDARD,
    )


@pytest.mark.parametrize("rows", [4, 33])
@pytest.mark.parametrize("target", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_qsa_sparse_attention_anomaly_isolation(rows, target, value):
    args = generate_data(rows)
    slots = args[3]
    slots.fill_(-1)
    slots[1, 0] = 1
    slots[2:, 0] = 2
    original = args[target][1].clone()
    reference = qsa_sparse_attention(*args, impl="torch_native")
    stage = ["finite"]

    def update(state):
        stage[0] = state
        args[target][1].copy_(original)
        if state in ["anomaly", "empty"]:
            args[target][1] = value
        if state == "empty":
            slots.fill_(-1)

    def compare(out):
        if stage[0] == "empty":
            assert torch.equal(out, torch.zeros_like(out))
        elif stage[0] == "anomaly":
            assert (~out[1].isfinite()).any(dim=1).all()
            healthy = [0, *range(2, rows)]
            torch.testing.assert_close(
                out[healthy], reference[healthy], atol=0.02, rtol=0.02
            )
        else:
            torch.testing.assert_close(out, reference, atol=0.02, rtol=0.02)

    check_graph_updates(
        qsa_sparse_attention,
        args,
        [
            lambda state=state: update(state)
            for state in ["anomaly", "recovered", "empty"]
        ],
        compare=compare,
    )


def test_qsa_sparse_attention_empty_cache():
    q, k, v, slots = generate_data()
    slots.fill_(-1)
    out = qsa_sparse_attention(q, k[:0], v[:0], slots)
    assert out.is_contiguous() and torch.equal(out, torch.zeros_like(q))


def test_qsa_sparse_attention_wide_address_graph():
    # Two real 4-GiB pools cross signed int32 element offsets. Only the two
    # selected keys are initialized; equal logits give an exact mean oracle.
    count = 2**23 + 1
    q = torch.ones(2, 3, 256, dtype=torch.bfloat16)
    k = torch.empty(count, 1, 256, dtype=q.dtype)
    v = torch.empty_like(k)
    k[0] = k[-1] = 0
    v[0], v[-1] = 1, 3
    slots = torch.full((2, 2051), -1, dtype=torch.int32)
    slots[1, 0], slots[1, 1] = 0, count - 1
    for _ in range(3):
        qsa_sparse_attention(q, k, v, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = qsa_sparse_attention(q, k, v, slots)
    for last in [3, 5]:
        v[-1] = last
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros_like(q)
        expected[1] = (1 + last) / 2
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
