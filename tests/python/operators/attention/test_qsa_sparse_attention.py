"""BF16 sparse GQA, D256, physical slots; output has Q shape and is contiguous.

Independent selected-key CPU FP64 attention, rounded directly to BF16.
atol=rtol=.02; all-padding rows must be exactly zero. No input may be modified.
Qwen3.8-Flash-Next has Q24/KV2/D256. SGLang uses attention TP = TP/attention DP
(CP=1 here): TP1/2/4/8 gives local Q/KV=24/2,12/1,6/1,3/1. Thus TP8 with
attention DP2/4/8 shares the TP4/2/1 shapes. EP/PP do not split these heads.
Decode uses local B; verify uses B*W4 rows. Ordinary prefill reaches this op,
but qwen_sparse_attn_backend.py sends CP prefill to a different attention path.
Inputs are synthetic; prefill slots obey each query's causal boundary.
"""

import pytest
import torch
from sgl_kernel_npu.operators.attention import qsa_sparse_attention
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

STANDARD = {"torch.bfloat16": AccuracyStandard(atol=0.02, rtol=0.02)}
perf_shape = [(1, 3, 1, 2051), (8, 3, 1, 2051), (32, 3, 1, 2051)]


def generate_data(rows=4, heads=3, kv_heads=1, width=2051, gaps=False, mode="masked"):
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
    pool_rows = (
        rows + 1
        if mode == "prefill"
        else (width + 1 if mode in ("full", "verify") else 72)
    )
    k, v = (
        rand((pool_rows + 1, kv_heads, 256))[1:],
        rand((pool_rows + 1, kv_heads, 256))[1:],
    )
    shape = (rows * 2 + 1, width + 7) if gaps else (rows, width)
    slots = torch.full(shape, -1, device="npu", dtype=torch.int32)
    if gaps:
        slots = slots[1::2, :width]
    if rows:
        if mode == "prefill":
            # Keep up to 512 complete blocks plus the query's 0..3 tail tokens.
            lengths = torch.arange(1, rows + 1, device="npu", dtype=torch.int32)[
                :, None
            ]
            columns = torch.arange(width, device="npu", dtype=torch.int32)[None, :]
            selected = (lengths // 4).clamp(max=512) * 4
            tail = lengths % 4
            slots.copy_(
                torch.where(
                    columns < selected,
                    columns + 1,
                    torch.where(
                        columns < selected + tail,
                        lengths // 4 * 4 + columns - selected + 1,
                        -1,
                    ),
                )
            )
        elif mode in ("full", "verify"):
            slots.copy_(torch.arange(1, width + 1, device="npu", dtype=torch.int32))
            if mode == "verify":
                # Four consecutive queries have tail lengths 1, 2, 3, 0.
                valid = 2048 + (torch.arange(rows, device="npu") + 1) % 4
                slots.masked_fill_(
                    torch.arange(width, device="npu")[None, :] >= valid[:, None], -1
                )
        elif mode in ("short", "masked"):
            slots[:, :33] = torch.randint(
                1, 72, (rows, 33), generator=g, dtype=torch.int32, device="cpu"
            ).to("npu")
            if mode == "masked":
                slots[0] = -1
        elif mode != "empty":
            raise ValueError(mode)
    k[0], v[0] = float("nan"), float("inf")
    return q, k, v, slots


@pytest.mark.parametrize(
    "rows,attn_tp,width,mode",
    [
        pytest.param(
            1, 8, 2051, "full", id="qwen3-8-flash-next-attn-tp8-decode-b1-q3-kv1-s2051"
        ),
        pytest.param(
            8, 8, 2051, "full", id="qwen3-8-flash-next-attn-tp8-decode-b8-q3-kv1-s2051"
        ),
        pytest.param(
            4,
            8,
            2051,
            "verify",
            id="qwen3-8-flash-next-attn-tp8-verify-b1-w4-t4-q3-kv1",
        ),
        pytest.param(
            32,
            8,
            2051,
            "verify",
            id="qwen3-8-flash-next-attn-tp8-verify-b8-w4-t32-q3-kv1",
        ),
        pytest.param(
            8,
            8,
            2055,
            "full",
            id="qwen3-8-flash-next-attn-tp8-draft-b8-shared-tail-s2055",
        ),
        pytest.param(
            4096,
            8,
            2051,
            "prefill",
            id="qwen3-8-flash-next-attn-tp8-prefill-t4096-q3-kv1-causal",
        ),
        pytest.param(
            1, 1, 2051, "full", id="qwen3-8-flash-next-attn-tp1-decode-b1-q24-kv2"
        ),
        pytest.param(
            4, 2, 2051, "verify", id="qwen3-8-flash-next-attn-tp2-verify-b1-w4-q12-kv1"
        ),
        pytest.param(
            3,
            4,
            2051,
            "full",
            id="qwen3-8-flash-next-tp8-attn-dp2-attn-tp4-local-b3-q6-kv1",
        ),
        pytest.param(0, 8, 2051, "empty", id="edge-empty-queries"),
        pytest.param(1, 8, 2051, "empty", id="edge-all-padding"),
        pytest.param(33, 8, 2051, "masked", id="edge-strided-row-tail-r33"),
    ],
)
def test_qsa_sparse_attention_accuracy(rows, attn_tp, width, mode):
    # A row with only -1 slots must return zeros; -1 is padding.
    heads, kv_heads = 24 // attn_tp, max(1, 2 // attn_tp)
    args = generate_data(rows, heads, kv_heads, width, gaps=mode == "masked", mode=mode)
    saved = [x.clone() for x in args]
    executor = check_accuracy(qsa_sparse_attention, args=args, standard=STANDARD)
    out = executor()
    assert out.is_contiguous()
    if rows and mode in ("empty", "masked"):
        assert torch.equal(out[0], torch.zeros_like(out[0]))
    _assert_unchanged(args, saved)
    check_performance(
        executor,
        enable_perf=(rows, heads, kv_heads, width) in perf_shape
        and mode in ("full", "verify"),
    )


@pytest.mark.parametrize(
    "rows,width",
    [
        pytest.param(8, 2051, id="qwen3-8-flash-next-tp8-decode-r8-s2051"),
        pytest.param(32, 2051, id="qwen3-8-flash-next-tp8-mtp-r32-s2051"),
    ],
)
def test_qsa_sparse_attention_graph_updates(rows, width):
    args = generate_data(rows, width=width, mode="short")

    def update_slots():
        args[1][0] = 0.5
        args[2][0] = 0.5
        args[3].fill_(-1)
        args[3][1:, :17] = 0

    _check_graph_updates(
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


@pytest.mark.extra_checks
@pytest.mark.parametrize("scale", [None, 0, 0.3, -0.1])
def test_qsa_sparse_attention_scale(scale):
    check_accuracy(
        qsa_sparse_attention,
        args=generate_data(),
        kwargs={"softmax_scale": scale},
        standard=STANDARD,
    )


@pytest.mark.extra_checks
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


@pytest.mark.parametrize(
    "rows,width",
    [
        pytest.param(32, 1025, id="edge-short-scan-r32-s1025"),
        pytest.param(33, 2051, id="edge-row-tail-r33-s2051"),
        pytest.param(32, 2054, id="edge-shared-tail-r32-s2054"),
        pytest.param(33, 2055, id="edge-shared-tail-r33-s2055"),
        pytest.param(33, 8195, id="edge-long-scan-r33-s8195"),
    ],
)
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

    _check_graph_updates(
        qsa_sparse_attention,
        args,
        [
            lambda last=last: place(last)
            for last in [width - 1, 255, 256, 511, 512, 0, None]
        ],
        standard=STANDARD,
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("rows", [4, 33], ids=["split-r4", "direct-r33"])
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

    _check_graph_updates(
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


@pytest.mark.extra_checks
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
