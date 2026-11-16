"""BF16 Q[R,4,128] -> FP32 packed/paged scores, including exact -inf masks.

Independent per-row/head CPU FP64 reference; atol=rtol=2e-5.
Indexer heads stay four for every attention TP degree. Pages contain 16 keys.
Qwen3.8-Flash-Next: TP/attention DP/EP do not shard the replicated indexer.
SGLang qsa_indexer.py caps prefill scores at 128 MiB: M65536 gives T512,
two packed long requests give M131072/T256. CP8 of T4096 has T512 local Q
and all 1024 compressed keys. This tests its local contract, not CP transport.
Decode graph capacity is 262144 / 4 = 65536, even for short live sequences.
Verify B1/B8 * W4 gives T4/T32. All tensor values are synthetic.
"""

import pytest
import torch
from sgl_kernel_npu.operators.indexer import qsa_mqa_packed, qsa_mqa_paged
from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

pytestmark = pytest.mark.skipif(
    get_hardware().generation not in {Generation.ARCH22, Generation.ARCH35},
    reason="These model operators target arch22 and arch35",
)

STANDARD = {"torch.float32": AccuracyStandard(atol=2e-5, rtol=2e-5)}
perf_shape = [
    ("packed", 4096, 1024),
    ("packed", 512, 65536),
    ("paged", 1, 65536),
    ("paged", 8, 65536),
    ("paged", 32, 65536),
]


def generate_data(kind, rows, width, scenario="edge"):
    torch.manual_seed(73)
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16)
    lengths = (
        torch.linspace(0, width, rows).to(torch.int32)
        if rows
        else torch.empty(0, dtype=torch.int32)
    )
    if rows == 1:
        lengths.fill_(width)
    if kind == "packed":
        k = torch.randn(width, 1, 128, dtype=torch.bfloat16)
        starts = torch.zeros(rows, dtype=torch.int32)
        if rows > 1:
            starts[rows // 2 :] = width // 2
            lengths[rows // 2 :] = width
        if scenario != "edge":
            starts.zero_()
            positions = torch.arange(rows, dtype=torch.int32)
            if scenario == "prefix":
                positions += width * 4 - rows
            elif scenario == "packed_prefix":
                # Two requests share the score buffer, not their key ranges.
                starts[rows // 2 :] = width // 2
                positions = positions % (rows // 2) + width // 2 * 4 - rows // 2
            elif scenario == "cp8":
                # Zigzag rank 1 gets chunks 1 and 14 of a 4096-token prefill.
                positions = torch.cat(
                    (torch.arange(256, 512), torch.arange(3584, 3840))
                ).int()
            lengths = starts + (positions + 1) // 4
        return qsa_mqa_packed, (q, k, starts, lengths)
    pages = width // 16
    cache = torch.randn(max(1, pages + 2), 16, 1, 128, dtype=torch.bfloat16)
    table = torch.arange(1, pages + 1, dtype=torch.int32).repeat(rows, 1)
    cache[0] = float("nan")  # An unused physical page must never contaminate scores.
    if scenario == "verify":
        lengths = width - 1 + ((torch.arange(rows, dtype=torch.int32) % 4 + 1) // 4)
    return qsa_mqa_paged, (q, cache, table, lengths, width)


@pytest.mark.parametrize(
    "kind,rows,width,scenario",
    [
        pytest.param(
            "packed", 4096, 1024, "prefill", id="qwen3-8-flash-next-prefill-t4096-m1024"
        ),
        pytest.param(
            "packed", 512, 65536, "prefix", id="qwen3-8-flash-next-prefix-t512-m65536"
        ),
        pytest.param(
            "packed",
            256,
            131072,
            "packed_prefix",
            id="qwen3-8-flash-next-packed-b2-prefix-t256-m131072",
        ),
        pytest.param(
            "packed",
            512,
            1024,
            "cp8",
            id="qwen3-8-flash-next-cp8-prefill-local-t512-m1024",
        ),
        pytest.param(
            "paged", 1, 65536, "decode", id="qwen3-8-flash-next-decode-b1-m65536"
        ),
        pytest.param(
            "paged", 8, 65536, "decode", id="qwen3-8-flash-next-decode-b8-m65536"
        ),
        pytest.param(
            "paged", 4, 65536, "verify", id="qwen3-8-flash-next-verify-b1-w4-t4-m65536"
        ),
        pytest.param(
            "paged",
            32,
            65536,
            "verify",
            id="qwen3-8-flash-next-verify-b8-w4-t32-m65536",
        ),
        pytest.param("packed", 0, 16, "edge", id="edge-packed-empty-queries"),
        pytest.param("paged", 0, 16, "edge", id="edge-paged-empty-queries"),
        pytest.param("packed", 3, 0, "edge", id="edge-packed-empty-keys"),
        pytest.param("paged", 3, 0, "edge", id="edge-paged-empty-keys"),
        pytest.param("packed", 33, 256, "edge", id="edge-packed-row-tail-r33"),
        pytest.param("paged", 33, 256, "edge", id="edge-paged-row-tail-r33"),
    ],
)
def test_qsa_mqa_accuracy(kind, rows, width, scenario):
    # Each query head contributes max(q @ k, 0) / sqrt(128) to the score.
    operator, args = generate_data(kind, rows, width, scenario)
    saved = [x.clone() for x in args if isinstance(x, torch.Tensor)]
    executor = check_accuracy(operator, args=args, standard=STANDARD)
    assert executor().is_contiguous()
    _assert_unchanged(args, saved)
    check_performance(executor, enable_perf=(kind, rows, width) in perf_shape)


@pytest.mark.parametrize(
    "kind,rows,width",
    [
        pytest.param("packed", 8, 1040, id="edge-packed-graph-r8-k1040"),
        pytest.param("paged", 8, 65536, id="qwen3-8-flash-next-decode-b8-m65536"),
        pytest.param(
            "paged", 32, 65536, id="qwen3-8-flash-next-verify-b8-w4-t32-m65536"
        ),
    ],
)
def test_qsa_mqa_graph_updates(kind, rows, width):
    operator, args = generate_data(kind, rows, width)
    updates = [
        lambda: args[0].mul_(-0.5),
        lambda: args[1].mul_(0.25),
        lambda: args[3].zero_(),
    ]
    if kind == "packed":
        updates[-1] = lambda: (args[2].zero_(), args[3].fill_(17))
    else:
        args[3].fill_(1024)  # Live length changes; graph capacity stays 65536.
        updates.insert(2, lambda: args[3].fill_(width))
        updates.insert(2, lambda: args[2].copy_(args[2].flip(1)))
    _check_graph_updates(operator, args, updates, standard=STANDARD)


@pytest.mark.extra_checks
@pytest.mark.parametrize("kind", ["packed", "paged"])
@pytest.mark.parametrize("bad", ["dtype", "heads", "metadata_dtype", "layout"])
def test_qsa_mqa_rejects_metadata(kind, bad):
    operator, values = generate_data(kind, 2, 32)
    args = list(values)
    if bad == "dtype":
        args[0] = args[0].float()
    elif bad == "heads":
        args[0] = args[0][:, :2].contiguous()
    elif bad == "metadata_dtype":
        args[3] = args[3].long()
    else:
        args[0] = torch.stack((args[0], args[0]), -1)[..., 0]
    with pytest.raises(ValueError):
        operator(*args)


@pytest.mark.extra_checks
@pytest.mark.parametrize("kind", ["packed", "paged"])
def test_qsa_mqa_normalized_cancelling_heads(kind):
    operator, inputs = generate_data(kind, 33, 1040)
    args = list(inputs)
    for index in [0, 1]:
        x = args[index].float()
        args[index] = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
    args[0][:, 1] = -args[0][:, 0]
    args[0][:, 3] = -args[0][:, 2]
    check_accuracy(operator, args=args, standard=STANDARD)


@pytest.mark.extra_checks
def test_qsa_mqa_invalid_unused_pages():
    q = torch.zeros(6, 4, 128, dtype=torch.bfloat16)
    cache = torch.ones(4, 16, 1, 128, dtype=q.dtype)
    cache[0] = float("nan")
    lengths = torch.tensor([0, 1, 15, 16, 17, 33], dtype=torch.int32)
    table = torch.tensor([[1, 2, 3, 0]] * 6, dtype=torch.int32)
    table.masked_fill_(torch.arange(4)[None, :] * 16 >= lengths[:, None], 123456)
    check_accuracy(
        qsa_mqa_paged, args=(q, cache, table, lengths, 64), standard=STANDARD
    )


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
