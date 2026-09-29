"""Model-contract sparse GQA: FP64 oracle, metadata rejection and NPU graphs.

BF16 acceptance is atol=rtol=0.02, not an exact-rounding guarantee.
"""

import importlib

import pytest
import torch
import torch_npu

impl = importlib.import_module("sgl_kernel_npu.qwen3_8_flash_next.qsa_sparse_attention")
pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU is required"
)


def oracle(q, k, v, slots, scale=None):
    q, k, v = (x.detach().cpu().double() for x in (q, k, v))
    slots = slots.cpu().long()
    out = torch.zeros_like(q)
    group = q.shape[1] // k.shape[1]
    for row in range(q.shape[0]):
        ids = slots[row][slots[row] >= 0]
        if not ids.numel():
            continue
        for head in range(q.shape[1]):
            scores = (k[ids, head // group] @ q[row, head]) * (scale or 1 / 16)
            out[row, head] = scores.softmax(0) @ v[ids, head // group]
    # Direct BF16 lattice rounding in FP64, avoiding an FP32 midpoint.
    _, exponent = torch.frexp(out)
    step = torch.ldexp(torch.ones_like(out), (exponent - 8).clamp(min=-133))
    return (torch.round(out / step) * step).to(torch.bfloat16)


def inputs(rows=4, heads=3, kv_heads=1, width=2051, gaps=False):
    g = torch.Generator().manual_seed(73 + rows + heads)

    def rand(shape):
        return torch.randn(shape, generator=g, dtype=torch.bfloat16).to("npu")

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
            1, 72, (rows, 33), generator=g, dtype=torch.int32
        ).to("npu")
        slots[0] = -1
    k[0], v[0] = float("nan"), float("inf")
    return q, k, v, slots


def check(args, out, scale=None):
    assert out.shape == args[0].shape and out.dtype == torch.bfloat16
    assert out.is_contiguous() and out.device == args[0].device
    torch.testing.assert_close(out.cpu(), oracle(*args, scale), atol=0.02, rtol=0.02)
    empty = (args[3].cpu() < 0).all(1)
    assert torch.equal(out.cpu()[empty], torch.zeros_like(out.cpu()[empty]))


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)])
@pytest.mark.parametrize("width", [1025, 2051, 2054, 2055, 8195])
def test_model_contract(heads, kv_heads, width):
    args = inputs(heads=heads, kv_heads=kv_heads, width=width, gaps=True)
    saved = [x.clone() for x in args]
    check(args, impl.sparse_attention(*args))
    for x, original in zip(args, saved):
        torch.testing.assert_close(x, original, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 31, 32, 33, 128])
def test_split_boundaries(rows):
    args = inputs(rows=rows)
    check(args, impl.sparse_attention(*args))


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)])
def test_graph_updates(heads, kv_heads):
    args = inputs(heads=heads, kv_heads=kv_heads, width=2055, gaps=True)
    for _ in range(2):
        impl.sparse_attention(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = impl.sparse_attention(*args)
    pointers = [x.data_ptr() for x in args]
    for change in (None, 0, 1, 2, 3, "empty"):
        if change in (0, 1, 2):
            args[change].mul_(0.5)
            if change in (1, 2):
                args[change][0] = 0.5
        elif change == 3:
            args[3].fill_(-1)
            args[3][1:, :17] = 0
        elif change == "empty":
            args[3].fill_(-1)
        graph.replay()
        torch.npu.synchronize()
        check(args, out)
        torch.testing.assert_close(out, impl.sparse_attention(*args), atol=0, rtol=0)
        assert pointers == [x.data_ptr() for x in args]


@pytest.mark.parametrize(
    "case",
    [
        "fp16",
        "fp32",
        "d128",
        "heads",
        "int64",
        "width0",
        "width33",
        "width2052",
        "kv_stride",
        "q_stride",
        "slot_stride",
        "cpu",
        "rows",
    ],
)
def test_reject_metadata(case):
    q, k, v, s = inputs()
    if case in ("fp16", "fp32"):
        dtype = torch.float16 if case == "fp16" else torch.float32
        q, k, v = (x.to(dtype) for x in (q, k, v))
    elif case == "d128":
        q, k, v = (x[..., :128] for x in (q, k, v))
    elif case == "heads":
        q = q[:, :2]
    elif case == "int64":
        s = s.long()
    elif case.startswith("width"):
        s = torch.full((4, int(case[5:])), -1, device="npu", dtype=s.dtype)
    elif case == "kv_stride":
        k = k[:1].expand_as(k)
    elif case == "q_stride":
        q = q[:1].expand_as(q)
    elif case == "slot_stride":
        s = s[:1].expand_as(s)
    elif case == "cpu":
        q = q.cpu()
    elif case == "rows":
        s = s[:2]
    assert not impl.can_run_sparse_attention(q, k, v, s)
    with pytest.raises(ValueError):
        impl.sparse_attention(q, k, v, s)


@pytest.mark.parametrize("empty", ["rows", "cache"])
def test_empty(empty):
    q, k, v, s = inputs(rows=0 if empty == "rows" else 4)
    if empty == "cache":
        k, v = k[:0], v[:0]
        s.fill_(-1)
    out = impl.sparse_attention(q, k, v, s)
    assert out.is_contiguous() and torch.equal(out, torch.zeros_like(q))


@pytest.mark.parametrize("scale", [None, 0, 0.3, -0.1])
def test_scale(scale):
    args = inputs()
    check(args, impl.sparse_attention(*args, scale), scale)


@pytest.mark.parametrize("heads,kv_heads", [(3, 1), (6, 1), (12, 1), (24, 2)])
@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_anomaly_and_padding(heads, kv_heads, target, value):
    args = inputs(heads=heads, kv_heads=kv_heads)
    args[target][1] = value
    args[3][1, 0] = 1
    out = impl.sparse_attention(*args)
    assert not torch.isfinite(out[1]).all()
    args[3].fill_(-1)
    out = impl.sparse_attention(*args)
    assert torch.equal(out, torch.zeros_like(out))


def test_real_wide_pool_graph():
    # Real allocation crossing int32 element offsets, not broadcast storage.
    n = 2**23 + 1
    q = torch.ones(2, 3, 256, device="npu", dtype=torch.bfloat16)
    k = torch.empty(n, 1, 256, device="npu", dtype=q.dtype)
    v = torch.empty_like(k)
    k[0] = k[-1] = 0
    v[0], v[-1] = 1, 3
    slots = torch.full((2, 2051), -1, device="npu", dtype=torch.int32)
    slots[1, 0], slots[1, 1] = 0, n - 1
    assert impl.dispatch_info(q, k, v, slots)["offset_bits"] == 64
    for _ in range(2):
        impl.sparse_attention(q, k, v, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = impl.sparse_attention(q, k, v, slots)
    for last in (3, 5):
        v[-1] = last
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros_like(q)
        expected[1] = (1 + last) / 2
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


def _inputs(rows, width, gaps=False):
    generator = torch.Generator().manual_seed(29472 + rows + width + int(gaps))

    def positive(shape):
        return (
            (torch.rand(shape, generator=generator) * 0.25 + 0.125).bfloat16().to("npu")
        )

    if gaps:
        q = positive((2 * rows + 1, 3, 272))[1::2, :, :256]
        slots = torch.full(
            (2 * rows + 1, width + 9), -(2**31), dtype=torch.int32, device="npu"
        )[1::2, 3 : width + 3]
        assert q.stride(0) > 3 * q.stride(1) and q.stride(1) > 256
        assert slots.stride(0) > width and slots.stride(1) == 1
        assert q.storage_offset() > 0 and slots.storage_offset() > 0
    else:
        q = positive((rows, 3, 256))
        slots = torch.full((rows, width), -(2**31), dtype=torch.int32, device="npu")
        assert q.is_contiguous() and slots.is_contiguous()
    k = positive((65, 1, 256))[1:]
    v = torch.randn((65, 1, 256), generator=generator).bfloat16().to("npu")[1:]
    assert k.is_contiguous() and v.is_contiguous()
    assert k.storage_offset() > 0 and v.storage_offset() > 0
    # Row zero remains empty throughout every graph, despite its nonfinite Q.
    q[0] = float("nan")
    return q, k, v, slots


def _assert_direct(impl, args):
    dispatch = impl.dispatch_info(*args)
    assert dispatch["path"] == "grouped_direct", dispatch
    assert dispatch["splits"] == 1 and dispatch["block"] == 32, dispatch
    assert dispatch["offset_bits"] == 32 and dispatch["scratch_bytes"] == 0, dispatch


def _capture(impl, args):
    _assert_direct(impl, args)
    for _ in range(2):
        impl.sparse_attention(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = impl.sparse_attention(*args)
    return graph, output


def _run_pair(impl, args, graph, graph_output, pointers):
    _assert_direct(impl, args)
    before = [x.cpu().clone() for x in args]
    eager = impl.sparse_attention(*args)
    graph.replay()
    torch.npu.synchronize()
    for out in (eager, graph_output):
        assert out.shape == args[0].shape and out.dtype == torch.bfloat16
        assert out.device == args[0].device and out.is_contiguous()
    eager_cpu, graph_cpu = eager.cpu(), graph_output.cpu()
    # Identical inputs must produce identical eager/replay values, including anomalies.
    torch.testing.assert_close(graph_cpu, eager_cpu, atol=0, rtol=0, equal_nan=True)
    assert pointers == [x.data_ptr() for x in args]
    for original, current in zip(before, args):
        torch.testing.assert_close(
            current.cpu(), original, atol=0, rtol=0, equal_nan=True
        )
    return (eager_cpu, graph_cpu), before


def _check_finite(out, expected, slots):
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    empty = (slots < 0).all(1)
    assert torch.equal(out[empty], torch.zeros_like(out[empty]))


@pytest.mark.parametrize("rows", [32, 33])
@pytest.mark.parametrize("target", [0, 1, 2], ids=["q", "k", "v"])
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf")],
    ids=["nan", "posinf", "neginf"],
)
def test_direct_live_anomaly_isolation(rows, target, value):
    args = _inputs(rows, 2051)
    q, k, v, slots = args
    # Unused physical slot zero is nonfinite; padding must never read it into output.
    k[0], v[0] = float("nan"), float("inf")
    for column, physical in ((0, 2), (2, 3), (31, 4), (96, 5)):
        slots[1:, column] = physical
    # Only row one uses physical row one. Other active rows must stay finite.
    slots[1, 0] = slots[1, 31] = 1
    assert (slots[:, 32:96] < 0).all()  # Internal empty tiles before a later live tile.
    assert (slots[:, 97:] < 0).all()  # Empty suffix after the mixed live tiles.
    expected = oracle(*args)
    assert torch.isfinite(expected).all()
    original_target = args[target][1].clone()
    graph, graph_output = _capture(impl, args)
    pointers = [x.data_ptr() for x in args]
    healthy = [0, *range(2, rows)]
    for state in ("finite", "anomaly", "recovered", "empty_anomaly"):
        if state in ("anomaly", "empty_anomaly"):
            args[target][1] = value
        elif state == "recovered":
            args[target][1].copy_(original_target)
        if state == "empty_anomaly":
            slots.fill_(-(2**31))
        outputs, before = _run_pair(impl, args, graph, graph_output, pointers)
        for out in outputs:
            if state == "anomaly":
                # The contract exposes nonfinite results; it does not prescribe NaN/Inf signs.
                assert (~torch.isfinite(out[1])).any(dim=1).all()
                _check_finite(out[healthy], expected[healthy], before[3][healthy])
            elif state == "empty_anomaly":
                assert torch.equal(out, torch.zeros_like(out))
            else:
                _check_finite(out, expected, before[3])


@pytest.mark.parametrize("rows", [32, 33])
@pytest.mark.parametrize("width", [1025, 2051, 2054, 2055, 8195])
@pytest.mark.parametrize("gaps", [False, True], ids=["packed", "gaps_offsets"])
def test_direct_scan_boundaries_widths_and_graph_updates(rows, width, gaps):
    args = _inputs(rows, width, gaps)
    slots = args[3]
    # Capture empty once, then reuse this graph for all live-boundary movements.
    graph, graph_output = _capture(impl, args)
    pointers = [x.data_ptr() for x in args]
    # Zero-based last positions cover both sides of 256/512, the final column,
    # forward/backward moves and the partial final scan block for every width.
    positions = (None, width - 1, 255, 256, 511, 512, 257, 254, width - 2, 0, None)
    for last in positions:
        host_slots = torch.full((rows, width), -(2**31), dtype=torch.int32)
        if last is not None:
            columns = sorted({c for c in (0, 2, 31, last) if c <= last})
            for index, column in enumerate(columns):
                host_slots[1:, column] = (torch.arange(1, rows) * 7 + index * 11) % 64
            host_slots[1, last] = 0  # Physical slot zero is a valid selection.
            if last >= 255:
                assert (host_slots[1:, 32:64] < 0).all()
                assert (host_slots[1:, last] >= 0).all()
            assert (host_slots[:, last + 1 :] < 0).all()
        slots.copy_(host_slots)
        # Full-row independent FP64 oracle, directly rounded to BF16.
        expected = oracle(*args)
        outputs, before = _run_pair(impl, args, graph, graph_output, pointers)
        for out in outputs:
            _check_finite(out, expected, before[3])
