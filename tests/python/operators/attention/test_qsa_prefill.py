"""Public QSA prefill operator: selection semantics, TP layouts and lifetimes."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance

# Production-like prefill chunks selected for performance collection. The
# shared and fully independent selections bound the sharing optimization.
perf_shape = [
    (4621, 16, 2),
    (4621, 8, 1),
]
perf_mode = ("shared", "different")

# Sharing-group granularities, interior mismatches and invalid mappings.
SELECTION_MODES = [
    "shared",  # All 16 queries select identical blocks: one shared group.
    "octets",  # Equality only within 8-query halves.
    "quartets",  # Equality only within 4-query groups.
    "pairs",  # Equality only within adjacent pairs.
    "different",  # Every query selects a different block list.
    "interior",  # One interior entry differs: no group may be shared.
    "fragmented",  # Random, duplicate, out-of-range blocks and broken slots.
    "empty",  # No valid base selection; only the causal tails remain.
]


def _native_available():
    """Report whether this build registered the A3-family native helpers."""
    try:
        import sgl_kernel_npu  # noqa: F401

        return hasattr(torch.ops.npu, "qsa_prefill_runs")
    except Exception:
        return False


requires_native = pytest.mark.skipif(
    not _native_available(),
    reason="QSA prefill needs an arch22 A3-family build of the native operator",
)


def generate_data(shape, mode, length, seed=9302026):
    """Create NPU inputs using the device context supplied by conftest.

    shape is (rows, local Q heads, local KV heads). length is the sequence
    length in tokens; the caller keeps base = length - rows so that every row
    attends to its own causal remainder. mode selects the block-sharing
    pattern described in SELECTION_MODES.
    """
    rows, heads, kv_heads = shape
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    capacity = max(6000, length)
    q = torch.randn(rows, heads, 256, dtype=torch.bfloat16)
    k = torch.randn(capacity, kv_heads, 256, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.arange(length, dtype=torch.int32)[None]
    blocks = torch.arange(512, dtype=torch.int32).repeat(rows, 1)
    if mode in ("pairs", "quartets", "octets", "different"):
        step = {"pairs": 2, "quartets": 4, "octets": 8, "different": 1}[mode]
        # Bound the shift so that long prefills keep in-range selections.
        blocks += (torch.arange(rows, dtype=torch.int32) // step % 500)[:, None]
    elif mode == "interior":
        blocks[14::16, 271] += 7
    elif mode == "empty":
        blocks.fill_(-1)
    elif mode == "fragmented":
        blocks = torch.randint(-3, 1500, (rows, 512), dtype=torch.int32)
        blocks[:, 10] = 2147483647  # Out-of-range block index.
        blocks[:, 20] = blocks[:, 19]  # Duplicate selection.
        if rows >= 16:
            blocks[:16] = blocks[0].clone()  # Keep one shared group.
        # randperm has no NPU kernel; build the permuted table on the host.
        table = torch.randperm(capacity, device="cpu").int()[:length][None].npu()
        table[:, ::13] = -1  # Unmapped logical positions.
        table[:, ::31] = capacity + 1  # Physical slots outside the cache.
    elif mode != "shared":
        raise ValueError(f"Unknown selection mode: {mode}")
    request_row = torch.tensor([0], dtype=torch.int64)
    return q, k, v, blocks, table, request_row


@requires_native
@pytest.mark.parametrize("mode", SELECTION_MODES)
@pytest.mark.parametrize(
    "shape",
    [
        # Local head layouts of Whittle-Next and Qwen3.8-Flash-Next prefill.
        pytest.param((4621, 16, 2), id="whittle-tp1-4621x16x2"),
        pytest.param((4621, 8, 1), id="whittle-tp2-4621x8x1"),
        pytest.param((1024, 2, 1), id="whittle-tp8-1024x2x1"),
        pytest.param((1024, 6, 1), id="qwen3.8-flash-next-tp4-1024x6x1"),
        pytest.param((1, 16, 2), id="edge-single-row"),
        pytest.param((17, 8, 1), id="edge-partial-group-17"),
        pytest.param((16, 6, 1), id="edge-exact-group-16"),
        pytest.param((0, 16, 2), id="edge-zero-rows"),
    ],
)
def test_qsa_prefill(shape, mode):
    from sgl_kernel_npu.operators.attention import qsa_prefill

    rows = shape[0]
    length = rows + 8192
    inputs = generate_data(shape, mode, length)
    executor = check_accuracy(qsa_prefill, args=(*inputs, length, length - rows))
    # An accuracy failure raises before performance collection.
    check_performance(executor, enable_perf=shape in perf_shape and mode in perf_mode)


@requires_native
@pytest.mark.parametrize("mode", ["shared", "fragmented", "empty"])
@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="length-0"),  # No visible token at all.
        pytest.param(1, id="length-1"),  # Single token, empty causal tails.
        pytest.param(3, id="length-3"),  # Shorter than one ratio-4 block.
        pytest.param(67, id="length-67"),  # Unaligned tail inside one block.
    ],
)
def test_qsa_prefill_short_sequences(length, mode):
    """Sequences shorter than the selected blocks keep only valid positions."""
    from sgl_kernel_npu.operators.attention import qsa_prefill

    shape = (17, 16, 2)
    inputs = generate_data(shape, mode, length)
    executor = check_accuracy(qsa_prefill, args=(*inputs, length, length - shape[0]))
    check_performance(executor, enable_perf=False)


@requires_native
@pytest.mark.parametrize("scale", [1 / 19, 1 / 21, 1 / 23])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((9, 16, 2), id="16x2"),
        pytest.param((9, 8, 1), id="8x1"),
        pytest.param((9, 6, 1), id="6x1"),
        pytest.param((9, 2, 1), id="2x1"),
    ],
)
def test_qsa_prefill_tiling_after_allocator_reuse(shape, scale):
    """A newly cached native tiling must survive queued allocator reuse.

    Each (head layout, scale) combination allocates a tiling entry on first
    use. Churning small temporaries first leaves pending writes on storage the
    caching allocator hands back, which previously corrupted that entry.
    """
    from sgl_kernel_npu.operators.attention import qsa_prefill

    length = 509
    inputs = generate_data(shape, "shared", length)
    for size in range(16, 1025, 16):
        temporary = torch.zeros(size, dtype=torch.uint8)
        del temporary
    executor = check_accuracy(
        qsa_prefill, args=(*inputs, length, length - shape[0], scale)
    )
    check_performance(executor, enable_perf=False)


@requires_native
@pytest.mark.parametrize("mode", ["shared", "fragmented"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((17, 16, 2), id="16x2"),
        pytest.param((17, 8, 1), id="8x1"),
        pytest.param((17, 6, 1), id="6x1"),
        pytest.param((17, 2, 1), id="2x1"),
    ],
)
def test_qsa_prefill_padded_request_table(shape, mode):
    """A padded table row stride and a nonzero request row must be honored."""
    from sgl_kernel_npu.operators.attention import qsa_prefill

    length = 257
    q, k, v, blocks, table, _ = generate_data(shape, mode, length)
    # Widen the table, then slice it back so the row stride exceeds length.
    table = torch.cat((table, table, table[:, :1]), 1).repeat(2, 1)[:, :length]
    request_row = torch.tensor([1], dtype=torch.int64)
    executor = check_accuracy(
        qsa_prefill,
        args=(q, k, v, blocks, table, request_row, length, length - shape[0]),
    )
    check_performance(executor, enable_perf=False)


@requires_native
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((17, 16, 2), id="16x2"),
        pytest.param((17, 8, 1), id="8x1"),
        pytest.param((17, 6, 1), id="6x1"),
        pytest.param((17, 2, 1), id="2x1"),
    ],
)
def test_qsa_prefill_queued_launches(shape, impl_pair):
    """Queued launches must not read arguments the caller already dropped.

    The native launch owns its tensor arguments until the device executes it.
    This flow is not expressible through check_accuracy: the block, table and
    request-row arguments are temporaries released as soon as each call
    returns, and the results are only compared after one synchronization.

    These shapes occupy few AI cores. Queueing two launches that each occupy
    every AI core without an intervening synchronization currently times out
    the vector cores on CANN 9.1.0 with triton-ascend 3.2.2, so that pattern
    is not exercised here.
    """
    from sgl_kernel_npu.operators.attention import qsa_prefill

    impl = impl_pair[0]
    length = 257
    base = length - shape[0]
    q, k, v, blocks, table, request_row = generate_data(shape, "fragmented", length)
    reference = qsa_prefill(
        q, k, v, blocks, table, request_row, length, base, impl=impl
    ).clone()
    torch.npu.synchronize()
    outputs = [
        qsa_prefill(
            q,
            k,
            v,
            blocks.clone(),
            table.clone(),
            request_row.clone(),
            length,
            base,
            impl=impl,
        )
        for _ in range(20)
    ]
    torch.npu.synchronize()
    for out in outputs:
        assert torch.equal(out, reference)


@requires_native
@pytest.mark.parametrize(
    "tp",
    [
        pytest.param(2, id="whittle-tp2-local-8x1"),
        pytest.param(8, id="whittle-tp8-local-2x1"),
    ],
)
def test_qsa_prefill_shards_reconstruct_global_attention(tp, impl_pair):
    """Concatenated per-rank shards must equal global 16/2 attention.

    Every rank runs sequentially on one NPU, so this checks KV replication and
    output head order rather than a distributed deployment. The global-head
    reference cannot come from a single sharded call, so this test compares a
    manual concatenation instead of using check_accuracy. Only global layouts
    the public operator accepts are used; the 6/1 local layout of
    Qwen3.8-Flash-Next TP4 is covered by test_qsa_prefill.
    """
    from sgl_kernel_npu.operators.attention import qsa_prefill

    global_heads, rows, length = 16, 17, 257
    base = length - rows
    q, k, v, blocks, table, request_row = generate_data(
        (rows, global_heads, 2), "fragmented", length
    )
    impl, impl_ref = impl_pair
    local_heads = global_heads // tp
    shards = []
    for rank in range(tp):
        kv_head = rank // (tp // 2)
        shards.append(
            qsa_prefill(
                q[:, rank * local_heads : (rank + 1) * local_heads].contiguous(),
                k[:, kv_head : kv_head + 1].contiguous(),
                v[:, kv_head : kv_head + 1].contiguous(),
                blocks,
                table,
                request_row,
                length,
                base,
                impl=impl,
            )
        )
    reference = qsa_prefill(
        q, k, v, blocks, table, request_row, length, base, impl=impl_ref
    )
    assert_accuracy(torch.cat(shards, 1), reference)


@requires_native
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 4, 1), id="unsupported-4x1"),
        pytest.param((1, 6, 2), id="unsupported-6x2"),
        pytest.param((1, 16, 1), id="unsupported-16x1"),
        pytest.param((1, 8, 2), id="unsupported-8x2"),
    ],
)
def test_qsa_prefill_rejects_head_layouts(shape):
    """Unsupported local Q/KV head counts must be rejected before launching."""
    from sgl_kernel_npu.operators.attention import qsa_prefill

    length = 67
    inputs = generate_data(shape, "shared", length)
    with pytest.raises(ValueError, match="unsupported Q/KV heads"):
        qsa_prefill(*inputs, length, length - shape[0])


@requires_native
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("scale-zero", id="scale-zero"),
        pytest.param("scale-nan", id="scale-nan"),
        pytest.param("scale-inf", id="scale-inf"),
        pytest.param("dtype-float32", id="dtype-float32"),
        pytest.param("blocks-511", id="blocks-511-columns"),
        pytest.param("length-past-table", id="length-past-table"),
        pytest.param("request-row-int32", id="request-row-int32"),
        pytest.param("q-noncontiguous", id="q-noncontiguous"),
    ],
)
def test_qsa_prefill_rejects_invalid_arguments(invalid):
    """Every contract violation raises ValueError before any kernel launch."""
    from sgl_kernel_npu.operators.attention import qsa_prefill

    length = 67
    shape = (1, 16, 2)
    q, k, v, blocks, table, request_row = generate_data(shape, "shared", length)
    base = length - shape[0]
    scale = 1 / 16
    if invalid == "scale-zero":
        scale = 0
    elif invalid == "scale-nan":
        scale = float("nan")
    elif invalid == "scale-inf":
        scale = float("inf")
    elif invalid == "dtype-float32":
        q = q.float()
    elif invalid == "blocks-511":
        blocks = blocks[:, :511].contiguous()
    elif invalid == "length-past-table":
        length = table.shape[1] + 1
    elif invalid == "request-row-int32":
        request_row = request_row.int()
    elif invalid == "q-noncontiguous":
        q = torch.randn(shape[0], shape[1], 512, dtype=torch.bfloat16)[:, :, :256]
    else:
        raise ValueError(f"Unknown invalid case: {invalid}")
    with pytest.raises(ValueError):
        qsa_prefill(q, k, v, blocks, table, request_row, length, base, scale)
