"""Accuracy/performance tests for indexer.flash_decode_bnsd_with_topk_idx.

Run:
    python -m pytest tests/python/operators/indexer/test_flash_decode_bnsd_with_topk_idx.py -sv
    python -m pytest tests/python/operators/indexer/test_flash_decode_bnsd_with_topk_idx.py \
        --prof-dir=results/profilings --csv-path=results/flash_decode_bnsd.csv -q -s

The DUT (default ``impl="auto"``, i.e. Triton) is compared against the
``torch_native`` reference. The reference covers the score-only branch,
including ``max``/``lse``, ``init_blocks``/``local_blocks`` forcing, the direct
page source and ``block_table``; the attention-output branch is rejected and
not covered here. ``topk_idx`` is int32 and must match exactly, so score-only
cases use deliberately spaced block scores (see ``build_inputs``).
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Case ids selected for performance collection (production shapes).
perf_shape = [
    "m3-decode-1x128k",
    "m3-decode-16x4k",
    "m3-decode-4x32k-tp2",
    "m3-decode-1x1m",
    "m3-verify-packed-4x3",
]
# The 1M context case only collects performance; its reference comparison is
# skipped because the torch_native gather would dominate the runtime.
skip_acc_check_cases = {"m3-decode-1x1m"}


def case(cid, **overrides):
    """Describe one test case; shapes follow the production call sites."""
    values = dict(
        id=cid,
        batch=1,
        q_heads=1,
        kv_heads=1,
        head_dim=128,
        block_size=128,
        topk=16,
        seq_lens=[4096],
        max_seqlen=4096,
        score_type="max",
        page="direct",
        fused=True,
        init_blocks=0,
        local_blocks=0,
        packed=False,
        adaptive=None,
        score_max_chunks=32,
        blocks_per_chunk=16,
        sanitize=False,
        dtype=torch.bfloat16,
        index_dtype="int32",
        mode="spaced",
        runtime_fill_only=True,
        compat_flags=False,
        seed=42,
        expected=None,
        expected_rows=None,
    )
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def build_inputs(spec):
    """Create on-device inputs and keyword arguments for one case."""
    batch = spec["batch"]
    q_heads = spec["q_heads"]
    kv_heads = spec["kv_heads"]
    head_dim = spec["head_dim"]
    block_size = spec["block_size"]
    topk = spec["topk"]
    dtype = spec["dtype"]
    gqa = q_heads // kv_heads
    seq_lens = list(spec["seq_lens"])
    assert len(seq_lens) == (batch * gqa if spec["packed"] else batch)
    num_blocks = [(value + block_size - 1) // block_size for value in seq_lens]
    max_blocks = max(num_blocks)
    capacity = max(1, max_blocks)
    if spec["sanitize"]:
        capacity += 2

    torch.manual_seed(spec["seed"])
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "manual_seed"):
        npu.manual_seed(spec["seed"])
    q = torch.randn(batch, q_heads, head_dim, dtype=dtype)
    k_cache = torch.zeros(capacity, block_size, kv_heads, head_dim, dtype=dtype)
    index = torch.arange(capacity, dtype=torch.float32, device=q.device)
    if spec["mode"] == "spaced-desc":
        index = index.flip(0)
    elif spec["mode"] != "spaced":
        raise ValueError(f"unknown mode {spec['mode']!r}")
    # Encode the block index in two bf16-exact channels (low byte and high
    # bits) and read it with q=[1, 256, 1]; the block score is
    # ``index + last_valid_token``, so adjacent blocks stay distinct even at 1M
    # context (a single bf16 coefficient would collapse above 256) and a wrong
    # causal mask changes the tail block's score.
    for kv_head in range(kv_heads):
        shifted = (index + kv_head * 7) % capacity
        k_cache[:, :, kv_head, 0] = (shifted % 256).view(-1, 1)
        k_cache[:, :, kv_head, 1] = (shifted // 256).view(-1, 1)
    token_ids = torch.arange(block_size, dtype=torch.float32, device=q.device)
    k_cache[:, :, :, 2] = token_ids.view(1, -1, 1)
    q[:, :, 0] = 1.0
    q[:, :, 1] = 256.0
    q[:, :, 2] = 1.0
    encoded = k_cache[:, 0, 0, 0].float() + 256.0 * k_cache[:, 0, 0, 1].float()
    assert (
        torch.unique(encoded).numel() == capacity
    ), "block scores must stay distinct after the dtype cast"

    index_dtype = torch.int64 if spec["index_dtype"] == "int64" else torch.int32
    block_table = None
    req_to_token = None
    req_pool_indices = None
    if spec["page"] == "table":
        block_table = (
            torch.arange(capacity, dtype=index_dtype)
            .view(1, -1)
            .repeat(batch, 1)
            .contiguous()
        )
    else:
        columns = capacity * block_size
        req_to_token = torch.zeros(batch, columns, dtype=index_dtype)
        token_cols = torch.arange(capacity, dtype=torch.long) * block_size
        token_slots = torch.arange(capacity, dtype=index_dtype) * block_size
        req_to_token[:, token_cols] = token_slots
        if spec["sanitize"]:
            # The last used logical block points outside the page table; the
            # sanitize_page_ids clamp maps it to num_pages - 1.
            req_to_token[0, (max_blocks - 1) * block_size] = (capacity + 4) * block_size
        req_pool_indices = torch.arange(batch, dtype=index_dtype)
    lengths = torch.tensor(seq_lens, dtype=index_dtype)

    kwargs = dict(
        q=q,
        sink=None,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=None,
        block_table=block_table,
        seq_lens=lengths,
        max_seqlen=spec["max_seqlen"],
        block_size=block_size,
        topk=topk,
        init_blocks=spec["init_blocks"],
        local_blocks=spec["local_blocks"],
        sm_scale=None,
        score_type=spec["score_type"],
        disable_index_value=True,
        num_kv_chunks=None,
        max_num_kv_chunks=8,
        score_blocks_per_chunk=spec["blocks_per_chunk"],
        score_max_chunks=spec["score_max_chunks"],
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=capacity,
        num_pages=capacity,
        sanitize_page_ids=spec["sanitize"],
        packed_seq_lens=spec["packed"],
        runtime_fill_only=spec["runtime_fill_only"],
        fused_append_local=spec["fused"],
    )
    if spec["adaptive"] is not None:
        short_blocks, short_chunks = spec["adaptive"]
        kwargs["runtime_score_short_max_blocks"] = short_blocks
        kwargs["runtime_score_short_chunks"] = short_chunks
    if spec["compat_flags"]:
        kwargs.update(
            use_native=True,
            use_triton_topk=True,
            num_topk_chunks=8,
            use_chunked_score=True,
        )
    return kwargs


def _pass_through_expected(num_blocks, topk, fused):
    """C4: ``[0..nb-1]`` then repeat block 0; fused appends the local block."""
    values = list(range(num_blocks)) + [0] * (topk - num_blocks)
    if fused:
        local = num_blocks - 1
        values.append(-1 if local in values[:topk] else local)
    return values


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

PRODUCTION_CASES = [
    case(
        "m3-decode-1x128k",
        seq_lens=[131072],
        max_seqlen=131072,
        score_max_chunks=16,
    ),
    case("m3-decode-16x4k", batch=16, seq_lens=[4096] * 16),
    case(
        "m3-decode-4x32k-tp2",
        batch=4,
        q_heads=2,
        seq_lens=[32768] * 4,
        max_seqlen=32768,
    ),
    case(
        "m3-decode-graph-cap",
        seq_lens=[4096],
        max_seqlen=1048576,
        score_max_chunks=16,
    ),
    case(
        "m3-decode-4x8k-tp1",
        batch=4,
        q_heads=4,
        seq_lens=[8192] * 4,
        max_seqlen=8192,
    ),
    case(
        "m3-verify-packed-4x3",
        batch=4,
        q_heads=3,
        seq_lens=[4096 + index for index in range(12)],
        max_seqlen=1048576,
        packed=True,
        adaptive=(256, 16),
        blocks_per_chunk=8,
        score_max_chunks=64,
        sanitize=True,
        runtime_fill_only=False,
    ),
    case(
        "m3-verify-packed-flat",
        batch=12,
        seq_lens=[4096] * 12,
        max_seqlen=1048576,
        packed=True,
        score_max_chunks=64,
    ),
    case(
        "m3-decode-1x1m",
        seq_lens=[1048576],
        max_seqlen=1048576,
        score_max_chunks=16,
    ),
]

EDGE_CASES = [
    case(
        "edge-sl-lt-block",
        seq_lens=[100],
        expected=_pass_through_expected(1, 16, True),
    ),
    case(
        "edge-nb-eq-topk",
        batch=1,
        q_heads=2,
        seq_lens=[2048, 1024],
        packed=True,
        expected_rows=[
            list(range(16)) + [-1],
            list(range(8)) + [0] * 8 + [-1],
        ],
    ),
    case("edge-nb-topk+1", seq_lens=[2176]),
    case(
        "edge-sl-unaligned",
        batch=2,
        seq_lens=[129, 257],
        max_seqlen=257,
        topk=1,
    ),
    case(
        "edge-chunk-more-than-blocks",
        seq_lens=[2048],
        max_seqlen=131072,
        topk=4,
    ),
    case(
        "edge-fill-only-true",
        seq_lens=[4096],
        max_seqlen=512 * 128,
    ),
    case(
        "edge-fill-only-false",
        seq_lens=[4096],
        max_seqlen=513 * 128,
    ),
    case(
        "edge-adaptive-256",
        batch=3,
        seq_lens=[255 * 128, 256 * 128, 257 * 128],
        max_seqlen=1024 * 128,
        adaptive=(256, 16),
    ),
    # Ascending scores select the trailing (local) block, so the fused append
    # de-duplicates it to -1.
    case("edge-local-dedup", seq_lens=[512], topk=2),
    # Descending scores leave the local block out of the topk; it is appended.
    case("edge-local-append", seq_lens=[512], topk=2, mode="spaced-desc"),
    case(
        "edge-topk-ge-nb",
        seq_lens=[2048],
        topk=20,
        fused=False,
        expected=_pass_through_expected(16, 20, False),
    ),
    case("edge-topk-zero", topk=0, seq_lens=[4096], fused=False),
    case("edge-topk-zero-fused", topk=0, seq_lens=[4096], fused=True),
    case("edge-1x1", seq_lens=[1], max_seqlen=1),
    case(
        "edge-empty",
        seq_lens=[0],
        max_seqlen=128,
        topk=4,
        expected=[-1, -1, -1, -1, 0],
    ),
    case("edge-sanitize", seq_lens=[4096], sanitize=True),
    case(
        "edge-block-table",
        batch=2,
        seq_lens=[4096, 1024],
        page="table",
    ),
    case("edge-lse", seq_lens=[4096], score_type="lse"),
    # One block per forced class: two blocks of the same class would tie and
    # the reference/kernel tie order is undefined.
    case(
        "edge-init-local",
        seq_lens=[1024],
        topk=4,
        init_blocks=1,
        local_blocks=1,
    ),
    case(
        "edge-idx-int64",
        batch=2,
        seq_lens=[4096, 1024],
        index_dtype="int64",
    ),
    case(
        "edge-idx-int64-table",
        batch=2,
        seq_lens=[4096, 1024],
        page="table",
        index_dtype="int64",
    ),
    case(
        "edge-gqa-kvh2",
        batch=2,
        q_heads=4,
        kv_heads=2,
        seq_lens=[4096, 2048],
        page="table",
        topk=8,
    ),
    case("edge-dim-64", batch=2, head_dim=64, seq_lens=[4096, 2048]),
    case("edge-fp16", dtype=torch.float16, seq_lens=[4096]),
    case("edge-compat-flags", seq_lens=[4096], compat_flags=True),
]

CASES = PRODUCTION_CASES + EDGE_CASES


@pytest.mark.parametrize("spec", CASES, ids=[spec["id"] for spec in CASES])
def test_flash_decode_bnsd_with_topk_idx(spec):
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(spec)
    executor = check_accuracy(
        flash_decode_bnsd_with_topk_idx,
        kwargs=kwargs,
        skip_acc_check=spec["id"] in skip_acc_check_cases,
    )
    if spec["expected"] is not None or spec["expected_rows"] is not None:
        result = executor()[1]
        if spec["expected"] is not None:
            assert result[0, 0].tolist() == spec["expected"]
        if spec["expected_rows"] is not None:
            assert result[:, 0, :].tolist() == spec["expected_rows"]
    enable_perf = spec["id"] in perf_shape
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.xfail(
    strict=False,
    reason="C15: packed rows crossing runtime_score_short_max_blocks may truncate "
    "the merged candidate list; equivalence is not guaranteed",
)
def test_packed_adaptive_cross_short_boundary():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    spec = case(
        "edge-packed-adaptive-mixed",
        batch=1,
        q_heads=3,
        seq_lens=[255 * 128, 256 * 128, 257 * 128],
        max_seqlen=1024 * 128,
        packed=True,
        adaptive=(256, 16),
    )
    check_accuracy(flash_decode_bnsd_with_topk_idx, kwargs=build_inputs(spec))


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def _base_spec(**overrides):
    return case("negative", seq_lens=[256], **overrides)


def test_missing_page_source():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(_base_spec())
    kwargs.update(req_to_token=None, req_pool_indices=None)
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_num_blocks", None),
        ("max_num_blocks", 0),
        ("num_pages", None),
        ("num_pages", 0),
        ("req_pool_indices", None),
        ("req_to_token", None),
    ],
    ids=[
        "capacity-none",
        "capacity-zero",
        "num-pages-none",
        "num-pages-zero",
        "missing-pool-indices",
        "missing-request-map",
    ],
)
def test_invalid_direct_page_source(field, value):
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(_base_spec())
    kwargs[field] = value
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs)


def test_capacity_exceeds_request_map_columns():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(_base_spec())
    kwargs["max_num_blocks"] += 1
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs)


def test_direct_requires_capacity():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    for value in (None, 0):
        kwargs = build_inputs(_base_spec(fused=False))
        kwargs["max_num_blocks"] = value
        with pytest.raises(AssertionError):
            flash_decode_bnsd_with_topk_idx(**kwargs)


def test_input_ndim_validation():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    for impl in ("triton", "torch_native"):
        kwargs = build_inputs(_base_spec())
        kwargs["seq_lens"] = kwargs["seq_lens"].view(1, -1)
        with pytest.raises(AssertionError):
            flash_decode_bnsd_with_topk_idx(**kwargs, impl=impl)

        kwargs = build_inputs(_base_spec(page="table"))
        kwargs["block_table"] = kwargs["block_table"].view(-1)
        with pytest.raises(AssertionError):
            flash_decode_bnsd_with_topk_idx(**kwargs, impl=impl)


def test_packed_requires_single_kv_head():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    spec = case(
        "negative-packed-kvh2",
        q_heads=4,
        kv_heads=2,
        seq_lens=[256, 256],
        packed=True,
    )
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**build_inputs(spec))


def test_torch_native_rejects_attention_branch():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(_base_spec())
    kwargs["disable_index_value"] = False
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs, impl="torch_native")


def test_block_table_fused_requires_capacity():
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    kwargs = build_inputs(_base_spec(page="table"))
    kwargs["max_num_blocks"] = None
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs)


def test_registered_metadata():
    from sgl_kernel_npu.operators import list_impls
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx
    from sgl_kernel_npu.operators.utils.registry import get_op_name

    assert list_impls(flash_decode_bnsd_with_topk_idx) == [
        "auto",
        "torch_native",
        "triton",
    ]
    assert get_op_name(flash_decode_bnsd_with_topk_idx) == (
        "indexer.flash_decode_bnsd_with_topk_idx"
    )
