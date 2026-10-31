"""Accuracy/performance tests for indexer.flash_decode_bnsd_with_topk_idx.

``topk_idx`` is int32 and compared exactly, so ``build_inputs`` encodes the
block index in the scores. Run from the repo root with an NPU and
``torch_npu``/Triton-Ascend installed; see ``tests/python/README.md``.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

BLOCK = 128
HEAD_DIM = 128


def build_inputs(
    *,
    batch=1,
    q_heads=1,
    kv_heads=1,
    seq_lens=(4096,),
    max_seqlen=None,
    topk=16,
    score_type="max",
    page="direct",
    packed=False,
    index_dtype=torch.int32,
    mode="spaced",
    fused=True,
):
    """Create one case's inputs; ``mode="desc"`` flips the block-score order.

    Scores are ``block_index + last_valid_token``, encoded in bf16-exact
    channels so adjacent blocks never tie, even at 1M context.
    """
    gqa = q_heads // kv_heads
    seq_lens = tuple(seq_lens)
    assert len(seq_lens) == (batch * gqa if packed else batch)
    capacity = max(1, max((length + BLOCK - 1) // BLOCK for length in seq_lens))
    if max_seqlen is None:
        max_seqlen = max(seq_lens)

    torch.manual_seed(42)
    q = torch.randn(batch, q_heads, HEAD_DIM, dtype=torch.bfloat16)
    k_cache = torch.zeros(capacity, BLOCK, kv_heads, HEAD_DIM, dtype=torch.bfloat16)
    index = torch.arange(capacity, dtype=torch.float32)
    if mode == "desc":
        index = index.flip(0)
    k_cache[:, :, :, 0] = (index % 256).view(-1, 1, 1)
    k_cache[:, :, :, 1] = (index // 256).view(-1, 1, 1)
    k_cache[:, :, :, 2] = torch.arange(BLOCK, dtype=torch.float32).view(1, -1, 1)
    q[:, :, 0] = 1.0
    q[:, :, 1] = 256.0
    q[:, :, 2] = 1.0

    block_table = req_to_token = req_pool_indices = None
    if page == "table":
        block_table = torch.arange(capacity, dtype=index_dtype).view(1, -1)
        block_table = block_table.repeat(batch, 1).contiguous()
    else:
        req_to_token = torch.zeros(batch, capacity * BLOCK, dtype=index_dtype)
        req_to_token[:, ::BLOCK] = torch.arange(capacity, dtype=index_dtype) * BLOCK
        req_pool_indices = torch.arange(batch, dtype=index_dtype)

    return dict(
        q=q,
        sink=None,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=None,
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, dtype=index_dtype),
        max_seqlen=max_seqlen,
        block_size=BLOCK,
        topk=topk,
        init_blocks=0,
        local_blocks=0,
        score_type=score_type,
        disable_index_value=True,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=capacity,
        num_pages=capacity,
        packed_seq_lens=packed,
        runtime_fill_only=True,
        fused_append_local=fused,
    )


def run_case(*, perf=False, skip_acc=False, **params):
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    executor = check_accuracy(
        flash_decode_bnsd_with_topk_idx,
        kwargs=build_inputs(**params),
        skip_acc_check=skip_acc,
    )
    result = executor()[1]
    check_performance(executor, enable_perf=perf)
    return result


@pytest.mark.parametrize(
    ("batch", "q_heads", "seq_lens"),
    [
        pytest.param(1, 1, (131072,), id="minimax-m3-decode-1x128k"),
        pytest.param(16, 1, (4096,) * 16, id="minimax-m3-decode-16x4k"),
        pytest.param(4, 2, (32768,) * 4, id="minimax-m3-decode-4x32k-tp2"),
        pytest.param(4, 4, (8192,) * 4, id="minimax-m3-decode-4x8k-tp1"),
    ],
)
def test_decode(batch, q_heads, seq_lens):
    run_case(batch=batch, q_heads=q_heads, seq_lens=seq_lens, perf=True)


def test_graph_capture_capacity():
    run_case(seq_lens=(4096,), max_seqlen=1048576, perf=True)


def test_long_context_1m():
    run_case(seq_lens=(1048576,), perf=True, skip_acc=True)


@pytest.mark.parametrize(
    ("batch", "q_heads", "seq_lens"),
    [
        pytest.param(4, 3, tuple(range(4096, 4108)), id="minimax-m3-verify-packed-4x3"),
        pytest.param(12, 1, (4096,) * 12, id="minimax-m3-verify-packed-flat"),
    ],
)
def test_packed(batch, q_heads, seq_lens):
    run_case(
        batch=batch,
        q_heads=q_heads,
        seq_lens=seq_lens,
        max_seqlen=1048576,
        packed=True,
        perf=True,
    )


@pytest.mark.extra_checks
def test_block_table_int64():
    run_case(batch=2, seq_lens=(4096, 1024), page="table", index_dtype=torch.int64)


@pytest.mark.extra_checks
def test_lse():
    run_case(score_type="lse")


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"topk": 0}, id="edge-topk-zero"),
        pytest.param({"seq_lens": (0,), "max_seqlen": 128, "topk": 4}, id="edge-empty"),
        pytest.param({"seq_lens": (100,)}, id="edge-sl-lt-block"),
        pytest.param(
            {"seq_lens": (2048,), "topk": 20, "fused": False}, id="edge-topk-ge-blocks"
        ),
        pytest.param(
            {"batch": 2, "seq_lens": (129, 257), "topk": 1}, id="edge-unaligned"
        ),
    ],
)
def test_edges(params):
    run_case(**params)


@pytest.mark.extra_checks
def test_local_block_append_contract():
    assert run_case(seq_lens=(512,), topk=2)[0, 0].tolist() == [3, 2, -1]
    assert run_case(seq_lens=(512,), topk=2, mode="desc")[0, 0].tolist() == [0, 1, 3]


def _base(**overrides):
    return build_inputs(seq_lens=(256,), topk=4, **overrides)


def _rejected(**kwargs):
    from sgl_kernel_npu.operators.indexer import flash_decode_bnsd_with_topk_idx

    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_topk_idx(**kwargs)


@pytest.mark.extra_checks
def test_missing_page_source():
    _rejected(**{**_base(), "req_to_token": None, "req_pool_indices": None})


@pytest.mark.extra_checks
def test_invalid_direct_page_source():
    for field in ("max_num_blocks", "num_pages"):
        _rejected(**{**_base(), field: 0})


@pytest.mark.extra_checks
def test_packed_requires_single_kv_head():
    kwargs = build_inputs(
        batch=1, q_heads=4, kv_heads=2, seq_lens=(256, 256), packed=True
    )
    _rejected(**kwargs)


@pytest.mark.extra_checks
def test_torch_native_rejects_attention():
    _rejected(**{**_base(), "disable_index_value": False}, impl="torch_native")
