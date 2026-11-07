"""Public prefill API: independent data generation, accuracy, then performance."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.performance import check_performance

HEAD_DIM = 128
# MiniMax-M3 workload dimensions retained from the original tests (not runtime captures).
perf_shape = {"minimax-m3-80k", "minimax-m3-250k"}

# q lens, KV lengths, query grouping, index heads, KV heads, page size, topk
CASES = {
    "minimax-m3-80k": ((305,), (80817,), 64, 1, 1, 128, 16),
    "minimax-m3-250k": ((1536,), (251536,), 64, 1, 1, 128, 16),
    "packed": ((0, 5, 129, 40), (0, 700, 129, 4096), 16, 1, 1, 128, 16),
    "fresh": ((2048,), (2048,), 64, 1, 1, 128, 16),
    "single": ((1,), (1,), 1, 1, 1, 128, 1),
    "tail": ((37,), (300,), 10, 1, 1, 16, 4),
    "topk-gt-blocks": ((7,), (300,), 4, 1, 1, 128, 20),
    "page64": ((7,), (300,), 16, 1, 1, 64, 16),
    "empty": ((0,), (0,), 16, 1, 1, 128, 16),
    "kv-tail": ((129,), (257,), 16, 1, 1, 128, 8),
    "tie": ((37,), (300,), 16, 1, 1, 16, 4),
}


def generate_test_data(case, dtype=None, mode="encoded", topk=None):
    """NPU data: encoded ranks blocks; tie zeros Q; random tests attention."""
    q_lens, seq_lens, bsq, idx_heads, kv_heads, page, case_topk = CASES[case]
    topk = case_topk if topk is None else topk
    dtype = torch.bfloat16 if dtype is None else dtype
    torch.npu.manual_seed(42)
    total_q = sum(q_lens)
    blocks = max((length + page - 1) // page for length in seq_lens)
    requests = len(seq_lens)
    lengths = torch.tensor(q_lens, dtype=torch.long)
    cu = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0)
    prefix = (torch.tensor(seq_lens) - lengths).repeat_interleave(lengths)
    within = torch.arange(total_q) - cu[:-1].repeat_interleave(lengths)
    q = torch.randn(total_q, idx_heads, HEAD_DIM, dtype=dtype)
    k = torch.randn(requests * blocks, page, kv_heads, HEAD_DIM, dtype=dtype)
    v = torch.randn_like(k)

    if mode == "encoded":
        block_id = torch.arange(blocks, dtype=torch.float32)
        block_id = block_id.repeat(requests)
        q.zero_()
        # 32 * block_id in two exact channels; 65536 would overflow FP16.
        q[..., :3] = torch.tensor([32.0, 8192.0, 1.0], dtype=dtype)
        k.zero_()
        k[..., 0] = (block_id % 256).view(-1, 1, 1)
        k[..., 1] = (block_id // 256).view(-1, 1, 1)
        k[..., 2] = -torch.arange(page, dtype=torch.float32).view(1, -1, 1)
    elif mode == "tie":
        q.zero_()
    elif mode != "random":
        raise ValueError(f"unknown mode: {mode}")

    req_to_token = torch.zeros(requests, blocks * page, dtype=torch.int32)
    # Nonidentity page mapping catches accidental use of logical page ids.
    pages = torch.arange(requests * blocks).flip(0).reshape(requests, blocks)
    k = k.flip(0).contiguous()
    v = v.flip(0).contiguous()
    req_to_token[:, ::page] = pages * page
    inputs = dict(
        q=q,
        k_cache_bnsd=k,
        cu_seqlens=cu,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        req_to_token=req_to_token,
        req_pool_indices=torch.arange(requests, dtype=torch.int32),
        block_size_q=bsq,
        block_size_k=page,
        topk=topk,
        sm_scale=HEAD_DIM**-0.5,
        per_query_seq_lens=(prefix + within + 1).to(torch.int32),
    )
    inputs["v_cache_bnsd"] = v
    return inputs


def build_qblock_mappings(inputs):
    """Build caller-owned mappings independently of backend helpers."""
    cu, seq = inputs["cu_seqlens"].tolist(), inputs["seq_lens"].tolist()
    bsq, page = inputs["block_size_q"], inputs["block_size_k"]
    reqs = inputs["req_pool_indices"].tolist()
    rows = [
        (start, block, seq[r], end, reqs[r])
        for r, (start, end) in enumerate(zip(cu, cu[1:]))
        for block in range((end - start + bsq - 1) // bsq)
    ]
    # The Triton kernel reads these 1-D mappings contiguously, so unbind views
    # (stride = number of columns) must be materialized before being passed in.
    qstart, local, lengths, qend, req = (
        column.contiguous()
        for column in torch.tensor(rows, dtype=torch.int32).unbind(1)
    )
    cols = torch.arange((max(seq) + page - 1) // page) * page
    pages = inputs["req_to_token"][req.long()[:, None], cols] // page
    return qstart, local, lengths, qend, pages.int(), len(rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("score_type", ["max", "lse"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("minimax-m3-80k", id="minimax-m3-prefill-80k"),
        pytest.param("minimax-m3-250k", id="minimax-m3-prefill-250k"),
        pytest.param("packed", id="packed-ragged-requests"),
        pytest.param("fresh", id="fresh-prefill-causal-mask"),
        pytest.param("single", id="edge-single-block"),
        pytest.param("tail", id="edge-query-tail-and-nonpow2-q-block"),
        pytest.param("topk-gt-blocks", id="edge-topk-greater-than-block-count"),
        pytest.param("page64", id="edge-page-size-64"),
        pytest.param("empty", id="all-empty"),
        pytest.param("kv-tail", id="partial-kv-block"),
    ],
)
def test_flash_prefill_bnsd_indexer(case, dtype, score_type):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    inputs = generate_test_data(case, dtype=dtype)
    executor = check_accuracy(
        flash_prefill_bnsd_indexer,
        kwargs={**inputs, "score_type": score_type},
    )
    check_performance(
        executor,
        enable_perf=case in perf_shape
        and dtype == torch.bfloat16
        and score_type == "max",
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("topk", [0, 4, 32])
def test_fallback_without_per_query_seq_lens(topk):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    inputs = generate_test_data("tail", topk=topk)
    inputs.pop("per_query_seq_lens")
    executor = check_accuracy(flash_prefill_bnsd_indexer, kwargs=inputs)
    result = executor()
    assert result[1].shape == (1, 37, topk)

    assert result[1].dtype == torch.int32
    if topk > 19:
        assert (result[1][..., 19:] == -1).all()


@pytest.mark.extra_checks
def test_prebuilt_qblock_mappings_match_built_in():
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    inputs = generate_test_data("packed")
    executor = check_accuracy(
        flash_prefill_bnsd_indexer,
        kwargs={**inputs, "qblock_mappings": build_qblock_mappings(inputs)},
    )
    built_in = flash_prefill_bnsd_indexer(**inputs, impl=executor.impl)
    assert_accuracy(executor(), built_in)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "mode,local_len,expected",
    [
        ("tie", 300, [0, 1, 2, -1, 18]),  # Ties and unfilled slots.
        ("encoded", 300, [2, 1, 0, -1, 18]),  # Append distant local block.
        ("encoded", 37, [2, 1, 0, -1, -1]),  # Local already selected.
    ],
)
def test_topk_contract(mode, local_len, expected):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    inputs = generate_test_data("tie", mode=mode)
    inputs["per_query_seq_lens"][-1] = local_len
    executor = check_accuracy(flash_prefill_bnsd_indexer, kwargs=inputs)
    assert executor()[1][0, 36].tolist() == expected


@pytest.mark.extra_checks
@pytest.mark.parametrize("impl", ["triton", "torch_native"])
@pytest.mark.parametrize(
    "invalid", ["topk", "score_type", "length_dtype", "length_shape"]
)
def test_invalid_inputs(impl, invalid):
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    inputs = generate_test_data("tail")
    if invalid == "topk":
        inputs["topk"] = 0
    elif invalid == "score_type":
        inputs["score_type"] = "sum"
    elif invalid == "length_dtype":
        inputs["per_query_seq_lens"] = inputs["per_query_seq_lens"].long()
    elif invalid == "length_shape":
        inputs["per_query_seq_lens"] = inputs["per_query_seq_lens"][:-1]
    with pytest.raises(AssertionError):
        flash_prefill_bnsd_indexer(**inputs, impl=impl)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_random_attention(dtype):
    """Compare random attention independently of near-tie top-k rankings."""
    from sgl_kernel_npu.operators.indexer import flash_prefill_bnsd_indexer

    from tests.python.utils.common import (
        get_skip_acc_check,
        make_executor,
        require_distinct_impls,
    )

    inputs = generate_test_data("kv-tail", dtype=dtype, mode="random")
    executor = make_executor(flash_prefill_bnsd_indexer, kwargs=inputs)
    if get_skip_acc_check() is True:
        return  # This case only validates attention; it has no profiling workload.
    require_distinct_impls(flash_prefill_bnsd_indexer, executor.impl, executor.impl_ref)
    reference = flash_prefill_bnsd_indexer(**inputs, impl=executor.impl_ref)
    assert_accuracy(executor()[0], reference[0], path="idx_o")
