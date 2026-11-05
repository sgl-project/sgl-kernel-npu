"""Accuracy/performance tests for attention.flash_decode_bnsd_with_gqa_share_sparse.

Shapes follow the SGLang MiniMax-M3 call sites: decode (B rows over a direct
page map), verify (per-query rows sharing one request) and prefill (flattened
extend rows). ``test_production`` runs the CLI-selected impl pair over that
matrix and ``test_backends`` replays the three backend pairs with the bf16
tolerance measured on arch22. Closed forms pin the math without the reference.
Run with an NPU; add ``-m ""`` for the edge cases.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import make_executor, require_test_impl
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# Shapes selected for performance collection.
PERF_SEQUENCES = [(16384,), (16384,) * 16, (131072,)]

# Production cases; every entry fits the validated native configuration.
CASES = [
    pytest.param({"seq_lens": (16384,)}, id="minimax-m3-decode-b1-16k"),
    pytest.param({"seq_lens": (16384,) * 8}, id="minimax-m3-decode-b8-16k"),
    pytest.param({"seq_lens": (16384,) * 16}, id="minimax-m3-decode-b16-16k"),
    pytest.param({"seq_lens": (49152,)}, id="minimax-m3-decode-b1-48k"),
    pytest.param({"seq_lens": (131072,)}, id="minimax-m3-decode-b1-128k"),
    # TARGET_VERIFY: request-major rows at prefix+1 .. prefix+3, 4 requests.
    pytest.param(
        {"seq_lens": (8193, 8194, 8195) * 4, "shared_requests": 4, "sanitize": True},
        id="minimax-m3-verify-4x3",
    ),
    # Prefill: 8 requests x 64 extend queries, flattened request-major.
    pytest.param(
        {
            "seq_lens": tuple(
                8192 + request * 64 + offset
                for request in range(8)
                for offset in range(64)
            ),
            "shared_requests": 8,
            "sanitize": True,
        },
        id="minimax-m3-prefill-q512",
    ),
]


def build_inputs(
    *,
    seq_lens=(16384,),
    q_heads=8,  # M3 num_q_heads per TP=16 rank
    kv_heads=1,
    head_dim=128,  # M3 head_dim
    block_size=128,  # M3 sparse_block_size
    topk=17,  # 16 scored + 1 local block
    dtype=torch.bfloat16,
    page="direct",
    sink=False,
    num_topk_chunks=None,
    blocks_per_step=1,
    sanitize=False,
    shared_requests=None,
    num_pages=None,
    seed=42,
):
    """Create one case's inputs; ``batch`` is implied by ``seq_lens``."""
    seq_lens = tuple(seq_lens)
    batch = len(seq_lens)
    capacity = max(
        1, max((length + block_size - 1) // block_size for length in seq_lens)
    )
    torch.manual_seed(seed)

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype)
    k_cache = torch.randn(capacity, block_size, kv_heads, head_dim, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    # Distinct block ids per row, sorted; slots beyond the block count stay -1.
    valid = max(1, min(topk, capacity))
    topk_idx = torch.full((kv_heads, batch, topk), -1, dtype=torch.int32)
    for head in range(kv_heads):
        for row in range(batch):
            ids = torch.randperm(capacity)[:valid].sort().values
            topk_idx[head, row, :valid] = ids

    block_table = req_to_token = req_pool_indices = None
    if page == "table":
        block_table = torch.arange(capacity, dtype=torch.int32).repeat(batch, 1)
    else:
        # Direct map: block b starts at token slot b*block_size. Shared requests
        # mirror repeat_interleave, so several rows point at one request row.
        requests = batch if shared_requests is None else shared_requests
        req_to_token = torch.zeros(requests, capacity * block_size, dtype=torch.int32)
        req_to_token[:, ::block_size] = (
            torch.arange(capacity, dtype=torch.int32) * block_size
        )
        req_pool_indices = torch.arange(requests, dtype=torch.int32).repeat_interleave(
            batch // requests
        )

    return dict(
        q=q,
        sink=torch.randn(q_heads, head_dim, dtype=dtype) if sink else None,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=v_cache,
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_size=block_size,
        topk_idx=topk_idx,
        sm_scale=head_dim**-0.5,
        num_topk_chunks=num_topk_chunks,
        max_num_topk_chunks=8,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        max_num_blocks=capacity,
        num_pages=capacity if num_pages is None else num_pages,
        sanitize_page_ids=sanitize,
        topk_blocks_per_step=blocks_per_step,
    )


def run_case(*, perf=False, impl=None, impl_ref=None, **params):
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    executor = check_accuracy(
        flash_decode_bnsd_with_gqa_share_sparse,
        kwargs=build_inputs(**params),
        impl=impl,
        impl_ref=impl_ref,
    )
    check_performance(executor, enable_perf=perf)


def uniform_kv_inputs(*, seq_len=4096, batch=2):
    """K and V are constant, so any non-empty row must return exactly V."""
    kwargs = build_inputs(seq_lens=(seq_len,) * batch, seed=7)
    q = kwargs["q"]
    k0 = torch.randn(q.shape[-1], dtype=q.dtype)
    v0 = torch.randn(q.shape[-1], dtype=q.dtype)
    shape = kwargs["k_cache_bnsd"].shape
    kwargs["k_cache_bnsd"] = k0.view(1, 1, 1, -1).expand(shape).contiguous()
    kwargs["v_cache_bnsd"] = v0.view(1, 1, 1, -1).expand(shape).contiguous()
    return kwargs, q, v0


@pytest.mark.parametrize("params", CASES)
def test_production(params):
    """DUT (impl pair from the CLI) against the reference on production shapes."""
    run_case(perf=params["seq_lens"] in PERF_SEQUENCES, **params)


@pytest.mark.extra_checks
def test_multi_chunk():
    """Split-topk: several chunks are reduced by the merge kernel."""
    run_case(seq_lens=(4096, 8192), num_topk_chunks=4)


@pytest.mark.extra_checks
def test_block_fusion():
    """Two blocks per step overflow the UB on Triton, so only native runs it."""
    run_case(seq_lens=(4096,), blocks_per_step=2, impl="ascendc")


@pytest.mark.extra_checks
def test_block_table():
    """The other page source: a logical-to-physical table instead of the map."""
    run_case(seq_lens=(4096, 1024), page="table")


@pytest.mark.extra_checks
def test_fp16():
    run_case(seq_lens=(4096,), dtype=torch.float16)


@pytest.mark.extra_checks
def test_sanitize_page_ids():
    """Selected pages are beyond num_pages=1, so both backends must clamp."""
    run_case(seq_lens=(4096,), sanitize=True, num_pages=1)


@pytest.mark.extra_checks
def test_sink():
    """sink is Triton-only: the native op has no such input."""
    run_case(seq_lens=(4096,), sink=True, impl="triton")


@pytest.mark.extra_checks
def test_auto_keeps_sink_on_triton():
    """auto must not route a non-None sink to the native op."""
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    kwargs = build_inputs(seq_lens=(4096,), sink=True)
    auto_out = flash_decode_bnsd_with_gqa_share_sparse(**kwargs)
    triton_out = flash_decode_bnsd_with_gqa_share_sparse(**kwargs, impl="triton")
    assert_accuracy(auto_out, triton_out, {"torch.bfloat16": AccuracyStandard(0, 0)})


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "overrides", [{"sink": True}, {"head_dim": 64}], ids=["sink", "head-dim-64"]
)
def test_ascendc_rejects_unsupported(overrides):
    """The native op has no sink input and only D=128/block_size=128 tilings.

    ``ascendc`` is arch22-only, so the framework skips this case on other
    generations before the input asserts run.
    """
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    require_test_impl(flash_decode_bnsd_with_gqa_share_sparse, "ascendc")
    kwargs = build_inputs(seq_lens=(4096,), topk=4, **overrides)
    with pytest.raises(AssertionError):
        flash_decode_bnsd_with_gqa_share_sparse(**kwargs, impl="ascendc")


@pytest.mark.extra_checks
def test_masked_slots_may_hold_nan():
    """K/V are loaded under the position mask, so unused slots may be junk."""
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    kwargs = build_inputs(seq_lens=(100,), topk=1, num_topk_chunks=1)
    kwargs["k_cache_bnsd"][:, 100:] = float("nan")
    kwargs["v_cache_bnsd"][:, 100:] = float("nan")
    out = flash_decode_bnsd_with_gqa_share_sparse(**kwargs, impl="triton")
    assert torch.isfinite(out).all()


def test_uniform_kv_returns_v():
    """Constant K/V: the softmax weights sum to one, so the output is V."""
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    kwargs, q, v0 = uniform_kv_inputs()
    expected = v0.view(1, 1, -1).expand_as(q).contiguous()
    executor = make_executor(flash_decode_bnsd_with_gqa_share_sparse, kwargs=kwargs)
    assert_accuracy(
        executor(), expected, {"torch.bfloat16": AccuracyStandard(1e-2, 1e-2)}
    )


def test_empty_row_returns_zero():
    """A row with no valid token contributes nothing and must be exactly zero."""
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    kwargs, q, _ = uniform_kv_inputs(seq_len=0, batch=1)
    executor = make_executor(flash_decode_bnsd_with_gqa_share_sparse, kwargs=kwargs)
    assert_accuracy(
        executor(), torch.zeros_like(q), {"torch.bfloat16": AccuracyStandard(0, 0)}
    )
