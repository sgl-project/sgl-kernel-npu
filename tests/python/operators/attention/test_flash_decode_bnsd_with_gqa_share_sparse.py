"""Tests for attention.flash_decode_bnsd_with_gqa_share_sparse.

Production shapes come from the SGLang MiniMax-M3 call sites: decode, verify
and prefill. ``test_production`` runs the CLI-selected pair over that matrix and
``test_parity_production`` runs the three backend pairs; ``EDGE_CASES`` covers
configurations outside it behind ``extra_checks``. Two closed forms pin the
math without the reference. Ascendc cases skip when the native op is missing,
and sink / head_dim 64 stay on the Triton side. Use ``-m ""``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from tests.python.utils.accuracy import assert_accuracy, check_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

BLOCK = 128  # M3 sparse_block_size
HEAD_DIM = 128  # M3 head_dim
Q_HEADS = 8  # M3 num_q_heads per TP=16 rank
KV_HEADS = 1
TOPK = 17  # 16 scored + 1 local block
EXACT = {"torch.bfloat16": AccuracyStandard(atol=0, rtol=0)}
TIGHT = {"torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2)}


def _verify_lens(requests, ndt=3, prefix=8192):
    """TARGET_VERIFY: request-major per-query lengths ``prefix + 1 .. prefix + ndt``."""
    return tuple(
        prefix + offset for _ in range(requests) for offset in range(1, ndt + 1)
    )


def _prefill_lens(requests, per_request, prefix=8192):
    """Flattened extend queries, request-major, ascending inside each request."""
    return tuple(
        prefix + request * per_request + offset
        for request in range(requests)
        for offset in range(per_request)
    )


# M3 production shapes: 8 query heads, 1 KV head per TP=16 rank, block_size
# 128, topk 17, bf16, direct page map. Verify/prefill enable sanitize and share
# one request row per request; every entry fits the validated native config.
PRODUCTION_CASES = [
    ("m3-decode-b1-16k", {"seq_lens": (16384,)}, True, True),
    ("m3-decode-b8-16k", {"seq_lens": (16384,) * 8}, True, True),
    ("m3-decode-b16-16k", {"seq_lens": (16384,) * 16}, True, True),
    ("m3-decode-b1-48k", {"seq_lens": (49152,)}, True, True),
    ("m3-decode-b1-128k", {"seq_lens": (131072,)}, True, True),
    (
        "m3-verify-bs4-ndt3",
        {"seq_lens": _verify_lens(4), "shared_requests": 4, "sanitize": True},
        True,
        True,
    ),
    (
        "m3-prefill-q512",
        {"seq_lens": _prefill_lens(8, 64), "shared_requests": 8, "sanitize": True},
        True,
        True,
    ),
    # Triton cannot compile blocks_per_step=2 at 128/128 (the fused K/V tile
    # overflows the 192 KB UB), so only the native backend runs this case.
    (
        "m3-prefill-q64-bps2",
        {
            "seq_lens": _prefill_lens(8, 8),
            "shared_requests": 8,
            "blocks_per_step": 2,
            "num_stages": 1,
            "sanitize": True,
        },
        True,
        False,
    ),
]

# Only these production cases collect performance.
PERF_CASES = {"m3-decode-b1-16k", "m3-decode-b16-16k", "m3-decode-b1-128k"}

PRODUCTION_IDS = [case[0] for case in PRODUCTION_CASES]


def _native_available():
    """True when this generation and this build both provide the native op."""
    try:
        from sgl_kernel_npu.operators.attention import (
            flash_decode_bnsd_with_gqa_share_sparse as op,
        )
        from sgl_kernel_npu.operators.attention.flash_decode_bnsd_with_gqa_share_sparse import (
            _get_native_sparse_op,
        )
        from sgl_kernel_npu.operators.utils.registry import get_impl

        get_impl(op, "ascendc")
        return _get_native_sparse_op() is not None
    except Exception:
        # Source tree without the built library, wrong generation, or no device.
        return False


def _require_ascendc():
    if not _native_available():
        pytest.skip("native npu_sparse_attention_score is unavailable here")


def _resolved(name):
    """Resolve the auto alias to the backend it would pick on this machine."""
    if name != "auto":
        return name
    return "ascendc" if _native_available() else "triton"


def _pair_supported(impl, impl_ref, native_safe, triton_safe):
    for name in (impl, impl_ref):
        resolved = _resolved(name)
        if resolved == "ascendc" and not native_safe:
            return False
        if resolved == "triton" and not triton_safe:
            return False
    return True


# Measured on arch22: the worst passing tolerance is 2e-3 for triton/
# torch_native, 1e-2 for ascendc/torch_native, 2e-2 for ascendc/triton; each
# entry keeps a 2-2.5x margin. ascendc/triton stays at the repo default because
# its gap is the Triton p->bf16 cast. Keys are sorted as _standard_for() does.
PAIR_BF16_TOLERANCE = {
    ("torch_native", "triton"): 5e-3,
    ("ascendc", "torch_native"): 2e-2,
    ("ascendc", "triton"): 5e-2,
}


def _standard_for(impl, impl_ref):
    """Per-pair bf16 tolerance; dtypes not listed fall back to the defaults."""
    key = tuple(sorted((_resolved(impl), _resolved(impl_ref))))
    tolerance = PAIR_BF16_TOLERANCE.get(key)
    if tolerance is None:
        return None
    return {"torch.bfloat16": AccuracyStandard(atol=tolerance, rtol=tolerance)}


def _skip_unsupported(impl, impl_ref, native_safe, triton_safe):
    if not _pair_supported(impl, impl_ref, native_safe, triton_safe):
        pytest.skip(f"{impl} vs {impl_ref} is outside the supported configuration")


def _page_map(capacity, batch, block_size, dtype=torch.int32, shared_requests=None):
    """Direct request map: block b starts at slot b*block_size.

    ``shared_requests`` mirrors SGLang's ``repeat_interleave``: several rows
    point at one request row through ``req_pool_indices``.
    """
    requests = batch if shared_requests is None else shared_requests
    req_to_token = torch.zeros(requests, capacity * block_size, dtype=dtype)
    req_to_token[:, ::block_size] = torch.arange(capacity, dtype=dtype) * block_size
    if shared_requests is None:
        req_pool_indices = torch.arange(batch, dtype=dtype)
    else:
        req_pool_indices = torch.arange(requests, dtype=dtype).repeat_interleave(
            batch // requests
        )
    return req_to_token, req_pool_indices


def _topk_ids(capacity, batch, topk, kv_heads, dtype):
    """Per-(kv head, row) distinct block ids; remaining slots stay -1."""
    valid = max(1, min(topk, capacity))
    topk_idx = torch.full((kv_heads, batch, topk), -1, dtype=dtype)
    for kv_head in range(kv_heads):
        for row in range(batch):
            ids = torch.randperm(capacity)[:valid].sort().values
            topk_idx[kv_head, row, :valid] = ids
    return topk_idx


def build_inputs(
    *,
    seq_lens=(16384,),
    q_heads=Q_HEADS,
    kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    block_size=BLOCK,
    topk=TOPK,
    dtype=torch.bfloat16,
    page="direct",
    index_dtype=torch.int32,
    sink=False,
    num_topk_chunks=None,
    blocks_per_step=1,
    prefetch=False,
    sanitize=False,
    shared_requests=None,
    num_pages=None,
    num_stages=None,
    seed=42,
):
    """Create one case's inputs; ``batch`` is implied by ``seq_lens``."""
    seq_lens = tuple(seq_lens)
    batch = len(seq_lens)
    capacity = max(
        1, max((length + block_size - 1) // block_size for length in seq_lens)
    )
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)

    q = torch.randn(batch, q_heads, head_dim, dtype=dtype)
    k_cache = torch.randn(capacity, block_size, kv_heads, head_dim, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    topk_idx = _topk_ids(capacity, batch, topk, kv_heads, index_dtype)

    block_table = req_to_token = req_pool_indices = None
    if page == "table":
        block_table = (
            torch.arange(capacity, dtype=index_dtype)
            .view(1, -1)
            .repeat(batch, 1)
            .contiguous()
        )
    else:
        req_to_token, req_pool_indices = _page_map(
            capacity, batch, block_size, index_dtype, shared_requests
        )

    return dict(
        q=q,
        sink=torch.randn(q_heads, head_dim, dtype=dtype) if sink else None,
        k_cache_bnsd=k_cache,
        v_cache_bnsd=v_cache,
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, dtype=index_dtype),
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
        prefetch_idx=prefetch,
        num_stages=num_stages,
    )


def _op():
    """The public entry; imported lazily so collection works without the .so."""
    from sgl_kernel_npu.operators.attention import (
        flash_decode_bnsd_with_gqa_share_sparse,
    )

    return flash_decode_bnsd_with_gqa_share_sparse


def run_case(*, perf=False, impl=None, standard=None, **params):
    executor = check_accuracy(
        _op(),
        kwargs=build_inputs(**params),
        impl=impl,
        standard=standard,
    )
    check_performance(executor, enable_perf=perf)
    return executor


def _check_hand(inputs, expected, *, standard=None, impl=None):
    """Check the closed form against the tested backend."""
    executor = check_accuracy(_op(), kwargs=inputs, impl=impl)
    assert_accuracy(executor(), expected, standard)


@pytest.mark.parametrize(
    "case_id, params, native_safe, triton_safe", PRODUCTION_CASES, ids=PRODUCTION_IDS
)
def test_production(case_id, params, native_safe, triton_safe):
    """Production shape gate: DUT (impl pair from the CLI) against the reference."""
    impl, impl_ref = get_test_impls()
    _skip_unsupported(impl, impl_ref, native_safe, triton_safe)
    run_case(
        perf=case_id in PERF_CASES,
        standard=_standard_for(impl, impl_ref),
        **params,
    )


PAIRS = [
    ("triton", "torch_native"),
    ("ascendc", "torch_native"),
    ("ascendc", "triton"),
]

PARITY_TARGETS = [
    pytest.param(impl, impl_ref, id=f"{impl}-vs-{impl_ref}") for impl, impl_ref in PAIRS
]


@pytest.mark.parametrize(
    "case_id, params, native_safe, triton_safe", PRODUCTION_CASES, ids=PRODUCTION_IDS
)
@pytest.mark.parametrize("impl, impl_ref", PARITY_TARGETS)
def test_parity_production(impl, impl_ref, case_id, params, native_safe, triton_safe):
    """Default gate: all supported backend pairs on every production shape."""
    _skip_unsupported(impl, impl_ref, native_safe, triton_safe)
    check_accuracy(
        _op(),
        kwargs=build_inputs(**params),
        impl=impl,
        impl_ref=impl_ref,
        standard=_standard_for(impl, impl_ref),
    )


# (id, build_inputs kwargs, inside the validated native configuration). The
# native op has no sink input and only D=128/block_size=128 tiling keys, so
# those cases stay on the Triton side.
EDGE_CASES = [
    ("multi-chunk", {"seq_lens": (4096, 8192), "num_topk_chunks": 4}, True, True),
    # Triton cannot compile blocks_per_step=2 at 128/128 (UB overflow).
    ("chunk-fusion", {"seq_lens": (4096, 4096), "blocks_per_step": 2}, True, False),
    ("prefetch", {"seq_lens": (4096, 8192), "prefetch": True}, True, True),
    ("page-table", {"seq_lens": (4096, 1024), "page": "table"}, True, True),
    ("fp16", {"seq_lens": (4096,), "dtype": torch.float16}, True, True),
    ("sanitize", {"seq_lens": (4096,), "sanitize": True, "num_pages": 1}, True, True),
    ("sink", {"seq_lens": (4096,), "sink": True, "num_topk_chunks": 1}, False, True),
    ("head-64", {"seq_lens": (4096,), "head_dim": 64}, False, True),
]

EDGE_IDS = [case[0] for case in EDGE_CASES]


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "case_id, params, native_safe, triton_safe", EDGE_CASES, ids=EDGE_IDS
)
def test_edge_cases(case_id, params, native_safe, triton_safe):
    """Configurations outside the production matrix, on the CLI-selected pair."""
    impl, impl_ref = get_test_impls()
    _skip_unsupported(impl, impl_ref, native_safe, triton_safe)
    run_case(standard=_standard_for(impl, impl_ref), **params)


@pytest.mark.extra_checks
def test_auto_keeps_sink_on_triton():
    """auto must not route a non-None sink to the native op, which has no sink."""
    op = _op()
    kwargs = build_inputs(seq_lens=(4096,), sink=True, topk=4, num_topk_chunks=1)
    auto_out = op(**kwargs)
    triton_out = op(**kwargs, impl="triton")
    assert_accuracy(auto_out, triton_out, EXACT)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "overrides", [{"sink": True}, {"head_dim": 64}], ids=["sink", "head-dim-64"]
)
def test_ascendc_rejects_unsupported(overrides):
    """The native op has no sink input and only D=128/block_size=128 tilings."""
    _require_ascendc()
    kwargs = build_inputs(seq_lens=(4096,), topk=4, **overrides)
    with pytest.raises(AssertionError):
        _op()(**kwargs, impl="ascendc")


@pytest.mark.extra_checks
def test_masked_slots_may_hold_nan():
    """The kernels load K/V under the position mask, so unused slots may be junk."""
    kwargs = build_inputs(seq_lens=(100,), topk=1, num_topk_chunks=1)
    kwargs["k_cache_bnsd"][:, 100:] = float("nan")
    kwargs["v_cache_bnsd"][:, 100:] = float("nan")
    executor = check_accuracy(_op(), kwargs=kwargs, impl="triton")
    assert torch.isfinite(executor()).all()


def _uniform_inputs(*, seq_len=4096, batch=2):
    """K and V are constant, so any non-empty row must return exactly V."""
    kwargs = build_inputs(seq_lens=(seq_len,) * batch, seed=7)
    k0 = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
    v0 = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
    shape = (kwargs["max_num_blocks"], BLOCK, KV_HEADS, HEAD_DIM)
    kwargs["k_cache_bnsd"] = k0.view(1, 1, 1, -1).expand(shape).contiguous()
    kwargs["v_cache_bnsd"] = v0.view(1, 1, 1, -1).expand(shape).contiguous()
    return kwargs, kwargs["q"], v0


def test_hand_uniform_kv_returns_v():
    """Constant K/V: softmax weights must sum to one on every row."""
    inputs, q, v0 = _uniform_inputs()
    expected = v0.to(torch.bfloat16).view(1, 1, -1).expand_as(q).contiguous()
    _check_hand(inputs, expected, standard=TIGHT)


def test_hand_empty_row_returns_zero():
    inputs, q, _ = _uniform_inputs(seq_len=0, batch=1)
    _check_hand(inputs, torch.zeros_like(q), standard=EXACT)
