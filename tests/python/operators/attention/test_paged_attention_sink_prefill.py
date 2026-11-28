"""Causal paged prefill attention with per-head sink logits."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Profiled shapes, keyed by (batch, q_rows, max_kv_len, page_size, q_heads,
# kv_heads, head_dim): a full gpt-oss-120b prefill at TP1 and a two-request
# chunked prefill with a cached prefix.
perf_shape = [
    (1, 512, 512, 128, 64, 8, 64),
    (2, 146, 1153, 128, 64, 8, 64),
]

# ``implicit`` omits seqlens_kv, so each KV length equals its query length.
# ``explicit`` passes the prefixed KV length. q_lens/prefix_lens are per request.
CASES = [
    pytest.param(
        dict(
            q_lens=(512,),
            prefix_lens=(0,),
            page_size=128,
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=64,
            seqlens_mode="implicit",
            softmax_scale=None,
        ),
        id="gpt-oss-120b-tp1-prefill",
    ),
    pytest.param(
        dict(
            q_lens=(129, 17),
            prefix_lens=(1024, 112),
            page_size=128,
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=64,
            seqlens_mode="explicit",
            softmax_scale=0.125,
        ),
        id="gpt-oss-120b-tp1-chunked",
    ),
    pytest.param(
        dict(
            q_lens=(200, 33),
            prefix_lens=(257, 0),
            page_size=64,
            num_q_heads=32,
            num_kv_heads=4,
            head_dim=128,
            seqlens_mode="explicit",
            softmax_scale=None,
        ),
        id="page-aggregation-page64",
    ),
    pytest.param(
        dict(
            q_lens=(300,),
            prefix_lens=(100,),
            page_size=256,
            num_q_heads=16,
            num_kv_heads=16,
            head_dim=64,
            seqlens_mode="explicit",
            softmax_scale=None,
        ),
        id="mha-page256",
    ),
    pytest.param(
        dict(
            q_lens=(9,),
            prefix_lens=(0,),
            page_size=16,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=32,
            seqlens_mode="implicit",
            softmax_scale=None,
        ),
        id="mqa-page16-tail",
    ),
    pytest.param(
        dict(
            q_lens=(1, 128, 129),
            prefix_lens=(0, 0, 1),
            page_size=128,
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=64,
            seqlens_mode="explicit",
            softmax_scale=None,
        ),
        id="uneven-packed-requests",
    ),
]


def generate_data(case, dtype, sinks_mode, gqa_interleave):
    """Build packed queries, paged K/V, page tables and lengths for one case."""
    torch.npu.manual_seed(42)
    q_lens = case["q_lens"]
    prefix_lens = case["prefix_lens"]
    page_size = case["page_size"]
    kv_lens = [query + prefix for query, prefix in zip(q_lens, prefix_lens)]
    max_pages = max((length + page_size - 1) // page_size for length in kv_lens)
    total_pages = len(q_lens) * max_pages

    q = torch.randn(sum(q_lens), case["num_q_heads"], case["head_dim"], dtype=dtype)
    k_cache = torch.randn(
        total_pages, page_size, case["num_kv_heads"], case["head_dim"], dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.randperm(total_pages).to(torch.int32)
    block_tables = block_tables.reshape(len(q_lens), max_pages)

    boundaries = [0]
    for length in q_lens:
        boundaries.append(boundaries[-1] + length)
    cu_q_lens = torch.tensor(boundaries, dtype=torch.int32)
    seqlens_kv = (
        torch.tensor(kv_lens, dtype=torch.int32)
        if case["seqlens_mode"] == "explicit"
        else None
    )
    sinks = (
        torch.randn(case["num_q_heads"], dtype=dtype) if sinks_mode == "sink" else None
    )
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_q_lens=cu_q_lens,
        block_tables=block_tables,
        seqlens_kv=seqlens_kv,
        sinks=sinks,
        gqa_interleave=gqa_interleave,
        softmax_scale=case["softmax_scale"],
    )


def perf_key(case):
    return (
        len(case["q_lens"]),
        sum(case["q_lens"]),
        max(
            query + prefix for query, prefix in zip(case["q_lens"], case["prefix_lens"])
        ),
        case["page_size"],
        case["num_q_heads"],
        case["num_kv_heads"],
        case["head_dim"],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sinks_mode", ["sink", "no-sink"])
@pytest.mark.parametrize("gqa_interleave", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_paged_attention_sink_prefill(case, gqa_interleave, sinks_mode, dtype):
    from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

    kwargs = generate_data(case, dtype, sinks_mode, gqa_interleave)
    executor = check_accuracy(paged_attention_sink_prefill, kwargs=kwargs)
    check_performance(
        executor,
        enable_perf=(
            perf_key(case) in perf_shape
            and dtype == torch.bfloat16
            and sinks_mode == "sink"
            and not gqa_interleave
        ),
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        pytest.param("dtype", "float16 or bfloat16", id="unsupported-dtype"),
        pytest.param("head_dim", "head dimension in", id="head-dim-15"),
        pytest.param("page_size", "power-of-two page size", id="page-size-24"),
        pytest.param("q_heads", "divisible by KV heads", id="head-count-mismatch"),
        pytest.param("rows", "one page-table row per request", id="page-rows"),
    ],
)
def test_paged_attention_sink_prefill_rejects_bad_metadata(mutation, message):
    from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

    case = dict(
        q_lens=(5,),
        prefix_lens=(0,),
        page_size=128,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        seqlens_mode="implicit",
        softmax_scale=None,
    )
    kwargs = generate_data(case, torch.bfloat16, "sink", False)
    if mutation == "dtype":
        kwargs["q"] = kwargs["q"].float()
        kwargs["k_cache"] = kwargs["k_cache"].float()
        kwargs["v_cache"] = kwargs["v_cache"].float()
        kwargs["sinks"] = kwargs["sinks"].float()
    elif mutation == "head_dim":
        kwargs["q"] = kwargs["q"][..., :15].contiguous()
        kwargs["k_cache"] = kwargs["k_cache"][..., :15].contiguous()
        kwargs["v_cache"] = kwargs["v_cache"][..., :15].contiguous()
    elif mutation == "page_size":
        kwargs["k_cache"] = kwargs["k_cache"][:, :24].contiguous()
        kwargs["v_cache"] = kwargs["v_cache"][:, :24].contiguous()
    elif mutation == "q_heads":
        kwargs["q"] = kwargs["q"][:, :7].contiguous()
    elif mutation == "rows":
        kwargs["block_tables"] = kwargs["block_tables"][:0]
    with pytest.raises((ValueError, TypeError), match=message):
        paged_attention_sink_prefill(**kwargs)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_paged_attention_sink_prefill_unaligned_tiles(dtype):
    """Non-power-of-two head_dim and GQA group (validated on arch35)."""
    from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

    case = dict(
        q_lens=(33, 129),
        prefix_lens=(128, 0),
        page_size=64,
        num_q_heads=6,
        num_kv_heads=2,
        head_dim=96,
        seqlens_mode="explicit",
        softmax_scale=None,
    )
    kwargs = generate_data(case, dtype, "sink", False)
    check_accuracy(paged_attention_sink_prefill, kwargs=kwargs)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_paged_attention_sink_prefill_long_context(dtype):
    """Imbalanced long-context batch: LPT scheduling plus page aggregation.

    One 4k-row query over a 16k-token prefix dominates the cost estimate while
    two short requests fill the remaining cores; page_size=64 routes the batch
    through the page-aggregation kernel.
    """
    from sgl_kernel_npu.operators.attention import paged_attention_sink_prefill

    case = dict(
        q_lens=(4096, 97, 1),
        prefix_lens=(16384, 0, 2048),
        page_size=64,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        seqlens_mode="explicit",
        softmax_scale=None,
    )
    kwargs = generate_data(case, dtype, "sink", False)
    check_accuracy(paged_attention_sink_prefill, kwargs=kwargs)
