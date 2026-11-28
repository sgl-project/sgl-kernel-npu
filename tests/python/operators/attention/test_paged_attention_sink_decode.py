"""Paged decode attention with per-head sink logits."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# Profiled shapes, keyed by (batch, max_kv_len, page_size, q_heads, kv_heads,
# head_dim): a long single-request MQA decode that takes the KV-split path and
# a batched gpt-oss-120b TP1 decode.
perf_shape = [
    (1, 4096, 128, 8, 1, 64),
    (8, 1000, 128, 64, 8, 64),
]

CASES = [
    pytest.param(
        dict(
            kv_lens=(4096,),
            page_size=128,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=64,
            slack_pages=0,
            hint=True,
        ),
        id="mqa-long-kv-split",
    ),
    pytest.param(
        dict(
            kv_lens=(2048,),
            page_size=128,
            num_q_heads=2,
            num_kv_heads=2,
            head_dim=64,
            slack_pages=0,
        ),
        id="mha-long-kv-split",
    ),
    pytest.param(
        dict(
            kv_lens=(1, 33, 64, 129, 200, 257, 512, 1000),
            page_size=128,
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=64,
            slack_pages=0,
        ),
        id="gpt-oss-120b-tp1-batched",
    ),
    pytest.param(
        dict(
            kv_lens=(1,),
            page_size=128,
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=64,
            slack_pages=0,
        ),
        id="single-token-minimum",
    ),
    pytest.param(
        dict(
            kv_lens=(129, 300),
            page_size=64,
            num_q_heads=16,
            num_kv_heads=4,
            head_dim=64,
            slack_pages=1,
        ),
        id="page64-tail-and-table-slack",
    ),
    pytest.param(
        dict(
            kv_lens=(1,) * 8,
            page_size=16,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=32,
            slack_pages=0,
        ),
        id="mqa-page16-minimum",
    ),
]


def generate_data(case, dtype, sinks_mode, gqa_interleave):
    """Build decode queries, paged K/V, page tables and KV lengths for one case."""
    torch.npu.manual_seed(42)
    kv_lens = case["kv_lens"]
    page_size = case["page_size"]
    batch = len(kv_lens)
    max_pages = max((length + page_size - 1) // page_size for length in kv_lens)
    max_pages += case["slack_pages"]
    total_pages = batch * max_pages

    q = torch.randn(batch, case["num_q_heads"], case["head_dim"], dtype=dtype)
    k_cache = torch.randn(
        total_pages, page_size, case["num_kv_heads"], case["head_dim"], dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.randperm(total_pages).to(torch.int32)
    block_tables = block_tables.reshape(batch, max_pages)
    seqlens = torch.tensor(kv_lens, dtype=torch.int32)
    sinks = (
        torch.randn(case["num_q_heads"], dtype=dtype) if sinks_mode == "sink" else None
    )
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        seqlens=seqlens,
        block_tables=block_tables,
        sinks=sinks,
        gqa_interleave=gqa_interleave,
        softmax_scale=None,
        max_kv_len=max(kv_lens) if case.get("hint") else None,
    )


def perf_key(case):
    return (
        len(case["kv_lens"]),
        max(case["kv_lens"]),
        case["page_size"],
        case["num_q_heads"],
        case["num_kv_heads"],
        case["head_dim"],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sinks_mode", ["sink", "no-sink"])
@pytest.mark.parametrize("gqa_interleave", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_paged_attention_sink_decode(case, gqa_interleave, sinks_mode, dtype):
    from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

    kwargs = generate_data(case, dtype, sinks_mode, gqa_interleave)
    executor = check_accuracy(paged_attention_sink_decode, kwargs=kwargs)
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
        pytest.param("lengths", "int32", id="length-dtype"),
        pytest.param("hint", "positive integer or None", id="bad-max-kv-len"),
    ],
)
def test_paged_attention_sink_decode_rejects_bad_metadata(mutation, message):
    from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

    case = dict(
        kv_lens=(5,),
        page_size=128,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=64,
        slack_pages=0,
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
    elif mutation == "lengths":
        kwargs["seqlens"] = kwargs["seqlens"].to(torch.int64)
    elif mutation == "hint":
        kwargs["max_kv_len"] = 0

    with pytest.raises((ValueError, TypeError), match=message):
        paged_attention_sink_decode(**kwargs)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_paged_attention_sink_decode_unaligned_tiles(dtype):
    """Non-power-of-two head_dim and GQA group (validated on arch35)."""
    from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

    case = dict(
        kv_lens=(1, 65, 129, 300),
        page_size=128,
        num_q_heads=12,
        num_kv_heads=4,
        head_dim=96,
        slack_pages=0,
    )
    kwargs = generate_data(case, dtype, "sink", False)
    check_accuracy(paged_attention_sink_decode, kwargs=kwargs)


@pytest.mark.extra_checks
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_paged_attention_sink_decode_long_context(dtype):
    """Long KV rows with an under-filled batch take the multi-split path."""
    from sgl_kernel_npu.operators.attention import paged_attention_sink_decode

    case = dict(
        kv_lens=(32768, 8192),
        page_size=64,
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=64,
        slack_pages=0,
    )
    kwargs = generate_data(case, dtype, "sink", False)
    check_accuracy(paged_attention_sink_decode, kwargs=kwargs)
