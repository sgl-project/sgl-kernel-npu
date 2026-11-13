import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

perf_shape = [(1, 128), (17, 256)]


def generate_data(query_len, context_len, prefill, dtype):
    torch.manual_seed(42)
    batch, q_heads, kv_heads, dim, page_size = 2, 64, 8, 64, 128
    pages = (context_len + page_size - 1) // page_size
    query = torch.randn(
        batch * query_len if prefill else batch, q_heads * dim, dtype=dtype
    )
    k = torch.randn(batch * pages, page_size, kv_heads, dim, dtype=dtype)
    v = torch.randn_like(k)
    sinks = torch.randn(q_heads, dtype=torch.float32)
    table = torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages)
    context = torch.full((batch,), context_len, dtype=torch.int32)
    lengths = torch.full((batch,), query_len, dtype=torch.int32)
    return query, k, v, sinks, lengths, table, context


@pytest.mark.parametrize(
    "query_len,context_len",
    [
        pytest.param(1, 128, id="gpt-oss-decode"),
        pytest.param(17, 256, id="gpt-oss-prefill-tail"),
        pytest.param(32, 129, id="page-tail"),
    ],
)
@pytest.mark.parametrize("prefill", [True, False])
@pytest.mark.parametrize("window", [-1, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_sinks(query_len, context_len, prefill, window, dtype):
    from sgl_kernel_npu.operators.attention import (
        attention_sinks,
        attention_sinks_prefill,
    )

    q, k, v, sinks, lengths, table, context = generate_data(
        query_len, context_len, prefill, dtype
    )
    args = (
        (q, k, v, sinks)
        + ((lengths,) if prefill else ())
        + (table, context, 64**-0.5, window, 64, 8)
    )
    executor = check_accuracy(
        attention_sinks_prefill if prefill else attention_sinks, args=args
    )
    check_performance(executor, enable_perf=(query_len, context_len) in perf_shape)
