"""Compare sparse attention with a direct gathered-token reference."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

# MiniMax NPU uses page-size-dependent sparse decode with runtime head counts.
# The larger case uses block=128/topk=16 from that path; batch/heads are synthetic.
perf_shape = [(2, 8, 1, 128, 128, 2048, 32)]


def generate_test_data(shape, sink_enabled, chunks, dtype):
    batch, query_heads, kv_heads, dim, block_size, context, width = shape
    pages_per_request = (context + block_size - 1) // block_size
    num_pages = batch * pages_per_request
    q = torch.randn(batch, query_heads, dim, dtype=dtype) * 0.1
    k = torch.randn(num_pages, block_size, kv_heads, dim, dtype=dtype) * 0.1
    v = torch.randn_like(k)
    sink = torch.randn(query_heads, dim, dtype=dtype) * 0.1 if sink_enabled else None
    block_table = (
        torch.randperm(num_pages).reshape(batch, pages_per_request).to(torch.int32)
    )
    lengths = torch.tensor([context, context - block_size + 1], dtype=torch.int32)
    topk = torch.full((kv_heads, batch, width), -1, dtype=torch.int32)
    for row in range(batch):
        valid = (int(lengths[row].item()) + block_size - 1) // block_size
        topk[:, row, :valid] = torch.randperm(valid).to(torch.int32)
    return q, sink, k, v, block_table, lengths, topk, block_size


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 2, 64, 16, 35, 4), *perf_shape],
    ids=["edge-page16", "minimax-page128-candidate"],
)
@pytest.mark.parametrize("sink_enabled", [False, True])
@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gqa_share_sparse_decode(shape, sink_enabled, chunks, dtype):
    from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

    q, sink, k, v, block_table, lengths, topk, block_size = generate_test_data(
        shape, sink_enabled, chunks, dtype
    )

    executor = check_accuracy(
        gqa_share_sparse_decode,
        args=(q, sink, k, v, block_table, lengths, block_size, topk),
        kwargs={"num_topk_chunks": chunks},
    )
    check_performance(
        executor,
        enable_perf=(
            shape in perf_shape
            and dtype == torch.bfloat16
            and not sink_enabled
            and chunks == 1
        ),
    )
