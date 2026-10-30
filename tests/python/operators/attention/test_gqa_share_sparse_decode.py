"""Compare sparse attention with a direct gathered-token reference."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy


@pytest.mark.parametrize("sink_enabled", [False, True])
@pytest.mark.parametrize("chunks", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gqa_share_sparse_decode(sink_enabled, chunks, dtype):
    from sgl_kernel_npu.operators.attention import gqa_share_sparse_decode

    q = torch.randn(2, 4, 64, dtype=dtype) * 0.1
    k = torch.randn(4, 16, 2, 64, dtype=dtype) * 0.1
    v = torch.randn_like(k)
    sink = torch.randn(4, 64, dtype=dtype) * 0.1 if sink_enabled else None
    block_table = torch.tensor([[2, 0, 1], [0, 3, 1]], dtype=torch.int32)
    lengths = torch.tensor([35, 20], dtype=torch.int32)
    topk = torch.tensor(
        [[[0, 1, 2, -1], [0, 1, -1, -1]], [[0, 1, 2, -1], [0, 1, -1, -1]]],
        dtype=torch.int32,
    )
    check_accuracy(
        gqa_share_sparse_decode,
        args=(q, sink, k, v, block_table, lengths, 16, topk),
        kwargs={"num_topk_chunks": chunks},
    )
