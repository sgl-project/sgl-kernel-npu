import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.performance import check_performance

PAD_ROWS = 13
# W: row-count candidates; P/C: pool width 512 and output width 2051.
perf_shape = [(4096, 4096)]


def generate_test_data(n_real, n_pool_topk=512, pool_size=4):
    torch.manual_seed(20)
    index_topk = n_pool_topk * pool_size
    total_rows = n_real + PAD_ROWS
    pool_indices = torch.randint(0, 10000, (total_rows, n_pool_topk), dtype=torch.int32)
    seq_lens_per_token = torch.arange(
        1, total_rows + 1, dtype=torch.int32
    ) * pool_size + torch.randint(0, pool_size, (total_rows,), dtype=torch.int32)
    return pool_indices, seq_lens_per_token, pool_size, index_topk


@pytest.mark.parametrize(
    "n_real,num_q_padded,n_pool_topk,pool_size",
    [
        (4095, 4096, 512, 4),
        pytest.param(4096, 4096, 512, 4, id="W-glm53-flash-r4096-q4096"),
        pytest.param(1, 4, 1, 2, id="edge-one-pool-padding"),
        pytest.param(5, 8, 8, 2, id="edge-small-topk-pool2"),
        pytest.param(7, 8, 16, 8, id="edge-pool8-tail"),
    ],
)
def test_expand_pool_topk_and_append_tail_batched_npu(
    n_real, num_q_padded, n_pool_topk, pool_size
):
    from sgl_kernel_npu.operators.indexer import (
        expand_pool_topk_and_append_tail_batched_npu,
    )

    pool_indices, seq_lens, pool_size, index_topk = generate_test_data(
        n_real, n_pool_topk, pool_size
    )
    executor = check_accuracy(
        expand_pool_topk_and_append_tail_batched_npu,
        args=(pool_indices, seq_lens, n_real, num_q_padded, pool_size, index_topk),
    )
    check_performance(
        executor,
        enable_perf=(n_real, num_q_padded) in perf_shape
        and n_pool_topk == 512
        and pool_size == 4,
    )


def test_expand_rejects_noncontiguous_seq_lens():
    from sgl_kernel_npu.operators.indexer import (
        expand_pool_topk_and_append_tail_batched_npu,
    )

    pool_indices = torch.zeros(2, 4, dtype=torch.int32)
    base = torch.tensor([8, 8], dtype=torch.int32)
    seq_lens = torch.stack((base, base), dim=1)[:, 0]
    assert not seq_lens.is_contiguous()
    with pytest.raises(AssertionError, match="seq_lens_per_token must be contiguous"):
        expand_pool_topk_and_append_tail_batched_npu(
            pool_indices, seq_lens, 2, 2, 4, 16
        )
