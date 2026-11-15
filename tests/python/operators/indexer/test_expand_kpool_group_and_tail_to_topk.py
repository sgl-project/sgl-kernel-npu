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
        # Case 01: 4095 valid rows plus one padded row.
        (4095, 4096, 512, 4),
        # Case 02: Typical 4096 valid rows without padding; also profile.
        pytest.param(4096, 4096, 512, 4, id="W-glm53-flash-r4096-q4096"),
        # Model inputs are expected to have n_pool_topk >= 256 and index_topk >= 2048;
        # these guide test coverage rather than define operator support limits.
        # Case 03: Lower end of model inputs, with a partial eight-row block.
        pytest.param(1, 4, 256, 8, id="model-small-topk-block-b8"),
        # Case 04: Minimum pool size, with a partial two-row block.
        pytest.param(5, 8, 1024, 4, id="topk4096-pool4-block-b2"),
        # Case 05: Pool8 expansion with a tail and padding.
        pytest.param(7, 8, 512, 8, id="topk4096-pool8-block-b4"),
        # Case 06: Top-k8192/pool4 chooses one row and retains padded rows.
        pytest.param(17, 20, 2048, 4, id="topk8192-pool4-block-b1"),
    ],
)
def test_expand_kpool_group_and_tail_to_topk(
    n_real, num_q_padded, n_pool_topk, pool_size
):
    from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk

    pool_indices, seq_lens, pool_size, index_topk = generate_test_data(
        n_real, n_pool_topk, pool_size
    )
    executor = check_accuracy(
        expand_kpool_group_and_tail_to_topk,
        args=(pool_indices, seq_lens, n_real, num_q_padded, pool_size, index_topk),
    )
    check_performance(
        executor,
        enable_perf=(n_real, num_q_padded) in perf_shape
        and n_pool_topk == 512
        and pool_size == 4,
    )


# Case 08: Reject noncontiguous sequence lengths.
@pytest.mark.extra_checks
def test_expand_rejects_noncontiguous_seq_lens():
    from sgl_kernel_npu.operators.indexer import expand_kpool_group_and_tail_to_topk

    pool_indices = torch.zeros(2, 512, dtype=torch.int32)
    base = torch.tensor([8, 8], dtype=torch.int32)
    seq_lens = torch.stack((base, base), dim=1)[:, 0]
    assert not seq_lens.is_contiguous()
    with pytest.raises(AssertionError, match="seq_lens_per_token must be contiguous"):
        expand_kpool_group_and_tail_to_topk(pool_indices, seq_lens, 2, 2, 4, 2048)
