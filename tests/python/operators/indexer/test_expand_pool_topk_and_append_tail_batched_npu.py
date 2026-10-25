import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy

PAD_ROWS = 13


def _expand_pool_topk_and_append_tail_batched_torch(
    pool_indices,
    seq_lens_per_token,
    n_real,
    num_q_padded,
    pool_size=4,
):
    """Independent PyTorch expansion and uncompressed-tail reference."""
    n_pool_topk = pool_indices.shape[1]
    index_topk = n_pool_topk * pool_size
    tail_pool = pool_size - 1
    n_real = min(n_real, pool_indices.shape[0], seq_lens_per_token.shape[0])

    offsets = torch.arange(pool_size, dtype=torch.int32)
    token_indices = (
        pool_indices[:n_real].to(torch.int32).unsqueeze(-1) * pool_size + offsets
    )
    token_indices = token_indices.reshape(n_real, index_topk)

    out_cols = index_topk + tail_pool
    out = torch.full((num_q_padded, out_cols), -1, dtype=torch.int32)
    out[:n_real, :index_topk] = token_indices

    seq_lens_used = seq_lens_per_token[:n_real]
    pool_lens_per_token = torch.div(seq_lens_used, pool_size, rounding_mode="floor").to(
        torch.int32
    )
    tail_starts = pool_lens_per_token * pool_size
    tail_counts = (seq_lens_used - tail_starts).to(torch.int32)

    for index in range(tail_pool):
        mask = tail_counts > index
        values = (tail_starts + index).to(torch.int32)
        out[:n_real, index_topk + index] = torch.where(
            mask, values, torch.full_like(values, -1)
        )
    return out


@pytest.mark.parametrize(
    "n_real,num_q_padded",
    [
        (2048, 2048),
        (1024, 2048),
        (1035, 2048),
        (4095, 4096),
    ],
)
def test_expand_pool_topk_and_append_tail_batched_npu(n_real, num_q_padded):
    from sgl_kernel_npu.operators.indexer import (
        expand_pool_topk_and_append_tail_batched_npu,
    )

    torch.manual_seed(20)
    n_pool_topk = 512
    pool_size = 4
    index_topk = n_pool_topk * pool_size
    total_rows = n_real + PAD_ROWS
    pool_indices = torch.randint(0, 10000, (total_rows, n_pool_topk), dtype=torch.int32)
    seq_lens_per_token = torch.arange(
        1, total_rows + 1, dtype=torch.int32
    ) * pool_size + torch.randint(0, pool_size, (total_rows,), dtype=torch.int32)

    actual = expand_pool_topk_and_append_tail_batched_npu(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
        index_topk,
    )
    expected = _expand_pool_topk_and_append_tail_batched_torch(
        pool_indices,
        seq_lens_per_token,
        n_real,
        num_q_padded,
        pool_size,
    )

    # Token indices and padded rows are integers and must match exactly.
    assert_accuracy(actual, expected)
