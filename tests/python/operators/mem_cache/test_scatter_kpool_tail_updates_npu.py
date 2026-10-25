from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.standards import AccuracyStandard

EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float16": AccuracyStandard(atol=0, rtol=0),
}


def _scatter_kpool_tail_updates_torch(
    pool_size,
    chunk_k,
    chunk_score,
    tail_k,
    tail_score,
    req_pool_idx,
    dst_logical_start,
    chunk_src_start,
    n_write,
):
    """Pure Torch reference for the scatter kernel, without GPU Triton."""
    slots = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)
    write_slots = slots < n_write.to(torch.int64).unsqueeze(1)
    requests = req_pool_idx.to(torch.int64).unsqueeze(1).expand(-1, pool_size)
    destinations = (
        dst_logical_start.to(torch.int64).unsqueeze(1) + slots
    ) % tail_k.shape[1]
    sources = chunk_src_start.to(torch.int64).unsqueeze(1) + slots

    selected_requests = requests[write_slots]
    selected_destinations = destinations[write_slots]
    selected_sources = sources[write_slots]
    tail_k[selected_requests, selected_destinations] = chunk_k[selected_sources]
    tail_score[selected_requests, selected_destinations] = chunk_score[selected_sources]

    written = torch.zeros(tail_k.shape[:2], dtype=torch.bool)
    written[selected_requests, selected_destinations] = True
    return written


def _create_test_inputs(
    n_rows,
    total_tokens,
    req_pool_size,
    pool_size,
    tail_size,
    head_dim,
    dtype,
    seed=20,
):
    torch.manual_seed(seed)

    chunk_k = torch.empty(total_tokens, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    chunk_score = torch.empty(total_tokens, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    tail_k = torch.empty(req_pool_size, tail_size, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )
    tail_score = torch.empty(req_pool_size, tail_size, head_dim, dtype=dtype).normal_(
        mean=0.0, std=0.5
    )

    assert req_pool_size >= n_rows
    req_pool_idx = torch.arange(n_rows, dtype=torch.int64)
    dst_logical_start = torch.randint(0, tail_size, (n_rows,), dtype=torch.int32)
    n_write = torch.randint(1, pool_size + 1, (n_rows,), dtype=torch.int32)

    cumsum = torch.cumsum(n_write.to(torch.int64), dim=0)
    chunk_src_start = torch.zeros(n_rows, dtype=torch.int64)
    chunk_src_start[1:] = cumsum[:-1]
    assert chunk_src_start[-1] + n_write[-1] <= total_tokens

    return (
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )


@pytest.mark.parametrize(
    "n_rows,total_tokens,req_pool_size,pool_size,tail_size,head_dim,dtype",
    [
        (1, 16, 4, 4, 4, 128, torch.bfloat16),
        (8, 64, 16, 4, 8, 128, torch.bfloat16),
        (8, 64, 16, 4, 12, 128, torch.bfloat16),
        (64, 320, 128, 4, 8, 128, torch.bfloat16),
        (128, 640, 256, 4, 4, 128, torch.bfloat16),
        (128, 640, 256, 4, 12, 128, torch.bfloat16),
        (4, 32, 8, 4, 8, 128, torch.float16),
    ],
)
def test_scatter_kpool_tail_updates_npu(
    n_rows,
    total_tokens,
    req_pool_size,
    pool_size,
    tail_size,
    head_dim,
    dtype,
):
    from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

    (
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    ) = _create_test_inputs(
        n_rows,
        total_tokens,
        req_pool_size,
        pool_size,
        tail_size,
        head_dim,
        dtype,
    )
    initial_tail_k = tail_k.clone()
    initial_tail_score = tail_score.clone()
    tail_k_ref = tail_k.clone()
    tail_score_ref = tail_score.clone()

    written = _scatter_kpool_tail_updates_torch(
        pool_size,
        chunk_k,
        chunk_score,
        tail_k_ref,
        tail_score_ref,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )
    pool = SimpleNamespace(index_kpool=pool_size)
    scatter_kpool_tail_updates_npu(
        pool,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        dst_logical_start,
        chunk_src_start,
        n_write,
    )

    assert_accuracy(tail_k, tail_k_ref, EXACT_STANDARD, path="tail_k")
    assert_accuracy(tail_score, tail_score_ref, EXACT_STANDARD, path="tail_score")

    unchanged = ~written
    assert_accuracy(
        tail_k[unchanged],
        initial_tail_k[unchanged],
        EXACT_STANDARD,
        path="tail_k_unchanged",
    )
    assert_accuracy(
        tail_score[unchanged],
        initial_tail_score[unchanged],
        EXACT_STANDARD,
        path="tail_score_unchanged",
    )
