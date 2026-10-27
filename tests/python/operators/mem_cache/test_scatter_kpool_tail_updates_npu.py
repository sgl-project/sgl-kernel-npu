from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# W: Q/Nt workload candidates; tail_rows=8 is an independent synthetic value.
perf_shape = [(4096, 1024)]

EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float16": AccuracyStandard(atol=0, rtol=0),
}


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
        (128, 640, 256, 4, 4, 128, torch.bfloat16),
        (4, 32, 8, 4, 8, 128, torch.float16),
        pytest.param(1024, 4096, 1024, 4, 8, 128, torch.bfloat16, id="W-q4096-nt1024"),
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
    pool = SimpleNamespace(index_kpool=pool_size)
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(scatter_kpool_tail_updates_npu, impl, impl_ref)
    else:
        require_test_impl(scatter_kpool_tail_updates_npu, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        tail_k_ref = tail_k.clone()
        tail_score_ref = tail_score.clone()
        scatter_kpool_tail_updates_npu(
            pool,
            chunk_k,
            chunk_score,
            tail_k_ref,
            tail_score_ref,
            req_pool_idx,
            dst_logical_start,
            chunk_src_start,
            n_write,
            impl=impl_ref,
        )
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
            impl=impl,
        )

        assert_accuracy(tail_k, tail_k_ref, EXACT_STANDARD, path="tail_k")
        assert_accuracy(tail_score, tail_score_ref, EXACT_STANDARD, path="tail_score")

        written = torch.zeros(tail_k.shape[:2], dtype=torch.bool, device=tail_k.device)
        for row in range(req_pool_idx.numel()):
            req = int(req_pool_idx[row].item())
            start = int(dst_logical_start[row].item())
            count = int(n_write[row].item())
            for offset in range(count):
                written[req, (start + offset) % tail_k.shape[1]] = True

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

    snapshots = None
    call_index = 0

    def executor():
        nonlocal snapshots, call_index
        if snapshots is None:
            snapshots = [
                (initial_tail_k.clone(), initial_tail_score.clone()) for _ in range(11)
            ]
        perf_tail_k, perf_tail_score = snapshots[call_index]
        call_index += 1
        scatter_kpool_tail_updates_npu(
            pool,
            chunk_k,
            chunk_score,
            perf_tail_k,
            perf_tail_score,
            req_pool_idx,
            dst_logical_start,
            chunk_src_start,
            n_write,
            impl=impl,
        )

    executor.func = scatter_kpool_tail_updates_npu
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=(total_tokens, n_rows) in perf_shape)


def test_scatter_rejects_noncontiguous_tail():
    from sgl_kernel_npu.operators.mem_cache import scatter_kpool_tail_updates_npu

    inputs = list(_create_test_inputs(1, 4, 1, 4, 4, 128, torch.bfloat16))
    tail_k = inputs[2]
    inputs[2] = torch.stack((tail_k, tail_k), dim=-1)[..., 0]
    assert not inputs[2].is_contiguous()
    pool = SimpleNamespace(index_kpool=4)
    with pytest.raises(AssertionError, match="tail_k must be contiguous"):
        scatter_kpool_tail_updates_npu(pool, *inputs)
