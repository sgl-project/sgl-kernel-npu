from typing import Optional

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# W: Q candidates; Np is synthetic from full chunks, not a traced value. P/C: K=4.
perf_shape = [(4096, 1024), (8192, 2048)]

HEAD_DIM = 128
PAGE_SIZE = 128
CACHE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}
EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
}


@torch.no_grad()
def kpool_assemble_softmax_write_cache_torch(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
) -> None:
    """Torch numerical reference for BF16 KPool assembly and cache write."""
    n_pools = req_pool_idx.numel()
    if n_pools == 0:
        return

    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    if write_mask is None:
        active_rows = torch.arange(n_pools, dtype=torch.int64)
    else:
        active_rows = torch.nonzero(write_mask, as_tuple=False).flatten()

    req = req_pool_idx.index_select(0, active_rows)
    n_tail = n_from_tail.index_select(0, active_rows).to(torch.int64)
    chunk_src = chunk_src_start.index_select(0, active_rows)
    tail_base = tail_logical_base.index_select(0, active_rows).to(torch.int64)

    slots = torch.arange(pool_size, dtype=torch.int64).unsqueeze(0)
    use_tail = slots < n_tail.unsqueeze(1)
    tail_phys = torch.remainder(tail_base.unsqueeze(1) + slots, tail_size)
    tail_req = req.unsqueeze(1).expand_as(tail_phys)
    assembled_tail_k = tail_k[tail_req, tail_phys]
    assembled_tail_score = tail_score[tail_req, tail_phys]

    # torch.where evaluates both branches, so keep unselected indices valid.
    chunk_offsets = torch.clamp(slots - n_tail.unsqueeze(1), min=0)
    chunk_indices = chunk_src.unsqueeze(1) + chunk_offsets
    assembled_chunk_k = chunk_k[chunk_indices]
    assembled_chunk_score = chunk_score[chunk_indices]
    assembled_k = torch.where(
        use_tail.unsqueeze(-1), assembled_tail_k, assembled_chunk_k
    )
    assembled_score = torch.where(
        use_tail.unsqueeze(-1), assembled_tail_score, assembled_chunk_score
    )

    logits = assembled_score.float() + ape.float().unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    pooled = torch.sum(assembled_k.float() * probs, dim=1).to(torch.bfloat16)
    cache_2d = index_k_cache.view(-1, HEAD_DIM)
    active_locs = loc.index_select(0, active_rows)
    cache_2d.index_copy_(0, active_locs, pooled)


def _make_noncontiguous(tensor):
    if tensor.ndim == 1:
        result = torch.stack((tensor, tensor), dim=1)[:, 0]
    else:
        result = tensor.transpose(0, 1).contiguous().transpose(0, 1)
    assert result.shape == tensor.shape and not result.is_contiguous()
    return result


def _make_inputs(
    *,
    seed,
    n_pools,
    pool_size,
    tail_size,
    n_from_tail_values,
    score_dtype,
    use_write_mask,
    total_tokens=None,
):
    assert len(n_from_tail_values) == n_pools
    assert all(0 <= value <= pool_size for value in n_from_tail_values)
    assert max(n_from_tail_values) <= tail_size

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    chunk_src_starts = []
    chunk_rows = 0
    for n_tail in n_from_tail_values:
        chunk_src_starts.append(chunk_rows)
        chunk_rows += pool_size - n_tail

    if total_tokens is None:
        # Keep one sentinel row valid for pools assembled entirely from tail state.
        chunk_rows += 1
    else:
        chunk_rows = total_tokens
        chunk_src_starts = [row * (total_tokens // n_pools) for row in range(n_pools)]
        assert all(
            start + pool_size - n_tail <= total_tokens
            for start, n_tail in zip(chunk_src_starts, n_from_tail_values)
        )
    n_requests = max(4, (n_pools + 1) // 2)
    cache_pages = max(2, (n_pools + PAGE_SIZE) // PAGE_SIZE)
    cache_rows = cache_pages * PAGE_SIZE

    def randn(shape, dtype):
        return torch.randn(shape, dtype=torch.float32).to(dtype)

    initial_cache = randn((cache_pages, PAGE_SIZE, 1, HEAD_DIM), torch.bfloat16)
    chunk_k = randn((chunk_rows, HEAD_DIM), torch.bfloat16)
    tail_k = randn((n_requests, tail_size, HEAD_DIM), torch.bfloat16)
    chunk_score = randn((chunk_rows, HEAD_DIM), score_dtype) * 4
    tail_score = randn((n_requests, tail_size, HEAD_DIM), score_dtype) * 4
    ape = randn((pool_size, HEAD_DIM), torch.float32) * 2

    req_pool_idx = torch.tensor(
        [(row * 3 + 1) % n_requests for row in range(n_pools)],
        dtype=torch.int64,
    )
    n_from_tail = torch.tensor(n_from_tail_values, dtype=torch.int32)
    chunk_src_start = torch.tensor(chunk_src_starts, dtype=torch.int64)
    tail_logical_base = torch.tensor(
        [(tail_size - 2 + row * 3) % tail_size for row in range(n_pools)],
        dtype=torch.int32,
    )
    loc = torch.tensor(
        [(row * 7 + 3) % cache_rows for row in range(n_pools)],
        dtype=torch.int64,
    )
    write_mask = None
    if use_write_mask:
        write_mask = torch.tensor(
            [row % 3 != 1 for row in range(n_pools)], dtype=torch.bool
        )

    inputs = {
        "chunk_k": chunk_k.contiguous(),
        "chunk_score": chunk_score.contiguous(),
        "tail_k": tail_k.contiguous(),
        "tail_score": tail_score.contiguous(),
        "req_pool_idx": req_pool_idx.contiguous(),
        "n_from_tail": n_from_tail.contiguous(),
        "chunk_src_start": chunk_src_start.contiguous(),
        "tail_logical_base": tail_logical_base.contiguous(),
        "ape": ape.contiguous(),
        "loc": loc.contiguous(),
    }
    if write_mask is not None:
        inputs["write_mask"] = write_mask.contiguous()
    return initial_cache.contiguous(), inputs


def _run_accuracy_case(
    operator,
    *,
    seed,
    n_pools,
    pool_size,
    tail_size,
    n_from_tail_values,
    score_dtype,
    use_write_mask=False,
    use_noncontiguous_dynamic_inputs=False,
    num_programs=None,
    total_tokens=None,
):
    initial_cache, inputs = _make_inputs(
        seed=seed,
        n_pools=n_pools,
        pool_size=pool_size,
        tail_size=tail_size,
        n_from_tail_values=n_from_tail_values,
        score_dtype=score_dtype,
        use_write_mask=use_write_mask,
        total_tokens=total_tokens,
    )
    if use_noncontiguous_dynamic_inputs:
        dynamic_names = ["chunk_k", "chunk_score", "ape", "loc"]
        if "write_mask" in inputs:
            dynamic_names.append("write_mask")
        for name in dynamic_names:
            inputs[name] = _make_noncontiguous(inputs[name])

    impl, _ = get_test_impls()
    require_test_impl(operator, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        expected_cache = initial_cache.clone()
        actual_cache = initial_cache.clone()
        kpool_assemble_softmax_write_cache_torch(index_k_cache=expected_cache, **inputs)
        operator(
            index_k_cache=actual_cache,
            num_programs=num_programs,
            impl=impl,
            **inputs,
        )

        assert_accuracy(actual_cache, expected_cache, CACHE_STANDARD, path="cache")
        write_mask = inputs.get("write_mask")
        if write_mask is not None:
            inactive_locs = inputs["loc"][~write_mask]
            initial_2d = initial_cache.view(-1, HEAD_DIM)
            actual_2d = actual_cache.view(-1, HEAD_DIM)
            assert_accuracy(
                actual_2d.index_select(0, inactive_locs),
                initial_2d.index_select(0, inactive_locs),
                EXACT_STANDARD,
                path="inactive_cache",
            )

    snapshots = None
    call_index = 0

    def executor():
        nonlocal snapshots, call_index
        if snapshots is None:
            snapshots = [initial_cache.clone() for _ in range(11)]
        operator(
            index_k_cache=snapshots[call_index],
            num_programs=num_programs,
            impl=impl,
            **inputs,
        )
        call_index += 1

    executor.func = operator
    executor.impl = impl
    executor.impl_ref = "torch_reference"
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=(total_tokens, n_pools) in perf_shape)


@pytest.mark.parametrize(
    "score_dtype,seed",
    [
        pytest.param(torch.bfloat16, 100, id="bf16"),
        pytest.param(torch.float16, 101, id="fp16"),
        pytest.param(torch.float32, 102, id="fp32"),
    ],
)
def test_all_chunk_score_dtypes(score_dtype, seed):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_write_cache_npu,
    )

    _run_accuracy_case(
        kpool_assemble_softmax_write_cache_npu,
        seed=seed,
        n_pools=5,
        pool_size=8,
        tail_size=13,
        n_from_tail_values=[0] * 5,
        score_dtype=score_dtype,
    )


def test_mixed_tail_chunk_wraparound_mask_and_noncontiguous_inputs():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_write_cache_npu,
    )

    _run_accuracy_case(
        kpool_assemble_softmax_write_cache_npu,
        seed=200,
        n_pools=9,
        pool_size=8,
        tail_size=11,
        n_from_tail_values=[0, 1, 3, 7, 8, 2, 5, 8, 4],
        score_dtype=torch.float32,
        use_write_mask=True,
        use_noncontiguous_dynamic_inputs=True,
        num_programs=4,
    )


def test_grid_stride_when_pools_exceed_programs():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_write_cache_npu,
    )

    n_pools = 37
    _run_accuracy_case(
        kpool_assemble_softmax_write_cache_npu,
        seed=300,
        n_pools=n_pools,
        pool_size=16,
        tail_size=19,
        n_from_tail_values=[(row * 5) % 17 for row in range(n_pools)],
        score_dtype=torch.bfloat16,
        num_programs=3,
    )


@pytest.mark.parametrize(
    "total_tokens,n_pools",
    [
        pytest.param(4096, 1024, id="W-extend-q4096-np1024"),
        pytest.param(8192, 2048, id="W-extend-q8192-np2048"),
    ],
)
def test_extend_workload_accuracy_and_performance(total_tokens, n_pools):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_write_cache_npu,
    )

    _run_accuracy_case(
        kpool_assemble_softmax_write_cache_npu,
        seed=400,
        n_pools=n_pools,
        pool_size=4,
        tail_size=5,
        n_from_tail_values=[2] * n_pools,
        score_dtype=torch.bfloat16,
        total_tokens=total_tokens,
    )
