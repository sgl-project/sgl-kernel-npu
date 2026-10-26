from typing import Optional

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import get_test_impls
from tests.python.utils.performance import check_performance
from tests.python.utils.standards import AccuracyStandard

# W: target-verify batch/draft candidates; compression remains enabled.
perf_shape = [(64, 4), (64, 5)]

HEAD_DIM = 128
PAGE_SIZE = 128
CACHE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-2, rtol=1e-2),
}
EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float16": AccuracyStandard(atol=0, rtol=0),
    "torch.float32": AccuracyStandard(atol=0, rtol=0),
}


@torch.no_grad()
def kpool_write_tail_and_maybe_compress_torch(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
) -> None:
    """Torch reference matching the original target-verify semantics."""
    total_rows = key.shape[0]
    if total_rows == 0:
        return

    batch_size = total_rows // num_draft_tokens
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    cache_2d = index_k_cache.view(-1, HEAD_DIM)

    for batch in range(batch_size):
        if out_cache_loc[batch * num_draft_tokens].item() == 0:
            continue

        req = int(req_pool_indices[batch].item())
        start = int(write_start[batch].item())
        for draft_offset in range(num_draft_tokens):
            row = batch * num_draft_tokens + draft_offset
            tail_slot = (start + draft_offset) % tail_size
            tail_k[req, tail_slot] = key[row]
            tail_score[req, tail_slot] = score[row]

        effective_n = (
            num_draft_tokens
            if effective_n_per_batch is None
            else int(effective_n_per_batch[batch].item())
        )
        effective_n = max(0, min(effective_n, num_draft_tokens))
        base_pool = start // pool_size
        completed_pools = (start + effective_n) // pool_size - base_pool
        if completed_pools == 0:
            continue

        first_logical_start = int(tail_logical_start[batch].item())
        slots = torch.arange(pool_size, dtype=torch.int64)
        for pool_offset in range(completed_pools):
            logical_start = first_logical_start + pool_offset * pool_size
            tail_slots = torch.remainder(logical_start + slots, tail_size)
            pool_keys = tail_k[req].index_select(0, tail_slots).float()
            pool_scores = tail_score[req].index_select(0, tail_slots).float()
            probs = torch.softmax(pool_scores + ape.float(), dim=0)
            pooled = torch.sum(pool_keys * probs, dim=0).to(torch.bfloat16)
            cache_2d[int(write_loc[batch, pool_offset].item())] = pooled


def _make_noncontiguous(tensor):
    result = torch.stack((tensor, tensor), dim=-1)[..., 0]
    assert result.shape == tensor.shape and not result.is_contiguous()
    return result


def _make_inputs(
    *,
    seed,
    batch_size,
    num_draft_tokens,
    pool_size,
    write_start_values,
    score_dtype,
    effective_n_values=None,
    invalid_batches=(),
    use_noncontiguous_inputs=False,
):
    assert len(write_start_values) == batch_size
    if effective_n_values is not None:
        assert len(effective_n_values) == batch_size

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    total_rows = batch_size * num_draft_tokens
    tail_size = pool_size + num_draft_tokens
    max_closed_pools = (num_draft_tokens + pool_size - 1) // pool_size
    required_cache_rows = batch_size * max_closed_pools + 3
    cache_pages = max(2, (required_cache_rows + PAGE_SIZE - 1) // PAGE_SIZE)
    cache_rows = cache_pages * PAGE_SIZE

    def randn(shape, dtype):
        return torch.randn(shape, dtype=torch.float32).to(dtype)

    initial_cache = randn(
        (cache_pages, PAGE_SIZE, 1, HEAD_DIM), torch.bfloat16
    ).contiguous()
    initial_tail_k = randn(
        (batch_size, tail_size, HEAD_DIM), torch.bfloat16
    ).contiguous()
    initial_tail_score = randn(
        (batch_size, tail_size, HEAD_DIM), score_dtype
    ).contiguous()
    key = randn((total_rows, HEAD_DIM), torch.bfloat16)
    score = randn((total_rows, HEAD_DIM), score_dtype) * 4
    ape = randn((pool_size, HEAD_DIM), torch.float32) * 2
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64)
    write_start = torch.tensor(write_start_values, dtype=torch.int32)
    tail_logical_start = torch.tensor(
        [(value // pool_size) * pool_size for value in write_start_values],
        dtype=torch.int32,
    )
    write_loc = (
        torch.arange(batch_size * max_closed_pools, dtype=torch.int64).view(
            batch_size, max_closed_pools
        )
        + 3
    )
    assert int(write_loc.max().item()) < cache_rows
    out_cache_loc = torch.arange(1, total_rows + 1, dtype=torch.int64)
    for batch in invalid_batches:
        start = batch * num_draft_tokens
        out_cache_loc[start : start + num_draft_tokens] = 0

    inputs = {
        "key": key.contiguous(),
        "score": score.contiguous(),
        "ape": ape.contiguous(),
        "req_pool_indices": req_pool_indices.contiguous(),
        "write_start": write_start.contiguous(),
        "tail_logical_start": tail_logical_start.contiguous(),
        "write_loc": write_loc.contiguous(),
        "out_cache_loc": out_cache_loc.contiguous(),
        "num_draft_tokens": num_draft_tokens,
    }
    if effective_n_values is not None:
        inputs["effective_n_per_batch"] = torch.tensor(
            effective_n_values, dtype=torch.int32
        ).contiguous()

    if use_noncontiguous_inputs:
        dynamic_names = [
            "key",
            "score",
            "ape",
            "req_pool_indices",
            "write_start",
            "tail_logical_start",
            "out_cache_loc",
        ]
        if "effective_n_per_batch" in inputs:
            dynamic_names.append("effective_n_per_batch")
        for name in dynamic_names:
            inputs[name] = _make_noncontiguous(inputs[name])

    return initial_cache, initial_tail_k, initial_tail_score, inputs


def _run_accuracy_case(
    operator,
    *,
    seed,
    batch_size,
    num_draft_tokens,
    pool_size,
    write_start_values,
    score_dtype,
    effective_n_values=None,
    invalid_batches=(),
    use_noncontiguous_inputs=False,
    num_programs=None,
):
    initial_cache, initial_tail_k, initial_tail_score, inputs = _make_inputs(
        seed=seed,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        pool_size=pool_size,
        write_start_values=write_start_values,
        score_dtype=score_dtype,
        effective_n_values=effective_n_values,
        invalid_batches=invalid_batches,
        use_noncontiguous_inputs=use_noncontiguous_inputs,
    )
    expected_cache = initial_cache.clone()
    expected_tail_k = initial_tail_k.clone()
    expected_tail_score = initial_tail_score.clone()
    actual_cache = initial_cache.clone()
    actual_tail_k = initial_tail_k.clone()
    actual_tail_score = initial_tail_score.clone()

    kpool_write_tail_and_maybe_compress_torch(
        index_k_cache=expected_cache,
        tail_k=expected_tail_k,
        tail_score=expected_tail_score,
        **inputs,
    )
    impl, _ = get_test_impls()
    operator(
        index_k_cache=actual_cache,
        tail_k=actual_tail_k,
        tail_score=actual_tail_score,
        num_programs=num_programs,
        impl=impl,
        **inputs,
    )

    assert_accuracy(actual_tail_k, expected_tail_k, EXACT_STANDARD, path="tail_k")
    assert_accuracy(
        actual_tail_score,
        expected_tail_score,
        EXACT_STANDARD,
        path="tail_score",
    )
    assert_accuracy(actual_cache, expected_cache, CACHE_STANDARD, path="cache")

    inactive_locs = []
    write_loc = inputs["write_loc"]
    max_closed_pools = write_loc.shape[1]
    for batch, start in enumerate(write_start_values):
        valid = batch not in invalid_batches
        gate_n = (
            num_draft_tokens
            if effective_n_values is None
            else effective_n_values[batch]
        )
        gate_n = max(0, min(gate_n, num_draft_tokens))
        completed = (start + gate_n) // pool_size - start // pool_size
        first_inactive = completed if valid else 0
        inactive_locs.extend(write_loc[batch, first_inactive:max_closed_pools].tolist())

    if inactive_locs:
        inactive_index = torch.tensor(inactive_locs, dtype=torch.int64)
        assert_accuracy(
            actual_cache.view(-1, HEAD_DIM).index_select(0, inactive_index),
            initial_cache.view(-1, HEAD_DIM).index_select(0, inactive_index),
            EXACT_STANDARD,
            path="inactive_cache",
        )

    for batch in invalid_batches:
        assert_accuracy(
            actual_tail_k[batch],
            initial_tail_k[batch],
            EXACT_STANDARD,
            path=f"invalid_tail_k[{batch}]",
        )
        assert_accuracy(
            actual_tail_score[batch],
            initial_tail_score[batch],
            EXACT_STANDARD,
            path=f"invalid_tail_score[{batch}]",
        )

    assert inputs["write_loc"].is_contiguous()
    snapshots = None
    call_index = 0

    def executor():
        nonlocal snapshots, call_index
        if snapshots is None:
            snapshots = [
                (
                    initial_cache.clone(),
                    initial_tail_k.clone(),
                    initial_tail_score.clone(),
                )
                for _ in range(11)
            ]
        perf_cache, perf_tail_k, perf_tail_score = snapshots[call_index]
        call_index += 1
        operator(
            index_k_cache=perf_cache,
            tail_k=perf_tail_k,
            tail_score=perf_tail_score,
            num_programs=num_programs,
            impl=impl,
            **inputs,
        )

    executor.func = operator
    executor.impl = impl
    executor.impl_ref = "torch_reference"
    executor.accuracy_checked = True
    check_performance(
        executor, enable_perf=(batch_size, num_draft_tokens) in perf_shape
    )


@pytest.mark.parametrize(
    "score_dtype,seed",
    [
        pytest.param(torch.bfloat16, 100, id="bf16"),
        pytest.param(torch.float16, 101, id="fp16"),
        pytest.param(torch.float32, 102, id="fp32"),
    ],
)
def test_score_dtypes_and_ring_wrap_compression(score_dtype, seed):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    _run_accuracy_case(
        kpool_write_tail_and_maybe_compress_npu,
        seed=seed,
        batch_size=4,
        num_draft_tokens=2,
        pool_size=4,
        write_start_values=[2, 3, 6, 7],
        score_dtype=score_dtype,
    )


def test_no_compression_and_invalid_batch():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    _run_accuracy_case(
        kpool_write_tail_and_maybe_compress_npu,
        seed=200,
        batch_size=5,
        num_draft_tokens=2,
        pool_size=8,
        write_start_values=[0, 5, 6, 7, 6],
        score_dtype=torch.float32,
        invalid_batches=(4,),
        num_programs=2,
    )


def test_effective_n_noncontiguous_and_grid_stride():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    batch_size = 17
    _run_accuracy_case(
        kpool_write_tail_and_maybe_compress_npu,
        seed=300,
        batch_size=batch_size,
        num_draft_tokens=4,
        pool_size=8,
        write_start_values=[(batch * 3) % 24 for batch in range(batch_size)],
        score_dtype=torch.bfloat16,
        effective_n_values=[batch % 5 for batch in range(batch_size)],
        use_noncontiguous_inputs=True,
        num_programs=3,
    )


def test_multiple_completed_pools():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    _run_accuracy_case(
        kpool_write_tail_and_maybe_compress_npu,
        seed=400,
        batch_size=4,
        num_draft_tokens=10,
        pool_size=4,
        write_start_values=[0, 3, 6, 7],
        score_dtype=torch.float32,
        effective_n_values=[10, 10, 9, 0],
        num_programs=2,
    )


@pytest.mark.parametrize(
    "batch_size,num_draft_tokens",
    [
        pytest.param(64, 4, id="W-target-verify-b64-n4"),
        pytest.param(64, 5, id="W-target-verify-b64-n5"),
    ],
)
def test_target_verify_workload_accuracy_and_performance(batch_size, num_draft_tokens):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    _run_accuracy_case(
        kpool_write_tail_and_maybe_compress_npu,
        seed=500,
        batch_size=batch_size,
        num_draft_tokens=num_draft_tokens,
        pool_size=4,
        write_start_values=[row % 4 for row in range(batch_size)],
        score_dtype=torch.bfloat16,
    )
