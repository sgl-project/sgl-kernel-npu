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

# W: target-verify batch/draft candidates; compression remains enabled.
perf_shape = [(64, 4)]

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
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(operator, impl, impl_ref)
    else:
        require_test_impl(operator, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        expected_cache = initial_cache.clone()
        expected_tail_k = initial_tail_k.clone()
        expected_tail_score = initial_tail_score.clone()
        actual_cache = initial_cache.clone()
        actual_tail_k = initial_tail_k.clone()
        actual_tail_score = initial_tail_score.clone()

        operator(
            index_k_cache=expected_cache,
            tail_k=expected_tail_k,
            tail_score=expected_tail_score,
            **inputs,
            impl=impl_ref,
        )
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
            inactive_locs.extend(
                write_loc[batch, first_inactive:max_closed_pools].tolist()
            )

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
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor, enable_perf=(batch_size, num_draft_tokens) in perf_shape
    )


# Each row covers a distinct dtype, state transition, boundary, or workload.
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(
                seed=100,
                batch_size=4,
                num_draft_tokens=2,
                pool_size=4,
                write_start_values=[2, 3, 6, 7],
                score_dtype=torch.bfloat16,
            ),
            id="bf16",
        ),
        pytest.param(
            dict(
                seed=101,
                batch_size=4,
                num_draft_tokens=2,
                pool_size=4,
                write_start_values=[2, 3, 6, 7],
                score_dtype=torch.float16,
            ),
            id="fp16",
        ),
        pytest.param(
            dict(
                seed=102,
                batch_size=4,
                num_draft_tokens=2,
                pool_size=4,
                write_start_values=[2, 3, 6, 7],
                score_dtype=torch.float32,
            ),
            id="fp32",
        ),
        pytest.param(
            dict(
                seed=200,
                batch_size=5,
                num_draft_tokens=2,
                pool_size=8,
                write_start_values=[0, 5, 6, 7, 6],
                score_dtype=torch.float32,
                invalid_batches=(4,),
                num_programs=2,
            ),
            id="no_compression_and_invalid_batch",
        ),
        pytest.param(
            dict(
                seed=300,
                batch_size=17,
                num_draft_tokens=4,
                pool_size=8,
                write_start_values=[batch * 3 % 24 for batch in range(17)],
                score_dtype=torch.bfloat16,
                effective_n_values=[batch % 5 for batch in range(17)],
                use_noncontiguous_inputs=True,
                num_programs=3,
            ),
            id="effective_n_noncontiguous_and_grid_stride",
        ),
        pytest.param(
            dict(
                seed=400,
                batch_size=4,
                num_draft_tokens=10,
                pool_size=4,
                write_start_values=[0, 3, 6, 7],
                score_dtype=torch.float32,
                effective_n_values=[10, 10, 9, 0],
                num_programs=2,
            ),
            id="multiple_completed_pools",
        ),
        pytest.param(
            dict(
                seed=500,
                batch_size=64,
                num_draft_tokens=4,
                pool_size=4,
                write_start_values=[row % 4 for row in range(64)],
                score_dtype=torch.bfloat16,
            ),
            id="W-target-verify-b64-n4",
        ),
    ],
)
def test_kpool_write_tail_and_maybe_compress_npu(case):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_write_tail_and_maybe_compress_npu,
    )

    _run_accuracy_case(kpool_write_tail_and_maybe_compress_npu, **case)
