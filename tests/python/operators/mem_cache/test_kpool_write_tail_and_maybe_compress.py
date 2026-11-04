# In-place Operator
"""In-place draft tail writes/compression with explicit unchanged-row checks."""

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


def generate_test_data(
    *,
    seed=100,
    batch_size=4,
    num_draft_tokens=2,
    pool_size=4,
    write_start_values=None,
    score_dtype=torch.bfloat16,
    effective_n_values=None,
    invalid_batches=(),
    use_noncontiguous_inputs=False,
    num_programs=None,
):
    """Return operator kwargs and protected row IDs in the [-1, 128] view."""
    if write_start_values is None:
        write_start_values = [2, 3, 6, 7]
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
        "index_k_cache": initial_cache,
        "tail_k": initial_tail_k,
        "tail_score": initial_tail_score,
        "num_programs": num_programs,
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
            tensor = inputs[name]
            inputs[name] = torch.stack((tensor, tensor), dim=-1)[..., 0]
            assert not inputs[name].is_contiguous()

    # effective_n gates completed pools only; all draft tokens still write tail.
    inactive_cache_rows = []
    for batch, start in enumerate(write_start_values):
        effective = (
            num_draft_tokens
            if effective_n_values is None
            else effective_n_values[batch]
        )
        completed = (
            start + max(0, min(effective, num_draft_tokens))
        ) // pool_size - start // pool_size
        first_inactive = 0 if batch in invalid_batches else completed
        inactive_cache_rows.extend(
            batch * max_closed_pools + slot + 3
            for slot in range(first_inactive, max_closed_pools)
        )
    invalid_tail_rows = torch.tensor(
        [
            batch * tail_size + slot
            for batch in invalid_batches
            for slot in range(tail_size)
        ],
        dtype=torch.int64,
    )
    unchanged = {
        "index_k_cache": torch.tensor(inactive_cache_rows, dtype=torch.int64),
        "tail_k": invalid_tail_rows,
        "tail_score": invalid_tail_rows,
    }
    assert inputs["write_loc"].is_contiguous()
    return inputs, unchanged


# Each row covers a distinct dtype, state transition, boundary, or workload.
@pytest.mark.parametrize(
    "case",
    [
        # Case 01: BF16 scores: tail writes and completed-pool compression.
        pytest.param(dict(score_dtype=torch.bfloat16), id="bf16"),
        # Case 02: FP16 scores: tail writes and completed-pool compression.
        pytest.param(dict(seed=101, score_dtype=torch.float16), id="fp16"),
        # Case 03: FP32 scores: tail writes and completed-pool compression.
        pytest.param(dict(seed=102, score_dtype=torch.float32), id="fp32"),
        # Case 04: Incomplete pools and an invalid batch must not write cache.
        pytest.param(
            dict(
                seed=200,
                batch_size=5,
                pool_size=8,
                write_start_values=[0, 5, 6, 7, 6],
                score_dtype=torch.float32,
                invalid_batches=(4,),
                num_programs=2,
            ),
            id="no_compression_and_invalid_batch",
        ),
        # Case 05: Effective draft counts 0..4, strided inputs and grid stride.
        pytest.param(
            dict(
                seed=300,
                batch_size=17,
                num_draft_tokens=4,
                pool_size=8,
                write_start_values=[batch * 3 % 24 for batch in range(17)],
                effective_n_values=[batch % 5 for batch in range(17)],
                use_noncontiguous_inputs=True,
                num_programs=3,
            ),
            id="effective_n_noncontiguous_and_grid_stride",
        ),
        # Case 06: Ten draft tokens can close multiple pools; zero effective count too.
        pytest.param(
            dict(
                seed=400,
                num_draft_tokens=10,
                write_start_values=[0, 3, 6, 7],
                score_dtype=torch.float32,
                effective_n_values=[10, 10, 9, 0],
                num_programs=2,
            ),
            id="multiple_completed_pools",
        ),
        # Case 07: Typical target verify: B64, four draft tokens; also profile.
        pytest.param(
            dict(
                seed=500,
                batch_size=64,
                num_draft_tokens=4,
                write_start_values=[row % 4 for row in range(64)],
            ),
            id="W-target-verify-b64-n4",
        ),
    ],
)
def test_kpool_write_tail_and_maybe_compress(case):
    """In-place: write cache/tails; preserve incomplete pools and invalid requests."""
    from sgl_kernel_npu.operators.mem_cache import kpool_write_tail_and_maybe_compress

    # Inputs: named operator arguments and independent unchanged-row checks.
    inputs, unchanged = generate_test_data(**case)
    state_names = ("index_k_cache", "tail_k", "tail_score")
    # Accuracy: isolate DUT/reference state for this in-place operator.
    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(kpool_write_tail_and_maybe_compress, impl, impl_ref)
    else:
        require_test_impl(kpool_write_tail_and_maybe_compress, impl)

    if accuracy_checked:
        actual = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        reference = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        kpool_write_tail_and_maybe_compress(**reference, impl=impl_ref)
        kpool_write_tail_and_maybe_compress(**actual, impl=impl)
        for name in state_names:
            standard = CACHE_STANDARD if name == "index_k_cache" else EXACT_STANDARD
            assert_accuracy(actual[name], reference[name], standard, path=name)
        # Independent invariant: untouched rows must match the original exactly.
        for name, rows in unchanged.items():
            assert_accuracy(
                actual[name].view(-1, HEAD_DIM).index_select(0, rows),
                inputs[name].view(-1, HEAD_DIM).index_select(0, rows),
                EXACT_STANDARD,
                path=f"{name}.unchanged",
            )

    shape = (inputs["tail_k"].shape[0], inputs["num_draft_tokens"])
    # Read/write tail state needs fresh inputs; allocate all copies before profiling.
    warmup, active = 5, 5
    states = None

    # Closure for in place operator. Refresh buffer to keep every iter equivalent
    def executor():
        nonlocal states
        if states is None:
            states = iter(
                [
                    dict(inputs, **{name: inputs[name].clone() for name in state_names})
                    for _ in range(1 + warmup + active)
                ]
            )
        return kpool_write_tail_and_maybe_compress(**next(states), impl=impl)

    executor.func = kpool_write_tail_and_maybe_compress
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=shape in perf_shape,
        warmup=warmup,
        active=active,
    )
