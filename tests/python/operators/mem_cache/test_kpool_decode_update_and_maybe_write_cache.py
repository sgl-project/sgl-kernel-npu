# In-place Operator
"""In-place decode cache/tail updates with isolated accuracy and performance state."""

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

# T: representative B=16 decode case; not a captured production tensor.
HEAD_DIM = 128
perf_shape = [(16, 64)]

BUF_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=2e-2, rtol=1e-5),
}
TAIL_K_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=1e-3, rtol=1e-5),
}
TAIL_SCORE_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float32": AccuracyStandard(atol=1e-6, rtol=1e-5),
}
EXACT_STANDARD = {
    "torch.bfloat16": AccuracyStandard(atol=0, rtol=0),
    "torch.float32": AccuracyStandard(atol=0, rtol=0),
}


def generate_test_data(
    batch,
    block_table_cols,
    pool_size=4,
    slots_per_page=64,
    distinct_page_columns=False,
    cache_rank=4,
    score_dtype=torch.bfloat16,
    invalid_row=False,
):
    """Return operator kwargs and protected row IDs in the [-1, 128] view."""
    torch.manual_seed(420 + batch)
    tail_size = pool_size + 1
    head_dim = 128
    num_pages = batch * block_table_cols + 1 if distinct_page_columns else batch + 2

    # P: GLM-5.3-Flash NPU decode uses [pages, slots, 1, head_dim] BF16 cache.
    initial_buf = torch.zeros(
        num_pages, slots_per_page, 1, head_dim, dtype=torch.bfloat16
    )
    initial_tail_k = torch.randn(batch, tail_size, head_dim, dtype=torch.bfloat16) * 0.1
    initial_tail_score = (
        torch.randn(batch, tail_size, head_dim, dtype=score_dtype) * 0.1
    )
    key = torch.randn(batch, head_dim, dtype=torch.bfloat16) * 0.1
    slot_score = torch.randn(batch, head_dim, dtype=score_dtype) * 0.1
    ape = torch.randn(pool_size, head_dim, dtype=torch.float32) * 0.01
    block_tables = torch.randint(
        1, num_pages, (batch, block_table_cols), dtype=torch.int32
    )
    # Every compressed row targets a distinct, nonzero physical page.
    if distinct_page_columns:
        block_tables[:] = torch.arange(
            1, batch * block_table_cols + 1, dtype=torch.int32
        ).view(batch, block_table_cols)
    else:
        block_tables[:] = torch.arange(1, batch + 1, dtype=torch.int32).unsqueeze(1)
    req_pool_indices = torch.arange(batch, dtype=torch.int32)
    positions = torch.arange(batch, dtype=torch.int32) * pool_size + pool_size - 1
    if batch >= 3:
        positions[:3] = torch.tensor(
            [
                slots_per_page * pool_size - 1,
                slots_per_page * pool_size,
                slots_per_page * pool_size + pool_size - 1,
            ],
            dtype=torch.int32,
        )
    seq_lens = positions + 10
    out_cache_loc = torch.ones(batch, dtype=torch.int32)
    if invalid_row:
        out_cache_loc[1] = 0
        # Complete a pool so this invalid row must suppress both cache and tail writes.
        positions[1] += pool_size - 1
        # Retain a valid, incomplete pool in the same 2D/FP32 edge case.
        positions[-1] -= 1
    if cache_rank == 2:
        initial_buf = initial_buf.view(num_pages, -1)
    else:
        assert cache_rank == 4
    inputs = dict(
        buf=initial_buf,
        tail_k=initial_tail_k,
        tail_score=initial_tail_score,
        key=key,
        slot_score=slot_score,
        ape=ape,
        block_tables=block_tables,
        req_pool_indices=req_pool_indices,
        positions=positions,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        pool_size=pool_size,
        slots_per_page=slots_per_page,
        head_dim=head_dim,
    )
    invalid_tail_rows = torch.arange(tail_size, 2 * tail_size, dtype=torch.int64)
    unchanged = (
        {name: invalid_tail_rows for name in ("tail_k", "tail_score")}
        if invalid_row
        else {}
    )
    return inputs, unchanged


# Case 01: Reject a noncontiguous in-place cache.
@pytest.mark.extra_checks
def test_decode_rejects_noncontiguous_inplace_cache():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache,
    )

    inputs, _ = generate_test_data(batch=1, block_table_cols=64)
    buf = inputs["buf"]
    inputs["buf"] = torch.stack((buf, buf), dim=-1)[..., 0]
    assert not inputs["buf"].is_contiguous()
    with pytest.raises(AssertionError, match="buf must be contiguous"):
        kpool_decode_update_and_maybe_write_cache(**inputs)


@pytest.mark.parametrize(
    "case",
    [
        # Case 02: Typical decode: B16, 4D BF16 cache and 64 page-table columns.
        pytest.param(
            dict(batch=16, block_table_cols=64), id="T-glm53-flash-decode-4d-bf16"
        ),
        # Case 03: 2D cache, FP32 scores, invalid row and an incomplete valid pool.
        pytest.param(
            dict(
                batch=5,
                block_table_cols=9,
                cache_rank=2,
                score_dtype=torch.float32,
                invalid_row=True,
            ),
            id="edge-2d-fp32-invalid-row",
        ),
        # Case 04: Pool2/page16: distinct physical columns at page boundaries.
        pytest.param(
            dict(
                batch=5,
                block_table_cols=9,
                pool_size=2,
                slots_per_page=16,
                distinct_page_columns=True,
            ),
            id="edge-pool2-page16",
        ),
        # Case 05: Pool4/page16: distinct physical columns at page boundaries.
        pytest.param(
            dict(
                batch=5,
                block_table_cols=9,
                pool_size=4,
                slots_per_page=16,
                distinct_page_columns=True,
            ),
            id="edge-pool4-page16",
        ),
        # Case 06: Pool8/page64: distinct physical columns at page boundaries.
        pytest.param(
            dict(
                batch=5,
                block_table_cols=9,
                pool_size=8,
                slots_per_page=64,
                distinct_page_columns=True,
            ),
            id="edge-pool8-page64",
        ),
    ],
)
def test_decode_cache_write(case):
    """In-place: update cache/tails, preserving invalid-request tail rows."""
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache,
    )

    # Inputs: named operator arguments and independent unchanged-row checks.
    inputs, unchanged = generate_test_data(**case)
    standards = {
        "buf": BUF_STANDARD,
        "tail_k": TAIL_K_STANDARD,
        "tail_score": TAIL_SCORE_STANDARD,
    }
    state_names = tuple(standards)
    # Accuracy: isolate DUT/reference state for this in-place operator.
    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(
            kpool_decode_update_and_maybe_write_cache, impl, impl_ref
        )
    else:
        require_test_impl(kpool_decode_update_and_maybe_write_cache, impl)

    if accuracy_checked:
        actual = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        reference = dict(inputs, **{name: inputs[name].clone() for name in state_names})
        kpool_decode_update_and_maybe_write_cache(**reference, impl=impl_ref)
        kpool_decode_update_and_maybe_write_cache(**actual, impl=impl)
        for name, standard in standards.items():
            assert_accuracy(actual[name], reference[name], standard, path=name)
        # Independent invariant: untouched rows must match the original exactly.
        for name, rows in unchanged.items():
            assert_accuracy(
                actual[name].view(-1, HEAD_DIM).index_select(0, rows),
                inputs[name].view(-1, HEAD_DIM).index_select(0, rows),
                EXACT_STANDARD,
                path=f"{name}.unchanged",
            )

    shape = (inputs["key"].shape[0], inputs["block_tables"].shape[1])
    profile_case = (
        shape in perf_shape
        and inputs["pool_size"] == 4
        and inputs["slots_per_page"] == 64
    )
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
        return kpool_decode_update_and_maybe_write_cache(**next(states), impl=impl)

    executor.func = kpool_decode_update_and_maybe_write_cache
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=profile_case,
        warmup=warmup,
        active=active,
    )
