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


def _assert_decode_outputs(actual, expected):
    actual_buf, actual_tail_k, actual_tail_score = actual
    expected_buf, expected_tail_k, expected_tail_score = expected
    assert_accuracy(actual_tail_k, expected_tail_k, TAIL_K_STANDARD, path="tail_k")
    assert_accuracy(
        actual_tail_score,
        expected_tail_score,
        TAIL_SCORE_STANDARD,
        path="tail_score",
    )
    assert_accuracy(actual_buf, expected_buf, BUF_STANDARD, path="buf")


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
    inputs = (
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
    )
    kwargs = {
        "pool_size": pool_size,
        "slots_per_page": slots_per_page,
        "head_dim": head_dim,
    }
    return initial_buf, initial_tail_k, initial_tail_score, inputs, kwargs


def _run_production_layout_case(
    batch,
    block_table_cols,
    pool_size=4,
    slots_per_page=64,
    distinct_page_columns=False,
    cache_rank=4,
    score_dtype=torch.bfloat16,
    invalid_row=False,
):
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache_bf16,
    )

    initial_buf, initial_tail_k, initial_tail_score, inputs, kwargs = (
        generate_test_data(
            batch,
            block_table_cols,
            pool_size,
            slots_per_page,
            distinct_page_columns,
            cache_rank,
            score_dtype,
            invalid_row,
        )
    )
    impl, impl_ref = get_test_impls()
    if get_skip_acc_check() is not True:
        require_distinct_impls(
            kpool_decode_update_and_maybe_write_cache_bf16, impl, impl_ref
        )
    else:
        require_test_impl(kpool_decode_update_and_maybe_write_cache_bf16, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        buf = initial_buf.clone()
        tail_k = initial_tail_k.clone()
        tail_score = initial_tail_score.clone()
        buf_ref = initial_buf.clone()
        tail_k_ref = initial_tail_k.clone()
        tail_score_ref = initial_tail_score.clone()
        kpool_decode_update_and_maybe_write_cache_bf16(
            buf, tail_k, tail_score, *inputs, **kwargs, impl=impl
        )
        kpool_decode_update_and_maybe_write_cache_bf16(
            buf_ref, tail_k_ref, tail_score_ref, *inputs, **kwargs, impl=impl_ref
        )
        if invalid_row:
            assert_accuracy(
                tail_k[1], initial_tail_k[1], EXACT_STANDARD, path="invalid_tail_k"
            )
            assert_accuracy(
                tail_score[1],
                initial_tail_score[1],
                EXACT_STANDARD,
                path="invalid_tail_score",
            )
        _assert_decode_outputs(
            (buf, tail_k, tail_score), (buf_ref, tail_k_ref, tail_score_ref)
        )

    snapshots = None
    call_index = 0

    def executor():
        nonlocal snapshots, call_index
        if snapshots is None:
            # Build all 11 states before benchmark_npu starts profiling.
            snapshots = [
                (
                    initial_buf.clone(),
                    initial_tail_k.clone(),
                    initial_tail_score.clone(),
                )
                for _ in range(11)
            ]
        perf_buf, perf_tail_k, perf_tail_score = snapshots[call_index]
        call_index += 1
        kpool_decode_update_and_maybe_write_cache_bf16(
            perf_buf,
            perf_tail_k,
            perf_tail_score,
            *inputs,
            **kwargs,
            impl=impl,
        )

    executor.func = kpool_decode_update_and_maybe_write_cache_bf16
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=(batch, block_table_cols) in perf_shape
        and pool_size == 4
        and slots_per_page == 64,
    )


def test_decode_rejects_noncontiguous_inplace_cache():
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_decode_update_and_maybe_write_cache_bf16,
    )

    base = torch.zeros(2, 64, 1, 128, dtype=torch.bfloat16)
    buf = torch.stack((base, base), dim=-1)[..., 0]
    assert not buf.is_contiguous()
    with pytest.raises(AssertionError, match="buf must be contiguous"):
        kpool_decode_update_and_maybe_write_cache_bf16(
            buf,
            torch.zeros(1, 5, 128, dtype=torch.bfloat16),
            torch.zeros(1, 5, 128, dtype=torch.bfloat16),
            torch.zeros(1, 128, dtype=torch.bfloat16),
            torch.zeros(1, 128, dtype=torch.bfloat16),
            torch.zeros(4, 128, dtype=torch.float32),
            torch.zeros(1, 64, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            pool_size=4,
            slots_per_page=64,
        )


# One typical 4D BF16 workload; small cases cover the other public contracts.
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(batch=16, block_table_cols=64), id="T-glm53-flash-decode-4d-bf16"
        ),
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
    _run_production_layout_case(**case)
