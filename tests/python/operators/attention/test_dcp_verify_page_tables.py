"""Exact in-place page-table comparison and device timing through the public API."""

import pytest

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

torch = pytest.importorskip("torch")

# GLM-5.2 Target Verify dimensions with 128K active tokens and four query rows.
perf_shape = [(4, 128, 4, 1000, 1024, torch.int64, "verify")]


def generate_data(
    dcp_size, page_size, query_len, active_cols, cols, index_dtype, pattern
):
    generator = torch.Generator(device="cpu").manual_seed(123)
    slot_cols = cols * page_size
    if pattern == "shrink":
        # The final page has only its first token.
        slot_cols -= page_size - 1
    slots = torch.randint(
        0, 524_288, (5, slot_cols), generator=generator, dtype=torch.int32, device="cpu"
    ).npu()
    request_ids = [3] if pattern == "verify" else [4, -1, 3, 5, 4]
    requests = torch.tensor(request_ids, dtype=index_dtype, device="npu")
    ordinary = torch.full(
        (len(request_ids), cols), -99, dtype=torch.int32, device="npu"
    )
    local_cols = (cols + dcp_size - 1) // dcp_size + 1
    local = torch.full(
        (len(request_ids) * query_len, local_cols), -99, dtype=torch.int32, device="npu"
    )
    return slots, requests, ordinary, local, active_cols, query_len, page_size, dcp_size


@pytest.mark.parametrize(
    "dcp_size,page_size,query_len,active_cols,cols,index_dtype,pattern",
    [
        pytest.param(*perf_shape[0], id="glm52-verify-128k-dcp4"),
        pytest.param(
            2,
            64,
            1,
            129,
            129,
            torch.int32,
            "shrink",
            id="dcp2-partial-tile-invalid-requests-and-clear",
        ),
    ],
)
def test_dcp_verify_page_tables(
    dcp_size, page_size, query_len, active_cols, cols, index_dtype, pattern
):
    from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

    args = generate_data(
        dcp_size, page_size, query_len, active_cols, cols, index_dtype, pattern
    )
    executor = make_executor(dcp_verify_page_tables, args=args)
    if get_skip_acc_check():
        require_test_impl(dcp_verify_page_tables, executor.impl)
    else:
        require_distinct_impls(dcp_verify_page_tables, executor.impl, executor.impl_ref)
        slots, requests, ordinary, local, _, *params = args
        actual = ordinary.clone(), local.clone()
        expected = ordinary.clone(), local.clone()
        # Independent output buffers also verify that stale pages are cleared after reuse.
        active_sizes = (active_cols, 5, 0) if pattern == "shrink" else (active_cols,)
        for active in active_sizes:
            dcp_verify_page_tables(
                slots, requests, *actual, active, *params, impl=executor.impl
            )
            dcp_verify_page_tables(
                slots, requests, *expected, active, *params, impl=executor.impl_ref
            )
            assert_accuracy(actual, expected)
        executor.accuracy_checked = True
    check_performance(
        executor,
        enable_perf=(
            dcp_size,
            page_size,
            query_len,
            active_cols,
            cols,
            index_dtype,
            pattern,
        )
        in perf_shape,
    )


@pytest.mark.extra_checks
def test_dcp_verify_page_tables_rejects_active_cols_beyond_capacity():
    from sgl_kernel_npu.operators.attention import dcp_verify_page_tables

    require_test_impl(dcp_verify_page_tables, "ascendc")
    slots, requests, ordinary, local, _, *params = generate_data(*perf_shape[0])
    with pytest.raises(RuntimeError, match="Active pages exceed"):
        dcp_verify_page_tables(
            slots,
            requests,
            ordinary,
            local,
            ordinary.shape[1] + 1,
            *params,
            impl="ascendc",
        )
