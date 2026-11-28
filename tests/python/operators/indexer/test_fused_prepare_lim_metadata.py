import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def generate_data(batch, cols, q):
    slots = torch.arange(batch, dtype=torch.int32)
    mapping = torch.arange(batch + 1, dtype=torch.int32).flip(0)
    capacity = 4 if q == 4 else 8192
    data = dict(
        seq_lens=torch.arange(batch, dtype=torch.int32) + (1 if q == 4 else 8191),
        request_slots=slots,
        last_prefix=torch.full((batch + 1,), 2, dtype=torch.int32),
        block_table=torch.arange(batch * cols, dtype=torch.int32).reshape(batch, cols),
        out_block_table=torch.zeros((batch, cols + 3), dtype=torch.int32),
        safe_slots=torch.zeros(batch, dtype=torch.int32),
        resident_safe_slots=torch.zeros(batch, dtype=torch.int32),
        lengths=torch.zeros(batch, dtype=torch.int32),
        sparse_rows=torch.zeros(batch, dtype=torch.bool),
        valid_rows=torch.zeros(batch, dtype=torch.bool),
        request_states=torch.zeros(batch, dtype=torch.int32),
        next_prefixes=torch.zeros(batch, dtype=torch.int32),
        cache_sizes=torch.zeros(batch, dtype=torch.int32),
        capacity=capacity,
        request_rows=batch,
        resident_rows=batch,
        request_to_resident=mapping,
    )
    if q == 4:
        data["query_width"] = q
        data["positions"] = torch.tensor(
            [[-3, -2, -1, 0], [0, 1, 2, 3], [1, 2, 3, 4], [-4, -3, -2, -1]],
            dtype=torch.int64,
        )
        data["last_prefix"][2] = 6  # Visible length 5 has rolled back.
    return data


@pytest.mark.parametrize(
    "batch,cols,q",
    [
        # DSA decode with a 128K context, page size 128 and C=8192.
        pytest.param(16, 1024, 1, id="dsa-decode-b16-context128k"),
        pytest.param(4, 7, 4, id="mtp-q4-invalid-mapping-and-tail"),
    ],
)
def test_fused_prepare_lim_metadata(batch, cols, q):
    from sgl_kernel_npu.operators.indexer import fused_prepare_lim_metadata

    actual = generate_data(batch, cols, q)
    executor = make_executor(fused_prepare_lim_metadata, kwargs=actual)
    if get_skip_acc_check() is True:
        require_test_impl(fused_prepare_lim_metadata, executor.impl)
    else:
        require_distinct_impls(
            fused_prepare_lim_metadata, executor.impl, executor.impl_ref
        )
        reference = generate_data(batch, cols, q)
        executor()
        fused_prepare_lim_metadata(**reference, impl=executor.impl_ref)
        for key in (
            "out_block_table",
            "safe_slots",
            "resident_safe_slots",
            "lengths",
            "sparse_rows",
            "valid_rows",
            "request_states",
            "next_prefixes",
            "cache_sizes",
        ):
            assert_accuracy(actual[key], reference[key])
        if q == 4:
            assert_accuracy(
                actual["lengths"], torch.tensor([0, 4, 5, 0], dtype=torch.int32)
            )
            assert_accuracy(
                actual["request_states"],
                torch.tensor([-3, -1, -2, -3], dtype=torch.int32),
            )
            assert_accuracy(
                actual["resident_safe_slots"],
                torch.tensor([4, 3, 2, 4], dtype=torch.int32),
            )
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=batch == 16)
