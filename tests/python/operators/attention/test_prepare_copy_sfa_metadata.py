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


def generate_data(batch, q, topk, pages, page_size):
    valid = torch.ones(batch, dtype=torch.bool)
    valid[-1] = False
    return dict(
        source_ids=torch.arange(batch * q * topk, dtype=torch.int32).reshape(
            batch * q, topk
        ),
        valid_rows=valid,
        sparse_rows=torch.arange(batch) % 2 == 0,
        request_slots=torch.arange(batch, dtype=torch.int32),
        seq_lens=torch.arange(batch, dtype=torch.int32) + 1,
        actual_seq_lengths_kv=torch.zeros(batch, dtype=torch.int32),
        hbm_block_table=torch.zeros((batch, pages), dtype=torch.int32),
        capacity=pages * page_size,
        query_width=q,
    )


@pytest.mark.parametrize(
    "shape",
    [
        # DSA verify with Q=4, K=2048 and C=8192.
        pytest.param((16, 4, 2048, 64, 128), id="dsa-verify-b16-q4"),
        pytest.param((3, 1, 16, 8, 1), id="invalid-and-short"),
    ],
)
def test_prepare_copy_sfa_metadata(shape):
    from sgl_kernel_npu.operators.attention import prepare_copy_sfa_metadata

    actual = generate_data(*shape)
    executor = make_executor(prepare_copy_sfa_metadata, kwargs=actual)
    if get_skip_acc_check() is True:
        require_test_impl(prepare_copy_sfa_metadata, executor.impl)
    else:
        require_distinct_impls(
            prepare_copy_sfa_metadata, executor.impl, executor.impl_ref
        )
        reference = generate_data(*shape)
        executor()
        prepare_copy_sfa_metadata(**reference, impl=executor.impl_ref)
        for key in ("source_ids", "actual_seq_lengths_kv", "hbm_block_table"):
            assert_accuracy(actual[key], reference[key])
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape[0] == 16)
