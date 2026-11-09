"""Check changing-length graph replay; paired compute tests check schedule semantics."""

import pytest
import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    make_executor,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

perf_shape = [1, 6]


def generate_data(rows):
    cu = torch.tensor([0, rows], dtype=torch.int32)
    lengths = torch.full((rows,), 4096, dtype=torch.int32)
    kwargs = dict(
        ori_topk_length=torch.full((rows, 1), 128, dtype=torch.int32),
        cmp_topk_length=lengths.reshape(rows, 1),
        cu_seqlens_q=cu,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        has_ori_kv=True,
        has_cmp_kv=True,
    )
    return kwargs, lengths


@pytest.mark.parametrize(
    "rows",
    [pytest.param(1, id="dsv41-decode"), pytest.param(6, id="dsv41-static-verify6")],
)
def test_mixed_quant_sparse_flash_mla_metadata(rows):
    from sgl_kernel_npu.operators.attention import mixed_quant_sparse_flash_mla_metadata

    kwargs, lengths = generate_data(rows)
    executor = make_executor(mixed_quant_sparse_flash_mla_metadata, kwargs=kwargs)
    require_test_impl(mixed_quant_sparse_flash_mla_metadata, executor.impl)
    if not get_skip_acc_check():
        initial = executor().clone()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = executor()
        pointer = captured.data_ptr()
        changed = False
        for value in (64, 4096, 192):
            lengths.fill_(value)
            expected = executor()
            graph.replay()
            torch.npu.synchronize()
            assert captured.dtype == torch.int32 and captured.data_ptr() == pointer
            assert_accuracy(captured, expected)
            changed |= not torch.equal(initial, expected)
        assert changed, "Lengths must cross a scheduling boundary"
        # This records replay parity. Independent semantic checks live in the compute test.
        executor.accuracy_checked = True
    lengths.fill_(4096)
    check_performance(executor, enable_perf=rows in perf_shape)
