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
    lengths = torch.full((1,), 4096, dtype=torch.int32)
    kwargs = dict(
        cu_seqlens_q=cu,
        seqused_k=lengths,
        batch_size=1,
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        mask_mode=3,
        cmp_ratio=1,
        layout_q="TND",
        layout_k="PA_BBND",
        max_seqlen_q=rows,
        max_seqlen_k=4096,
    )
    return kwargs, lengths


@pytest.mark.parametrize(
    "rows",
    [pytest.param(1, id="dsv41-decode"), pytest.param(6, id="dsv41-static-verify6")],
)
def test_quant_lightning_indexer_metadata(rows):
    from sgl_kernel_npu.operators.indexer import quant_lightning_indexer_metadata

    kwargs, lengths = generate_data(rows)
    executor = make_executor(quant_lightning_indexer_metadata, kwargs=kwargs)
    require_test_impl(quant_lightning_indexer_metadata, executor.impl)
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
