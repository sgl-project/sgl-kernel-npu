"""Verify reordered pages, lengths, padding and caller-owned outputs."""

from functools import wraps

import pytest

torch = pytest.importorskip("torch")


from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import make_executor
from tests.python.utils.performance import check_performance


def test_fia_prep():
    from sgl_kernel_npu.operators.attention import fia_prep

    ti = torch.tensor(
        [[0, 2, 1, -1], [1, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    lengths = torch.tensor([35, 18, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1, 0], dtype=torch.int32)
    pages = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    args = (ti, lengths, requests, pages, 16)

    @wraps(fia_prep)
    def isolated_prep(*, impl="auto"):
        table = torch.empty((3, 4), dtype=torch.int32)
        out_lengths = torch.empty((3,), dtype=torch.int32)
        actual = fia_prep(*args, table, out_lengths, impl=impl)
        assert actual[0] is table and actual[1] is out_lengths
        return actual

    executor = check_accuracy(isolated_prep)
    table = torch.empty((3, 4), dtype=torch.int32)
    out_lengths = torch.empty((3,), dtype=torch.int32)
    perf_executor = make_executor(
        fia_prep,
        args=(*args, table, out_lengths),
        impl=executor.impl,
        impl_ref=executor.impl_ref,
        accuracy_checked=executor.accuracy_checked,
    )
    check_performance(perf_executor, enable_perf=True)
