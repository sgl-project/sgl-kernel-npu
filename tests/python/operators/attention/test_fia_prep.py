"""Verify reordered pages, lengths, padding and caller-owned outputs."""

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


def test_fia_prep():
    from sgl_kernel_npu.operators.attention import fia_prep

    ti = torch.tensor(
        [[0, 2, 1, -1], [1, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32
    )
    lengths = torch.tensor([35, 18, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1, 0], dtype=torch.int32)
    pages = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    args = (ti, lengths, requests, pages, 16)

    def isolated_prep(*, impl="auto"):
        table = torch.empty((3, 4), dtype=torch.int32)
        out_lengths = torch.empty((3,), dtype=torch.int32)
        actual = fia_prep(*args, table, out_lengths, impl=impl)
        assert actual[0] is table and actual[1] is out_lengths
        return actual

    impl, impl_ref = get_test_impls()
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        require_distinct_impls(fia_prep, impl, impl_ref)
        assert_accuracy(isolated_prep(impl=impl), isolated_prep(impl=impl_ref))
    else:
        require_test_impl(fia_prep, impl)
    table = torch.empty((3, 4), dtype=torch.int32)
    out_lengths = torch.empty((3,), dtype=torch.int32)

    def executor():
        return fia_prep(*args, table, out_lengths, impl=impl)

    executor.func = fia_prep
    executor.impl = impl
    executor.impl_ref = impl_ref
    executor.accuracy_checked = accuracy_checked
    check_performance(executor, enable_perf=True)
