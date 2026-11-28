"""Bitwise comparison and repeatable profiling for in-place KV moves."""

import torch

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


def check_move(
    operator, buffers, locations, options, expected, *, repeats=1, perf=False
):
    impl, impl_ref = get_test_impls()
    checked = not get_skip_acc_check()
    if checked:
        require_distinct_impls(operator, impl, impl_ref)
        ref = [cache.clone() for cache in buffers]
        for _ in range(repeats):
            operator(*ref, *locations, *options, impl=impl_ref)
            operator(*buffers, *locations, *options, impl=impl)
        for dut, native, oracle in zip(buffers, ref, expected, strict=True):
            assert_accuracy(dut.view(torch.int16).cpu(), oracle)
            assert_accuracy(native.view(torch.int16).cpu(), oracle)
    else:
        require_test_impl(operator, impl)
    executor = make_executor(
        operator,
        args=(*buffers, *locations, *options),
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=checked,
    )
    # Reuse buffers: each iteration moves the same slots and the same number of bytes.
    check_performance(executor, enable_perf=perf)
