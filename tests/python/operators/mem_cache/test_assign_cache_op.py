"""Packed segment copies: integer payload coverage, mutation and performance."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance


@pytest.mark.parametrize("gap,width", [(1, 1024), (2, 1024), (50000, 100000)])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_assign_cache_op(gap, width, dtype):
    # Cover small tails and multi-UB copies, with independent reference and DUT storage.
    from sgl_kernel_npu.operators.mem_cache import assign_cache_op

    generator = torch.Generator(device="cpu").manual_seed(29)
    starts = torch.randint(
        0, width - gap - 1, (96,), generator=generator, device="cpu"
    ).npu()
    source_starts = torch.arange(97, dtype=torch.int64, device="npu") * gap
    pool = torch.randint(
        32, (96, width), dtype=dtype, generator=generator, device="cpu"
    ).npu()
    src = torch.randint(
        32, (96 * gap,), dtype=dtype, generator=generator, device="cpu"
    ).npu()
    actual, reference = pool.clone(), pool.clone()
    args = (actual[:48], src, starts, starts + gap, source_starts, source_starts[1:])
    saved = tuple(t.clone() for t in args[1:])
    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(assign_cache_op, impl)
    else:
        require_distinct_impls(assign_cache_op, impl, impl_ref)
    pointer = actual.data_ptr()
    # Check every iteration, independently of whether performance profiling is enabled.
    for iteration in range(32):
        torch.npu.synchronize()
        if not skip:
            assert assign_cache_op(reference[:48], *args[1:], impl=impl_ref) is True
            torch.npu.synchronize()
        assert assign_cache_op(*args, impl=impl) is True
        torch.npu.synchronize()
        assert actual.data_ptr() == pointer
        if not skip:
            assert_accuracy(actual, reference, path=f"iteration[{iteration}].output")
        assert_accuracy(args[1:], saved, path=f"iteration[{iteration}].inputs")
    executor = make_executor(
        assign_cache_op,
        args=args,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )
    check_performance(executor, enable_perf=dtype == torch.int32)
