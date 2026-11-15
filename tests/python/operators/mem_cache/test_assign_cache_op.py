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

# Generic segment copy; payload widths and segment sizes are model-independent.
perf_shape = [(1, 1024), (2, 1024), (50000, 100000)]


def generate_data(gap, width, dtype):
    generator = torch.Generator(device="cpu").manual_seed(29)
    starts = torch.randint(
        0, width - gap - 1, (96,), generator=generator, device="cpu"
    ).npu()
    pool = torch.randint(
        32, (96, width), dtype=dtype, generator=generator, device="cpu"
    ).npu()
    src = torch.randint(
        32, (96 * gap,), dtype=dtype, generator=generator, device="cpu"
    ).npu()
    indices = torch.arange(48, dtype=torch.int32, device="npu")
    return indices, pool, starts, starts + gap, src


def _assign_to_pool(pool, indices, starts, ends, src, impl):
    from sgl_kernel_npu.operators.mem_cache import assign_cache_op

    lengths = ends - starts
    selected = pool[indices]
    source_ends = torch.cumsum(lengths, dim=0)
    source_starts = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=pool.device), source_ends)
    )
    args = (selected, src, starts, ends, source_starts, source_ends)
    pointer = selected.data_ptr()
    assert assign_cache_op(*args, impl=impl) is True
    assert selected.data_ptr() == pointer
    pool[indices] = selected
    return args


@pytest.mark.parametrize(
    "gap,width",
    [
        pytest.param(1, 1024, id="single-element-segment"),
        pytest.param(2, 1024, id="two-element-segment"),
        pytest.param(50000, 100000, id="large-segment-50k-in-100k"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int8, id="payload-int8-1byte"),
        pytest.param(torch.int16, id="payload-int16-2bytes"),
        pytest.param(torch.int32, id="payload-int32-4bytes"),
        pytest.param(torch.int64, id="payload-int64-8bytes"),
    ],
)
def test_assign_cache_op(gap, width, dtype):
    from sgl_kernel_npu.operators.mem_cache import assign_cache_op

    indices, pool, starts, ends, src = generate_data(gap, width, dtype)
    actual, reference = pool.clone(), pool.clone()
    inputs = (indices, starts, ends, src)
    saved = tuple(t.clone() for t in inputs)
    impl, impl_ref = get_test_impls()
    skip = bool(get_skip_acc_check())
    if skip:
        require_test_impl(assign_cache_op, impl)
    else:
        require_distinct_impls(assign_cache_op, impl, impl_ref)
    pointer = actual.data_ptr()
    for iteration in range(32):
        torch.npu.synchronize()
        if not skip:
            _assign_to_pool(reference, indices, starts, ends, src, impl_ref)
            torch.npu.synchronize()
        args = _assign_to_pool(actual, indices, starts, ends, src, impl)
        torch.npu.synchronize()
        assert actual.data_ptr() == pointer
        if not skip:
            assert_accuracy(actual, reference, path=f"iteration[{iteration}].output")
        assert_accuracy(inputs, saved, path=f"iteration[{iteration}].inputs")
    executor = make_executor(
        assign_cache_op,
        args=args,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip,
    )
    check_performance(
        executor, enable_perf=(gap, width) in perf_shape and dtype == torch.int32
    )
