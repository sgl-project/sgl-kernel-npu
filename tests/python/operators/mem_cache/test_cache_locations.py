"""Cache-location inputs, accuracy, in-place writes, round trips and performance."""

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

CACHE_LOC_CASES = [
    pytest.param((8, 64), (1,), 0, False, id="minimum"),
    pytest.param((4, 32), (2, 1, 3), 2, False, id="mixed-with-tail"),
    pytest.param(
        (64, 256), (0, 1, 0, 16, 3) * 7 + (2, 1), 4, False, id="batch-tail-37"
    ),
    # SGLang speculative cache_locs uses packed B * draft_token_num int32 slots.
    pytest.param((512, 8192), (1,) * 64, 0, False, id="speculative-step-1"),
    pytest.param((512, 8192), (16,) * 128, 0, False, id="maximum-step"),
    # Exercise a batch of 300 requests with alternating interval lengths.
    pytest.param((2000, 16384), (1, 2) * 150, 0, False, id="batch-300-mixed"),
    pytest.param((32, 128), (4,) * 16, 0, True, id="matrix-locations"),
    # None selects B300 random step1/2 data across the full start range.
    pytest.param((2000, 16384), None, 0, False, id="batch-300-width-16384-random"),
    pytest.param((2000, 8192), None, 0, False, id="batch-300-width-8192-random"),
]

# Batch and interval-length combinations for changed-value round trips.
ROUNDTRIP_CASES = [
    (1, 4),
    (9, 4),
    (15, 4),
    (16, 4),
    (64, 1),
    (300, 2),
    (37, 3),
    (128, 16),
]


def generate_test_data(pool_shape, lengths, tail, matrix, req_dtype):
    """Generate random intervals; bounded cases keep 16-slot row reads in range."""
    torch.npu.manual_seed(42)
    device = torch.device("npu", torch.npu.current_device())
    rows, width = pool_shape
    if lengths is None:
        # B300 random step1/2 workload, including starts near the row end.
        requests = torch.arange(300, device=device, dtype=req_dtype)
        pool = torch.arange(width, device=device, dtype=torch.int32).repeat(rows, 1)
        starts = torch.randint(2, width, (300,), device=device, dtype=torch.int64) - 2
        ends = starts + torch.randint(1, 3, (300,), device=device, dtype=torch.int64)
        locations = torch.randint(
            0,
            10000,
            (int((ends - starts).sum().item()),),
            device=device,
            dtype=torch.int32,
        )
        return requests, pool, starts, ends, locations
    requests = torch.randperm(rows, device=device)[: len(lengths)].to(req_dtype)
    pool = torch.arange(rows * width, device=device, dtype=torch.int32).view(pool_shape)
    starts = torch.randint(
        0, width - 16 + 1, (len(lengths),), device=device, dtype=torch.int64
    )
    ends = starts + torch.tensor(lengths, device=device, dtype=torch.int64)
    count = sum(lengths)
    locations = torch.full((count + tail,), -777, device=device, dtype=torch.int32)
    # Different source/destination values make a missing write observable.
    locations[:count] = -1000000 - torch.arange(count, device=device, dtype=torch.int32)
    if matrix:
        locations = locations.view(8, 8)
    return requests, pool, starts, ends, locations


def generate_roundtrip_data(batch, step, req_dtype, seed=0, *, device="npu"):
    """Seeded random inputs, independent of other test suites."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, width = 512, 8192
    pool = torch.randint(
        0, 1 << 20, (rows, width), generator=generator, dtype=torch.int32, device="cpu"
    ).to(device)
    requests = torch.randperm(rows, generator=generator, device="cpu")[:batch].to(
        device=device, dtype=req_dtype
    )
    starts = torch.randint(
        0, width - step, (batch,), generator=generator, dtype=torch.int64, device="cpu"
    ).to(device)
    ends = starts + step
    locations = torch.empty((batch * step,), dtype=pool.dtype, device=device)
    return pool, requests, starts, ends, locations


@pytest.mark.parametrize("req_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pool_shape,lengths,tail,matrix", CACHE_LOC_CASES)
def test_cache_loc_assign(pool_shape, lengths, tail, matrix, req_dtype):
    # Compare independent in-place buffers, unchanged inputs and the returned alias.
    from sgl_kernel_npu.operators.mem_cache import cache_loc_assign

    inputs = generate_test_data(pool_shape, lengths, tail, matrix, req_dtype)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(cache_loc_assign, impl)
    else:
        require_distinct_impls(cache_loc_assign, impl, impl_ref)
        reference = tuple(t.clone() for t in inputs)
        result_ref = cache_loc_assign(*reference, impl=impl_ref)
    values = tuple(t.clone() for t in inputs)
    result = cache_loc_assign(*values, impl=impl)
    if not skip_accuracy:
        # The complete output includes untouched pool positions or the packed tail.
        assert_accuracy(result, result_ref)
    assert result.data_ptr() == values[1].data_ptr()
    for i in (0, 2, 3, 4):
        assert_accuracy(values[i], inputs[i])

    # Repeat the same inputs; input preparation stays outside profiling.
    executor = make_executor(
        cache_loc_assign,
        args=values,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    # Measure the B300 assign workload for both request dtypes.
    enable_perf = (lengths is None and pool_shape == (2000, 16384)) or (
        lengths is not None and len(lengths) in (64, 128) and req_dtype == torch.int64
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("req_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pool_shape,lengths,tail,matrix", CACHE_LOC_CASES)
def test_cache_loc_update(pool_shape, lengths, tail, matrix, req_dtype):
    # Compare independent in-place buffers, unchanged inputs and the returned alias.
    from sgl_kernel_npu.operators.mem_cache import cache_loc_update

    inputs = generate_test_data(pool_shape, lengths, tail, matrix, req_dtype)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    if skip_accuracy:
        require_test_impl(cache_loc_update, impl)
    else:
        require_distinct_impls(cache_loc_update, impl, impl_ref)
        reference = tuple(t.clone() for t in inputs)
        result_ref = cache_loc_update(*reference, impl=impl_ref)
    values = tuple(t.clone() for t in inputs)
    result = cache_loc_update(*values, impl=impl)
    if not skip_accuracy:
        # The complete output includes untouched pool positions or the packed tail.
        assert_accuracy(result, result_ref)
    assert result.data_ptr() == values[4].data_ptr()
    for i in (0, 1, 2, 3):
        assert_accuracy(values[i], inputs[i])

    # Repeat the same inputs; input preparation stays outside profiling.
    executor = make_executor(
        cache_loc_update,
        args=values,
        impl=impl,
        impl_ref=impl_ref,
        accuracy_checked=not skip_accuracy,
    )
    # Measure the B300 update workload for both request dtypes.
    enable_perf = (lengths is None and pool_shape == (2000, 8192)) or (
        lengths is not None and len(lengths) in (64, 128) and req_dtype == torch.int64
    )
    check_performance(executor, enable_perf=enable_perf)


@pytest.mark.parametrize("req_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch,step", ROUNDTRIP_CASES)
def test_cache_loc_changed_roundtrip(batch, step, req_dtype):
    # Compare random inputs against the selected reference backend.
    from sgl_kernel_npu.operators.mem_cache import cache_loc_assign, cache_loc_update

    pool, req, starts, ends, locations = generate_roundtrip_data(
        batch, step, req_dtype, seed=0
    )
    inputs = (req, pool, starts, ends, locations)
    values = tuple(t.clone() for t in inputs)
    impl, impl_ref = get_test_impls()
    skip_accuracy = bool(get_skip_acc_check())
    for operator in (cache_loc_assign, cache_loc_update):
        if skip_accuracy:
            require_test_impl(operator, impl)
        else:
            require_distinct_impls(operator, impl, impl_ref)
    if not skip_accuracy:
        reference = tuple(t.clone() for t in inputs)
        result_ref = cache_loc_update(*reference, impl=impl_ref)
    result = cache_loc_update(*values, impl=impl)
    assert result.data_ptr() == values[4].data_ptr()
    assert_accuracy(values[1], pool)
    if not skip_accuracy:
        assert_accuracy(result, result_ref)

    # A changed source prevents a no-op scatter from passing the round trip.
    values[4].add_(1)
    changed_locations = values[4].clone()
    if not skip_accuracy:
        reference[4].add_(1)
        result_ref = cache_loc_assign(*reference, impl=impl_ref)
    result = cache_loc_assign(*values, impl=impl)
    assert result.data_ptr() == values[1].data_ptr()
    assert_accuracy(values[4], changed_locations)
    for index in (0, 2, 3):
        assert_accuracy(values[index], inputs[index])
    if not skip_accuracy:
        assert_accuracy(result, result_ref)
