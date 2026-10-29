"""Public cache-location gather accuracy, in-place writes and performance."""

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

from ._cache_loc_inputs import CACHE_LOC_CASES, generate_test_data


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
