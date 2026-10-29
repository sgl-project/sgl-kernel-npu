"""Changed-value gather/scatter through public entries."""

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_distinct_impls,
    require_test_impl,
)

from ._cache_loc_inputs import ROUNDTRIP_CASES, generate_roundtrip_data


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
