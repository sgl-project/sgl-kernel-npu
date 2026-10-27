from functools import partial

import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    require_test_impl,
)
from tests.python.utils.performance import check_performance

# W: decode batch/page-table candidates; grouping=4 follows GLM-5.3-Flash.
perf_shape = [(16, 64), (64, 129)]


def generate_test_data(shape, dtype):
    torch.manual_seed(42)
    return torch.randint(0, 100000, shape, dtype=dtype)


# Typical workloads plus minimum, grouping tails, ranks, empty leading axes,
# and legal pool extremes; dtype is sampled rather than fully crossed.
@pytest.mark.parametrize(
    "shape,pool_size,dtype",
    [
        pytest.param((1,), 4, torch.int32, id="edge-minimum"),
        pytest.param((3,), 4, torch.int32, id="edge-short-group"),
        pytest.param((4,), 4, torch.int32, id="edge-exact-group"),
        pytest.param((5,), 4, torch.int32, id="edge-group-tail"),
        pytest.param((2, 3, 5), 4, torch.int64, id="edge-rank3-int64"),
        pytest.param((2, 0, 5), 4, torch.int32, id="edge-empty-leading-axis"),
        pytest.param((3, 9), 1, torch.int32, id="edge-pool1"),
        pytest.param((3, 9), 2, torch.int32, id="edge-pool2"),
        pytest.param((3, 17), 8, torch.int32, id="edge-pool8"),
        pytest.param((1, 129), 128, torch.int32, id="edge-largest-pool"),
        pytest.param((16, 64), 4, torch.int32, id="W-glm53-flash-decode-b16-m64"),
        pytest.param((64, 129), 4, torch.int32, id="W-glm53-flash-decode-page-tail"),
    ],
)
def test_build_pooled_page_table_64(shape, pool_size, dtype):
    from sgl_kernel_npu.operators.mem_cache import build_pooled_page_table_64

    page_table = generate_test_data(shape, dtype)
    impl, _ = get_test_impls()
    require_test_impl(build_pooled_page_table_64, impl)
    accuracy_checked = get_skip_acc_check() is not True
    if accuracy_checked:
        original = page_table.clone()
        # auto/native alias the same entry: retain independent slicing.
        expected = original[..., ::pool_size]
        actual = build_pooled_page_table_64(page_table, pool_size, impl=impl)
        assert_accuracy(actual, expected)
        assert_accuracy(page_table, original)
    executor = partial(build_pooled_page_table_64, page_table, pool_size, impl=impl)
    executor.impl = impl
    executor.impl_ref = "independent_slicing"
    executor.accuracy_checked = accuracy_checked
    check_performance(
        executor,
        enable_perf=shape in perf_shape and pool_size == 4 and dtype == torch.int32,
    )
