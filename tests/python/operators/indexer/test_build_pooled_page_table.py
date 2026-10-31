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


# Representative edge cases (pool extremes, rank, dtype) plus typical
# GLM-5.3-Flash decode workloads.
@pytest.mark.parametrize(
    "shape,pool_size,dtype",
    [
        # Case 01: Pool1 preserves every page ID.
        pytest.param((3, 9), 1, torch.int32, id="edge-pool1"),
        # Case 02: Rank3 page table with int64 IDs.
        pytest.param((2, 3, 5), 4, torch.int64, id="edge-rank3-int64"),
        # Case 03: Pool128 sampling with a remainder.
        pytest.param((1, 129), 128, torch.int32, id="edge-largest-pool"),
        # Case 04: Typical decode B16/M64; also profile.
        pytest.param((16, 64), 4, torch.int32, id="W-glm53-flash-decode-b16-m64"),
        # Case 05: Typical decode B64/M129 with a remainder; also profile.
        pytest.param((64, 129), 4, torch.int32, id="W-glm53-flash-decode-page-tail"),
    ],
)
def test_build_pooled_page_table(shape, pool_size, dtype):
    from sgl_kernel_npu.operators.indexer import build_pooled_page_table

    page_table = generate_test_data(shape, dtype)
    impl, _ = get_test_impls()
    require_test_impl(build_pooled_page_table, impl)
    accuracy_checked = get_skip_acc_check() is not True

    # Special accuracy case: auto/native alias one implementation, so compare
    # against independent slicing rather than using check_accuracy.
    if accuracy_checked:
        original = page_table.clone()
        expected = original[..., ::pool_size]
        actual = build_pooled_page_table(page_table, pool_size, impl=impl)
        assert_accuracy(actual, expected)
        assert_accuracy(page_table, original)

    executor = partial(build_pooled_page_table, page_table, pool_size, impl=impl)
    executor.impl = impl
    executor.impl_ref = "independent_slicing"
    executor.accuracy_checked = accuracy_checked

    # Performance: measure only the DUT on the selected representative shapes.
    check_performance(
        executor,
        enable_perf=shape in perf_shape and pool_size == 4 and dtype == torch.int32,
    )
