import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy
from tests.python.utils.common import (
    get_skip_acc_check,
    get_test_impls,
    make_executor,
    require_distinct_impls,
)
from tests.python.utils.performance import check_performance

# Paged allocation is model-independent; include empty extensions and page boundaries.
perf_shape = [(128, 17)]


def generate_data(page_size, batch, dtype):
    prefix = ([0, 1, page_size - 1, page_size] * batch)[:batch]
    extend = ([0, 1, page_size + 1, 2 * page_size] * batch)[:batch]
    pre = torch.tensor(prefix, dtype=dtype)
    seq = pre + torch.tensor(extend, dtype=dtype)
    last = torch.tensor([p - 1 for p in prefix], dtype=dtype)
    pages = torch.arange(100, 100 + batch * 4, dtype=dtype)
    out = torch.full((sum(extend),), -1, dtype=dtype)
    return pre, seq, last, pages, page_size, out, max(extend)


@pytest.mark.parametrize(
    "page_size,batch,dtype", [(16, 4, torch.int64), (128, 17, torch.int64)]
)
def test_alloc_extend_indices(page_size, batch, dtype):
    from sgl_kernel_npu.operators.mem_cache import alloc_extend_indices

    args = generate_data(page_size, batch, dtype)
    impl, impl_ref = get_test_impls()
    executor = make_executor(alloc_extend_indices, args=args)
    if not get_skip_acc_check():
        require_distinct_impls(alloc_extend_indices, impl, impl_ref)
        reference = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in args)
        assert alloc_extend_indices(*reference, impl=impl_ref) is None
        assert executor() is None
        assert_accuracy(args[5], reference[5])
        assert_accuracy(args[:4], reference[:4])
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=(page_size, batch) in perf_shape)
