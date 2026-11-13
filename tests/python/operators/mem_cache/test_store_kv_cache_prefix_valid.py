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
from tests.python.utils.standards import AccuracyStandard

# Generic speculative KV commit: no model-specific arithmetic or fixed head count.
perf_shape = [(16, 4, 8, 128)]


def generate_data(shape, dtype):
    batch, width, heads, dim = shape
    torch.manual_seed(42)
    rows = batch * width
    dk = torch.randn(rows + 16, heads, dim, dtype=dtype)
    dv = torch.randn(rows + 16, heads, dim // 2, dtype=dtype)
    sk = torch.randn(rows, heads, dim, dtype=dtype)
    sv = torch.randn(rows, heads, dim // 2, dtype=dtype)
    loc = torch.arange(8, 8 + rows, dtype=torch.int64).reshape(batch, width)
    commit = torch.arange(batch, dtype=torch.int32) % (width + 1)
    loc[torch.arange(width)[None, :] >= commit[:, None]] = -1
    return dk, dv, sk, sv, loc, commit


@pytest.mark.parametrize(
    "shape,dtype", [((4, 3, 2, 64), torch.float16), ((16, 4, 8, 128), torch.bfloat16)]
)
def test_store_prefix(shape, dtype):
    from sgl_kernel_npu.operators.mem_cache import store_kv_cache_prefix_valid

    args = generate_data(shape, dtype)
    impl, impl_ref = get_test_impls()
    executor = make_executor(store_kv_cache_prefix_valid, args=args)
    if not get_skip_acc_check():
        require_distinct_impls(store_kv_cache_prefix_valid, impl, impl_ref)
        reference = tuple(x.clone() for x in args)
        assert store_kv_cache_prefix_valid(*reference, impl=impl_ref) is None
        assert executor() is None
        exact = {str(dtype): AccuracyStandard(0, 0)}
        assert_accuracy(args, reference, standard=exact)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)
