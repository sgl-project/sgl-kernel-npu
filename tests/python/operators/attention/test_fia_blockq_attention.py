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

perf_shape = [(17, 16, 128)]


def generate_data(shape, reuse):
    tokens, heads, dim = shape
    torch.manual_seed(42)
    block_size, pages = 128, 8
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn(pages, block_size, 1, dim, dtype=q.dtype)
    v = torch.randn_like(k)
    topk = torch.tensor([0, 1, -1, 2], dtype=torch.int32).repeat(1, tokens, 1)
    lengths = torch.arange(257, 257 + tokens, dtype=torch.int32)
    requests = torch.zeros(tokens, dtype=torch.int32)
    mapping = torch.arange(pages * block_size, dtype=torch.int32).view(1, -1)
    table = torch.empty(tokens, 4, dtype=torch.int32) if reuse else None
    actual_len = torch.empty(tokens, dtype=torch.int32) if reuse else None
    return (
        q,
        k,
        v,
        topk,
        lengths,
        requests,
        mapping,
        block_size,
        None,
        pages,
        4,
        table,
        actual_len,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 16, 128), id="minimax-m3-tp4-single-query"),
        pytest.param((17, 16, 128), id="minimax-m3-tp4-prefill"),
    ],
)
def test_fia_blockq_attention(shape):
    reuse = True
    from sgl_kernel_npu.operators.attention import flash_prefill_bnsd_blockq_sparse_fia

    args = generate_data(shape, reuse)
    impl, impl_ref = get_test_impls()
    executor = make_executor(flash_prefill_bnsd_blockq_sparse_fia, args=args)
    if not get_skip_acc_check():
        require_distinct_impls(flash_prefill_bnsd_blockq_sparse_fia, impl, impl_ref)
        reference = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in args)
        expected = flash_prefill_bnsd_blockq_sparse_fia(*reference, impl=impl_ref)
        actual = executor()
        assert_accuracy(actual, expected)
        if reuse:
            assert_accuracy(args[-2:], reference[-2:])
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)
