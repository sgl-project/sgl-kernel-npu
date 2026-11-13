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

perf_shape = [(18, 100, 4, 4, 64, 64, 50)]
exact = {"torch.bfloat16": AccuracyStandard(0, 0)}


def generate_move_data(shape, transposed, dtype=torch.bfloat16):
    layers, pool, steps, heads, vdim, kdim, valid = shape
    torch.manual_seed(42)
    src = torch.randn(layers, pool, steps, heads, vdim, kdim, dtype=dtype)
    dst = torch.randn(
        layers,
        pool,
        heads,
        kdim if transposed else vdim,
        vdim if transposed else kdim,
        dtype=dtype,
    )
    if transposed:
        dst = dst.transpose(-1, -2)
    indices = torch.randperm(pool)[:valid].to(torch.int32)
    source = torch.arange(valid, dtype=torch.int32)
    last = torch.arange(valid, dtype=torch.int32) % steps
    if valid > 2:
        last[1] = -1
    return dst, src, indices, source, last


def _check_inplace(operator, args, monitor):
    impl, impl_ref = get_test_impls()
    executor = make_executor(operator, args=args)
    if not get_skip_acc_check():
        require_distinct_impls(operator, impl, impl_ref)
        reference = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in args)
        expected = operator(*reference, impl=impl_ref)
        actual = executor()
        assert_accuracy(args[0], reference[0], standard=exact)
        if actual is not None:
            assert actual.data_ptr() == args[0].data_ptr()
            assert actual.stride() == args[0].stride()
        assert_accuracy(args[1:], reference[1:], standard=exact)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=monitor)


@pytest.mark.parametrize(
    "shape",
    [
        (36, 229, 4, 8, 128, 128, 180),
        (18, 100, 4, 4, 64, 64, 50),
        (36, 229, 4, 8, 128, 128, 229),
        (18, 100, 4, 4, 64, 64, 100),
        (2, 5, 4, 2, 65, 16, 3),
        (2, 5, 4, 2, 128, 16, 3),
        (2, 5, 4, 2, 129, 16, 3),
    ],
)
@pytest.mark.parametrize(
    "kda,transposed", [(False, False), (False, True), (True, True)]
)
def test_move_cache(shape, kda, transposed):
    from sgl_kernel_npu.operators.mamba import (
        move_intermediate_cache,
        move_intermediate_cache_kda,
    )

    # GDN and KDA verification copy layer states; KDA preserves logical V/K strides.
    op = move_intermediate_cache_kda if kda else move_intermediate_cache
    _check_inplace(op, generate_move_data(shape, transposed), shape in perf_shape)


def generate_rollback_data(shape, transposed):
    layers, pool, dims, draft, requests = shape
    window = 3 + draft - 1
    torch.manual_seed(42)
    storage = torch.randn(
        layers,
        pool,
        dims if transposed else window,
        window if transposed else dims,
        dtype=torch.bfloat16,
    )
    states = storage.transpose(-1, -2) if transposed else storage
    indices = torch.randperm(pool)[:requests].to(torch.int32)
    steps = torch.arange(requests, dtype=torch.int32) % (draft + 1) - 1
    return states, indices, steps, draft


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 2048, 3, 3),
        (16, 16, 1024, 3, 2),
        (32, 32, 2048, 4, 4),
        (16, 16, 1024, 4, 1),
        (2, 5, 64, 4, 3),
    ],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_rollback(shape, transposed):
    from sgl_kernel_npu.operators.mamba import conv_state_rollback

    _check_inplace(
        conv_state_rollback,
        generate_rollback_data(shape, transposed),
        shape == (32, 32, 2048, 4, 4),
    )


@pytest.mark.extra_checks
@pytest.mark.parametrize("name", ["move_intermediate_cache_kda", "conv_state_rollback"])
def test_empty(name):
    from sgl_kernel_npu.operators import mamba

    impl, _ = get_test_impls()
    if name == "conv_state_rollback":
        args = generate_rollback_data((2, 5, 64, 4, 0), True)
        assert getattr(mamba, name)(*args, impl=impl) is None
    else:
        args = generate_move_data((2, 5, 4, 2, 128, 128, 0), True)
        assert getattr(mamba, name)(*args, impl=impl) is args[0]


def generate_different_pool_data(vdim, physical):
    layers, source_pool, steps, heads, kdim = (
        2,
        4 if physical else 5,
        3 if physical else 4,
        2,
        16,
    )
    values = torch.arange(
        layers * source_pool * steps * heads * vdim * kdim, dtype=torch.int64
    )
    source = (
        (values % 2048)
        .to(torch.bfloat16)
        .reshape(layers, source_pool, steps, heads, vdim, kdim)
    )
    if physical:
        destination = torch.full(
            (layers, source_pool + 2, heads, kdim, vdim), -7, dtype=torch.bfloat16
        ).transpose(-1, -2)
        dst, src, last = [0, 3, 5], [1, 2, 3], [2, -1, 0]
    else:
        destination = torch.full(
            (layers, source_pool + 2, heads, vdim, kdim), -1, dtype=torch.bfloat16
        )
        dst, src, last = [1, 4, 6], [0, 2, 4], [0, steps - 1, 1]
    return (
        destination,
        source,
        torch.tensor(dst, dtype=torch.int32),
        torch.tensor(src, dtype=torch.int32),
        torch.tensor(last, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "vdim,physical", [(65, False), (128, False), (129, False), (128, True)]
)
def test_different_pool_extents(vdim, physical):
    from sgl_kernel_npu.operators.mamba import move_intermediate_cache

    _check_inplace(
        move_intermediate_cache, generate_different_pool_data(vdim, physical), False
    )
