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
from tests.python.utils.standards import AccuracyStandard

perf_shape = [(576, 2048)]
exact = {"torch.float16": AccuracyStandard(0, 0)}


def generate_data(shape):
    width, mappings = shape
    if mappings == 2048:
        src = torch.randn(1, mappings, 1, width, dtype=torch.float16)
        dst = torch.full_like(src, -1)
        dst_index = torch.arange(mappings, dtype=torch.int64)
        return src, dst, dst_index.flip(0), dst_index, dst_index % 2 == 0, 2, 2
    src = torch.arange(8 * width, dtype=torch.float16).reshape(2, 2, 2, width)
    dst = torch.full((2, 2, width), -1, dtype=torch.float16)
    src_index = torch.tensor([0, 3, 5, 7], dtype=torch.int64)
    dst_index = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    valid_mask = torch.tensor([True, False, True, True])
    return src, dst, src_index, dst_index, valid_mask, 3, 2


@pytest.mark.parametrize(
    "shape,raw_mode",
    [
        pytest.param((32, 4), None, id="aligned-rows"),
        pytest.param((15, 4), None, id="unaligned-rows"),
        pytest.param((32, 4), "src", id="raw-source"),
        pytest.param((32, 4), "dst", id="raw-destination"),
        pytest.param((32, 4), "both", id="raw-both"),
        pytest.param(perf_shape[0], None, id="deepseek-v3-2-mla-row"),
    ],
)
def test_unidex_copy(shape, raw_mode):
    from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace

    impl, impl_ref = get_test_impls()
    require_test_impl(unidex_copy_inplace, impl)
    if raw_mode and impl not in ("auto", "ascendc"):
        pytest.skip("Raw addresses require AscendC")
    src, dst, *mapping = generate_data(shape)
    kwargs = {"block_dim": 24 if shape in perf_shape else 8}
    if raw_mode in ("src", "both"):
        kwargs["src_ptr"] = src.data_ptr()
    if raw_mode in ("dst", "both"):
        kwargs["dst_ptr"] = dst.data_ptr()
    executor = make_executor(
        unidex_copy_inplace, args=(src, dst, *mapping), kwargs=kwargs
    )
    if not get_skip_acc_check():
        require_distinct_impls(unidex_copy_inplace, impl, impl_ref)
        expected = dst.clone()
        unidex_copy_inplace(
            src, expected, *mapping, block_dim=kwargs["block_dim"], impl=impl_ref
        )
        assert executor() is dst
        assert_accuracy(dst, expected, exact)
        executor.accuracy_checked = True
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_unidex_copy_registered_shm(direction):
    from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace
    from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, free_shm

    impl, _ = get_test_impls()
    require_test_impl(unidex_copy_inplace, impl)
    if impl not in ("auto", "ascendc"):
        pytest.skip("Registered host memory requires AscendC")
    device_id = torch.npu.current_device()
    source_cpu = torch.arange(4 * 32, dtype=torch.float16, device="cpu").reshape(4, 32)
    expected = torch.full_like(source_cpu, -1)
    expected[0] = source_cpu[3]
    expected[3] = source_cpu[0]
    host_tensor, _, address = create_shm_tensor(
        source_cpu.shape, source_cpu.dtype, device_id
    )

    try:
        if direction == "h2d":
            host_tensor.copy_(source_cpu)
            src, dst = host_tensor, torch.full((4, 32), -1, dtype=torch.float16)
            pointers = {"src_ptr": address}
        else:
            host_tensor.fill_(-1)
            src, dst = source_cpu.to("npu"), host_tensor
            pointers = {"dst_ptr": address}
        src_index = torch.tensor([3, 1, 0], dtype=torch.int64)
        dst_index = torch.tensor([0, 2, 3], dtype=torch.int64)
        mask = torch.tensor([True, False, True])
        result = unidex_copy_inplace(
            src, dst, src_index, dst_index, mask, 1, 1, impl=impl, **pointers
        )
        torch.npu.synchronize()
        assert result is dst
        assert_accuracy(dst.cpu(), expected, exact)
    finally:
        torch.npu.synchronize()
        free_shm(device_id)


@pytest.mark.extra_checks
def test_unidex_copy_empty_mapping():
    from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace

    impl, _ = get_test_impls()
    require_test_impl(unidex_copy_inplace, impl)
    src, dst, index, _, mask, src_dims, dst_dims = generate_data((32, 4))
    before = dst.clone()
    result = unidex_copy_inplace(
        src, dst, index[:0], index[:0], mask[:0], src_dims, dst_dims, impl=impl
    )
    assert result is dst
    assert_accuracy(dst, before, exact)


@pytest.mark.extra_checks
def test_unidex_copy_rejects_wrong_index_dtype():
    from sgl_kernel_npu.operators.mem_cache import unidex_copy_inplace

    impl, _ = get_test_impls()
    require_test_impl(unidex_copy_inplace, impl)
    src, dst, src_index, dst_index, mask, src_dims, dst_dims = generate_data((32, 4))
    with pytest.raises(RuntimeError, match="src_index must be int64"):
        unidex_copy_inplace(
            src, dst, src_index.int(), dst_index, mask, src_dims, dst_dims, impl=impl
        )
