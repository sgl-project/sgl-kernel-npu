import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import check_accuracy
from tests.python.utils.common import get_test_impls, require_test_impl
from tests.python.utils.performance import check_performance

perf_shape = [(3, 163848, 2048)]


def generate_data(shape):
    slots, context, topk = shape
    slot_map = torch.full((slots, context), -1, dtype=torch.int32)
    slot_map[0, 0] = 10
    slot_map[0, -1] = 11
    slot_map[1, 7] = 20
    indices = torch.zeros((3, topk), dtype=torch.int32)
    indices[0, :3] = torch.tensor([context - 1, -1, context], dtype=torch.int32)
    indices[1].fill_(7)
    req_indices = torch.tensor([0, 1, -1], dtype=torch.int32)
    return slot_map, req_indices, indices


@pytest.mark.parametrize(
    "shape,block_dim",
    [
        *[
            pytest.param((3, 64, 2048), blocks, id=f"invalid-indices-cores{blocks}")
            for blocks in (0, 1, 2, 8, 16, 24, 32, 48)
        ],
        pytest.param(perf_shape[0], 0, id="deepseek-v3-2-index-topk"),
    ],
)
def test_slot_map_lookup(shape, block_dim):
    from sgl_kernel_npu.operators.mem_cache import slot_map_lookup

    impl, impl_ref = get_test_impls()
    require_test_impl(slot_map_lookup, impl)
    if {impl, impl_ref} & {"auto", "ascendc"}:
        cores = torch.npu.get_device_properties(
            torch.npu.current_device()
        ).vector_core_num
        if block_dim > cores:
            pytest.skip(
                f"block_dim={block_dim} exceeds the device's {cores} vector cores"
            )
    args = generate_data(shape)
    executor = check_accuracy(slot_map_lookup, args=(*args, block_dim))
    check_performance(executor, enable_perf=shape in perf_shape)


@pytest.mark.extra_checks
@pytest.mark.parametrize(
    "topk,slot_dtype,message",
    [
        (64, torch.int32, "requires topk=2048"),
        (1, torch.int64, "slot_map must be int32"),
    ],
    ids=["non-fixed-topk", "slot-dtype"],
)
def test_slot_map_lookup_rejects_invalid_input(topk, slot_dtype, message):
    from sgl_kernel_npu.operators.mem_cache import slot_map_lookup

    impl, _ = get_test_impls()
    require_test_impl(slot_map_lookup, impl)
    slot_map = torch.full((3, 64), -1, dtype=slot_dtype)
    req_indices = torch.tensor([0], dtype=torch.int32)
    indices = torch.zeros((1, topk), dtype=torch.int32)
    with pytest.raises(RuntimeError, match=message):
        slot_map_lookup(slot_map, req_indices, indices, impl=impl)
