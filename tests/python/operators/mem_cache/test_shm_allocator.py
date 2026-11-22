import pytest

torch = pytest.importorskip("torch")

from tests.python.utils.accuracy import assert_accuracy


def test_create_and_free_shm_tensor():
    from sgl_kernel_npu.operators.mem_cache import create_shm_tensor, free_shm
    from sgl_kernel_npu.operators.utils.generation import Generation, get_hardware

    if get_hardware().generation is not Generation.ARCH22:
        pytest.skip("Shared memory registration requires ARCH22")

    device_id = torch.npu.current_device()
    expected = torch.arange(1, 9, dtype=torch.int32, device="cpu").reshape(2, 4)
    for _ in range(2):
        # Allocating the same name after release must return cleared storage.
        tensor, host_ptr, dev_ptr = create_shm_tensor(
            expected.shape, expected.dtype, device_id, name="test_shm_allocator"
        )
        try:
            assert host_ptr == tensor.data_ptr() != 0
            assert dev_ptr > 0
            assert_accuracy(tensor, torch.zeros_like(expected))
            tensor.copy_(expected)
            assert_accuracy(tensor, expected)
        finally:
            del tensor
            free_shm(device_id)
