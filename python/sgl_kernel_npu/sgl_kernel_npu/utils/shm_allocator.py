"""NPU-registered host shared memory."""

import ctypes
from math import prod
from typing import Sequence, Tuple

import torch


def _ctype_for_dtype(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return ctypes.c_uint16
    if dtype == torch.float32:
        return ctypes.c_float
    if dtype == torch.float64:
        return ctypes.c_double
    if dtype == torch.int8:
        return ctypes.c_int8
    if dtype == torch.uint8:
        return ctypes.c_uint8
    if dtype == torch.int16:
        return ctypes.c_int16
    if dtype == torch.int32:
        return ctypes.c_int32
    if dtype == torch.int64:
        return ctypes.c_int64
    if dtype == torch.bool:
        return ctypes.c_bool
    raise TypeError(f"unsupported shm tensor dtype: {dtype}")


def create_shm_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    device_id: int = 0,
    name: str = "",
) -> Tuple[torch.Tensor, int, int]:
    """Create host shared memory and register it to an NPU device.

    Returns ``(host_tensor, host_ptr, dev_ptr)``. ``host_tensor`` is a CPU
    tensor backed by the registered shared memory. ``dev_ptr`` is the
    device-visible address and can be passed to sparse KV kernels through
    ``src_ptr``/``dst_ptr``.
    """
    shape_tuple = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape_tuple):
        raise ValueError(f"shape dimensions must be non-negative, got {shape_tuple}")

    numel = int(prod(shape_tuple))
    elem_size = torch.empty((), dtype=dtype).element_size()
    size = numel * elem_size
    if size <= 0:
        raise ValueError(f"shm tensor size must be positive, got shape={shape_tuple}")

    host_ptr, dev_ptr = torch.ops.npu.shm_allocator_create_and_register(
        size, device_id, name
    )
    buffer_type = _ctype_for_dtype(dtype) * numel
    buffer = buffer_type.from_address(host_ptr)
    tensor = torch.frombuffer(buffer, dtype=dtype).view(shape_tuple)
    if tensor.element_size() != elem_size:
        raise RuntimeError("shm tensor element size mismatch")
    return tensor, int(host_ptr), int(dev_ptr)


def free_shm(device_id: int = 0) -> None:
    """Free all shared-memory allocations registered by this process."""
    torch.ops.npu.shm_allocator_free_all(device_id)
