# Shared memory allocator

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| HOST C++ (Ascend HAL) | Supported | Unsupported |

## Signature

```python
from sgl_kernel_npu.operators.mem_cache import create_shm_tensor, free_shm


def create_shm_tensor(shape, dtype, device_id=0, name=""): ...
def free_shm(device_id=0): ...
```

## Description

`create_shm_tensor` allocates zero-initialized shared host memory and registers it with the NPU. It returns a CPU tensor backed by that memory and two addresses that refer to the same buffer. Pass the device address as `src_ptr` or `dst_ptr` to [unidex_copy_inplace](unidex_copy_inplace.md) for sparse KV offload.

The allocator tracks memory per process. Use one NPU device per process. `free_shm` unregisters and releases all allocations tracked by the process. Finish all NPU work that accesses the memory before freeing it. The returned tensors and addresses are valid until the memory is released.

## Parameters and return value

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `shape` | Sequence[int] | Required | Tensor dimensions. The resulting storage size must be positive. |
| `dtype` | torch.dtype | Required | FP16, BF16, FP32, FP64, INT8, UINT8, INT16, INT32, INT64 or Bool. |
| `device_id` | int | `0` | NPU device index used to register or unregister the memory. Use the same device for both calls. |
| `name` | str | `""` | Allocation name. An empty string generates a name. Reusing a name with the same byte size returns the existing allocation and preserves its contents. |

| Return | Type | Description |
| --- | --- | --- |
| `host_tensor` | Tensor | CPU view of the shared buffer with the requested shape and dtype. |
| `host_ptr` | int | Address of the buffer in host memory. |
| `dev_ptr` | int | Address of the buffer mapped for NPU access. |

`create_shm_tensor` returns `(host_tensor, host_ptr, dev_ptr)`. `free_shm` returns `None`.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.mem_cache import create_shm_tensor, free_shm

device_id = torch.npu.current_device()
host_tensor, host_ptr, dev_ptr = create_shm_tensor((2, 32), torch.float16, device_id)
try:
    host_tensor.fill_(1)
    assert host_tensor.data_ptr() == host_ptr
    assert dev_ptr > 0
finally:
    torch.npu.synchronize()
    free_shm(device_id)
```

## References

- [Ascend HAL driver documentation for `halHostRegister` and `halHostUnregister`](https://gitcode.com/cann/driver/blob/master/src/ascend_hal/svm/README_en.md#memory-sharing)
