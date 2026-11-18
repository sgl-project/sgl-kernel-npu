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

- [Upstream sparsity-driven KV offloading primitives](https://github.com/sgl-project/sgl-kernel-npu/blob/4a401a603f87d09a9c59ec195b47f3fad81d8b56/csrc/sparsity_driven_kv_offload/README.md)
- [CANN 9.1 host memory management and device address mapping](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0006.html)
- [PyTorch `torch.frombuffer`](https://docs.pytorch.org/docs/2.10/generated/torch.frombuffer.html)
