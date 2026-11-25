# 共享内存分配器

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| HOST C++（Ascend HAL） | 支持 | 不支持 |

## 接口签名

```python
from sgl_kernel_npu.operators.mem_cache import create_shm_tensor, free_shm


def create_shm_tensor(shape, dtype, device_id=0, name=""): ...
def free_shm(device_id=0): ...
```

## 功能描述

`create_shm_tensor` 申请清零的 Host 共享内存并注册到 NPU。它返回一个由这块内存支持的 CPU Tensor，以及指向同一块缓冲区的 Host 地址和 NPU 映射地址。在稀疏 KV 卸载中，可将 NPU 映射地址作为 `src_ptr` 或 `dst_ptr` 传给 [unidex_copy_inplace](unidex_copy_inplace.md)。

分配记录在进程内维护。每个进程应绑定一张 NPU 卡。`free_shm` 注销并释放进程内记录的全部共享内存。释放前需等待访问该内存的 NPU 任务完成。返回的 Tensor 和地址仅在内存释放前有效。

## 参数与返回值

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `shape` | Sequence[int] | 必填 | Tensor 的形状，所需存储的字节数须大于零。 |
| `dtype` | torch.dtype | 必填 | 支持 FP16、BF16、FP32、FP64、INT8、UINT8、INT16、INT32、INT64 和 Bool。 |
| `device_id` | int | `0` | 注册或注销内存的 NPU 设备编号。两次调用使用相同的设备编号。 |
| `name` | str | `""` | 分配名称。传入空字符串时自动生成名称。名称和字节数相同则复用已有内存，并保留其中的数据。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `host_tensor` | Tensor | 共享内存的 CPU 视图，形状和数据类型由输入决定。 |
| `host_ptr` | int | 缓冲区的 Host 地址。 |
| `dev_ptr` | int | 缓冲区映射后可供 NPU 访问的地址。 |

`create_shm_tensor` 返回 `(host_tensor, host_ptr, dev_ptr)`。`free_shm` 返回 `None`。

## 示例

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

## 参考资料

- [昇腾 HAL 驱动的 `halHostRegister` 和 `halHostUnregister` 接口说明](https://gitcode.com/cann/driver/blob/master/src/ascend_hal/svm/README.md#%E5%86%85%E5%AD%98%E5%85%B1%E4%BA%AB)
