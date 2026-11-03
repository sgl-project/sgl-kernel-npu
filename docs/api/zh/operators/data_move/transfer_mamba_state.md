# transfer_mamba_state

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `triton` | 未实现 | 未实现 |
| `tilelang` | 未实现 | 未实现 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, transfer_mamba_state

def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

按槽位在设备逐层布局与主机逐页布局之间双向搬运 Mamba 状态。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_buf` | Tensor | [L,D,...] | 源与目标一致 | 必填 | 按层排列的设备状态。 |
| `host_buf` | Tensor | [H,L,1,...] | 源与目标一致 | 必填 | 按槽位排列的锁页主机状态。 |
| `device_indices` | Tensor | [N] | int64 | 必填 | 设备槽位索引。KV 接口传 token 索引。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应的主机索引。 |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D 或 D2H。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## aclrt 后端

在当前 NPU 流提交异步二维拷贝。

## torch_native 后端

使用 PyTorch 在输入所在设备操作。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_mamba_state

dev = torch.zeros(2, 4, 8, device="npu")
host = torch.ones(4, 2, 1, 8, pin_memory=True)
slots = torch.tensor([1, 3])
transfer_mamba_state(dev, host, slots, slots)
torch.npu.synchronize()
print(dev[:, 1])
```
