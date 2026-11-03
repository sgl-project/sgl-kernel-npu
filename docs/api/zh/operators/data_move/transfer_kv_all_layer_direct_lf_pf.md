# transfer_kv_all_layer_direct_lf_pf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `triton` | 未实现 | 未实现 |
| `tilelang` | 未实现 | 未实现 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_all_layer_direct_lf_pf

def transfer_kv_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

将所选槽位的所有层和状态分量备份到主机。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Sequence[Tensor] | [L,D,...] | 源与目标一致 | 必填 | 逐层排列的设备状态分量列表。每个槽位的物理存储须稠密。 |
| `host_states` | Sequence[Tensor] | [H,L,1,...] | 源与目标一致 | 必填 | 与设备分量一一对应的连续锁页主机缓冲区。 |
| `device_indices` | Tensor | [N] | int64 | 必填 | 设备槽位索引。KV 接口传 token 索引。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应的主机索引。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
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
from sgl_kernel_npu.operators.data_move import transfer_kv_all_layer_direct_lf_pf

dev = torch.ones(2, 4, 8, device="npu")
host = torch.zeros(4, 2, 1, 8, pin_memory=True)
slots = torch.tensor([1, 3])
transfer_kv_all_layer_direct_lf_pf([dev], [host], slots, slots)
torch.npu.synchronize()
print(host[1, :, 0])
```
