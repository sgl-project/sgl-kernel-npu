# transfer_mamba_state

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferDirection, transfer_mamba_state

def transfer_mamba_state(device_buf, host_buf, device_indices, host_indices, direction=TransferDirection.H2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

按一一对应的设备槽位和主机槽位索引，在 NPU 与锁页主机内存之间搬运 Mamba/SSM 状态，支持 H2D 恢复和 D2H 备份。设备布局为 `[层, 槽位, *状态形状]`，主机布局为 `[槽位, 层, 1, *状态形状]`；每对索引搬运该槽位的所有层，支持一维或多维状态，未选中的槽位保持不变。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_buf` | Tensor | [L,D,...] | 源与目标一致 | 必填 | 按层排列的设备状态。 |
| `host_buf` | Tensor | [H,L,1,...] | 源与目标一致 | 必填 | 按槽位排列的锁页主机状态。 |
| `device_indices` | Tensor | [N] | int64 | 必填 | 非负且小于 D 的设备槽位索引，直接寻址，不按页大小换算。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应、非负且小于 H 的主机槽位索引。 |
| `direction` | TransferDirection | — | — | TransferDirection.H2D | TransferDirection.H2D 或 D2H。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## aclrt 后端

在当前 NPU 流提交异步二维拷贝。主机缓冲区须连续且锁页；设备槽位内部须物理稠密，槽位和层间 pitch 须紧凑，允许槽位内部稠密转置。源、目标的 dtype 和槽位字节数须一致。提交前检查所有槽位索引，ACL 提交错误直接上抛。

## torch_native 后端

复用 SGLang `MambaPoolHost` 的 PyTorch 参考逻辑：H2D 使用 `index_select/index_copy_` 逐层恢复，D2H 收集所选槽位并将层维移到槽位维之后。布局适配按 `aclrt` 实现的物理存储顺序解释状态，重复目标索引按输入顺序写入；不调用 `aclrt` 实现。

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
