# transfer_kv_all_layer_direct_lf_pf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_all_layer_direct_lf_pf

def transfer_kv_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

将 NPU 上选定槽位的所有层备份到锁页主机内存，方向固定为 D2H。设备状态分量按 `[层, 槽位, *状态形状]` 排列，对应的主机分量按 `[槽位, 层, 1, *状态形状]` 排列；对每组设备、主机槽位索引搬运所有分量的全部层。各分量可有不同的状态形状和数据类型，例如 Mamba 的 temporal 与 conv 状态；未选中的主机槽位保持不变。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Sequence[Tensor] | [L,D,...] | 源与目标一致 | 必填 | 逐层排列的设备状态分量列表。每个槽位的物理存储须稠密。 |
| `host_states` | Sequence[Tensor] | [H,L,1,...] | 源与目标一致 | 必填 | 与设备分量一一对应的连续锁页主机缓冲区。 |
| `device_indices` | Tensor | [N] | int64 | 必填 | 直接寻址的设备槽位索引，不按页大小换算；须非负且小于每个设备分量的槽位数。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应的主机槽位索引；须非负且小于每个主机分量的槽位数。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## aclrt 后端

在当前 NPU 流提交异步二维拷贝。主机分量须连续且锁页；设备槽位内部须物理稠密，槽位 pitch 须等于槽位数据大小。设备层间允许留空，层 pitch 须不小于设备槽位数乘以槽位 pitch。每对设备、主机分量的 dtype 和状态形状须一致。提交前校验全部分量和槽位索引，ACL 提交错误直接上抛。

## torch_native 后端

复用 SGLang `MambaPoolHost` 的全层 Torch fallback：通过 `index_select` 收集设备槽位，将层维移到槽位维之后，再写入主机。逐个处理状态分量，以物理存储视图保留转置稠密状态的字节顺序；重复目标索引按输入顺序覆盖。

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
