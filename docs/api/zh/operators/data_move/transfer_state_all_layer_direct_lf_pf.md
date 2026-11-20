# transfer_state_all_layer_direct_lf_pf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_state_all_layer_direct_lf_pf

def transfer_state_all_layer_direct_lf_pf(device_states, host_states, device_indices, host_indices, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `ascendc_aiv`。

## 功能说明

按一一对应的设备槽位和主机槽位索引，把选中的 Mamba/SSM 状态从 NPU 全层备份到主机内存（仅 D2H）。设备组件按层优先排列 `[层, 槽位, *状态形状]`，主机组件按页优先排列 `[槽位, 层, 1, *状态形状]`；每对索引搬运该槽位的所有层。支持最多 16 个组件一次提交，各组件层数可以不同（例如主干 `[L, 槽位, ...]` 加草稿层 `[1, 槽位, ...]`），主机组件可以是同一注册池按层 narrow 的视图。未选中的槽位保持不变。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `device_states` | Tensor 列表 | [L,D,...] 每组件 | 源与目标一致 | 必填 | 按层排列的设备状态组件；各组件层数可不同，槽位 payload 须物理稠密且按槽位打包。 |
| `host_states` | Tensor 列表 | [H,L,1,...] 每组件 | 源与目标一致 | 必填 | 与设备组件一一对应的主机状态组件，可为同一注册池的按层 narrow 视图。 |
| `device_indices` | Tensor | [N] | int64 | 必填 | 对所有组件有效的非负设备槽位索引。 |
| `host_indices` | Tensor | [N] | int64 | 必填 | 与设备索引一一对应的非负主机槽位索引。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## ascendc_aiv 后端

在当前 NPU 流提交异步 AIV 拷贝。主机内存必须已通过 `register_acl_host_memory`（aclrt 后端）或 MemFabric 池注册，注册表是唯一的 DVA 来源，未注册内存直接报错。每个 (槽位, 层) 行经由 88KB UB 双缓冲搬运，超宽行自动切分；槽位索引常驻设备内存，内核直接按元数据寻址，不经过 CPU。组件数上限 16，由 tiling 元数据静态大小决定。

## torch_native 后端

PyTorch 参考逻辑：`index_select` 收集所选设备槽位后 `movedim` 把层维移到槽位维之后，逐组件按层窄化的主机视图回写；主机缓冲区须锁页。不调用 `ascendc_aiv` 实现。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_state_all_layer_direct_lf_pf
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import register_acl_host_memory

dev = torch.randn(4, 8, 128, device="npu")
host = torch.zeros(8, 4, 1, 128)
register_acl_host_memory(host)
ids = torch.arange(8, dtype=torch.int64, device="npu")
transfer_state_all_layer_direct_lf_pf([dev], [host], ids, ids)
torch.npu.synchronize()
print(host[0, :, 0])
```
