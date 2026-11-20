# transfer_state_per_layer_direct_pf_lf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `ascendc_aiv` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_state_per_layer_direct_pf_lf

def transfer_state_per_layer_direct_pf_lf(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `ascendc_aiv`。

## 功能说明

按一一对应的主机槽位和设备槽位索引，把主机内存中指定一层的 Mamba/SSM 状态恢复到 NPU（仅 H2D）。主机状态按页优先排列 `[槽位, 层, 1, *状态形状]`，目标设备层视图为 `[槽位, *状态形状]`；`layer_id` 选择要恢复的主机层，每对索引只搬运该层。未选中的槽位保持不变。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | [H,L,1,...] | 源与目标一致 | 必填 | 按槽位、层排列的主机状态。 |
| `dst` | Tensor | [D,...] | 源与目标一致 | 必填 | 目标设备层视图 `[槽位, *状态形状]`，通常为设备池某一层的切片。 |
| `src_indices` | Tensor | [N] | int64 | 必填 | 按拷贝顺序排列的主机槽位索引。 |
| `dst_indices` | Tensor | [N] | int64 | 必填 | 与主机索引一一对应的设备槽位索引。 |
| `layer_id` | int | — | — | 必填 | 从零开始的主机源层号，须小于 `src` 的层数。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## ascendc_aiv 后端

在当前 NPU 流提交异步 AIV 拷贝。内部归一化为单组件的 state_exchange 提交：主机侧取 `src` 在 `layer_id` 处的层窄化视图，设备侧 `dst` 升维为 `[1, 槽位, ...]`，偏移由基地址和步长携带。主机内存必须已通过 `register_acl_host_memory`（aclrt 后端）或 MemFabric 池注册，注册表是唯一的 DVA 来源，未注册内存直接报错。

## torch_native 后端

PyTorch 参考逻辑：复用 SGLang `MambaPoolHost` 的逐层恢复路径，主机缓冲区须锁页。不调用 `ascendc_aiv` 实现。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_state_per_layer_direct_pf_lf
from sgl_kernel_npu.operators.mem_cache.memfabric.host_memory import register_acl_host_memory

dev = torch.zeros(8, 128, device="npu")
host = torch.randn(8, 4, 1, 128)
register_acl_host_memory(host)
h, d = torch.arange(8, dtype=torch.int64, device="npu"), torch.arange(8, dtype=torch.int64, device="npu")
transfer_state_per_layer_direct_pf_lf(host, dev, h, d, layer_id=2)
torch.npu.synchronize()
print(dev)
```
