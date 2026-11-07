# transfer_kv_per_layer_direct_pf_lf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_per_layer_direct_pf_lf

def transfer_kv_per_layer_direct_pf_lf(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

将锁页主机内存中选定槽位的指定层恢复到 NPU，方向固定为 H2D。主机源布局为 `[槽位, 层, 1, *状态形状]`，目标是设备上某一层的 `[槽位, *状态形状]` 视图；按 `src_indices[i] → dst_indices[i]` 搬运 `layer_id` 对应的数据，支持不连续索引及重复读取同一主机槽位。仅写入目标视图中选中的槽位，可用于 KV 或其他具有相同布局的状态恢复。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | [H,L,1,...] | 源与目标一致 | 必填 | 连续锁页主机源状态，按槽位、层、页内位置排列。 |
| `dst` | Tensor | [D,...] | 源与目标一致 | 必填 | 目标设备上的单层视图。 |
| `src_indices` | Tensor | [N] | int64 | 必填 | 按传输顺序给出的主机槽位。 |
| `dst_indices` | Tensor | [N] | int64 | 必填 | 对应的设备槽位。 |
| `layer_id` | int | — | — | 必填 | 要读取的主机层号，从 0 开始。 |
| `flags` | TransferFlag | — | — | TransferFlag.FAST2D | 二维拷贝标志 FAST2D。 |
| `impl` | str | — | — | 'auto' | 选择实现，默认为 auto。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| `None` | NoneType | 原地写入目标。在读取结果或复用源缓冲区前同步提交拷贝的流。 |

## aclrt 后端

在当前 NPU 流提交异步二维拷贝。主机源状态须连续且锁页；设备目标槽位内部须物理稠密，槽位 pitch 须等于槽位数据大小。源、目标的 dtype 和状态形状须一致；槽位索引须非负且小于各自缓冲区的槽位数，`layer_id` 须在主机层范围内。提交前校验输入，ACL 提交错误直接上抛。

## torch_native 后端

复用 SGLang `MambaPoolHost` 的单层 Torch fallback，通过 `index_select/index_copy_` 完成主机槽位到设备槽位的映射。先将稠密状态按物理存储顺序展平，保持转置状态与原生字节拷贝一致；重复目标索引按输入顺序覆盖。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.data_move import transfer_kv_per_layer_direct_pf_lf

host = torch.ones(4, 2, 1, 8, pin_memory=True)
layer = torch.zeros(4, 8, device="npu")
slots = torch.tensor([1, 3])
transfer_kv_per_layer_direct_pf_lf(host, layer, slots, slots, layer_id=1)
torch.npu.synchronize()
print(layer[1])
```
