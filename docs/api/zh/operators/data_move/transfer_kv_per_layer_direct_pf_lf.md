# transfer_kv_per_layer_direct_pf_lf

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `aclrt` | 支持 | 支持 |
| `triton` | 未实现 | 未实现 |
| `tilelang` | 未实现 | 未实现 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.data_move import TransferFlag, transfer_kv_per_layer_direct_pf_lf

def transfer_kv_per_layer_direct_pf_lf(src, dst, src_indices, dst_indices, layer_id, flags=TransferFlag.FAST2D, *, impl='auto'): ...
```

`auto` 选择 `aclrt`。

## 功能说明

将所选主机槽位的一层恢复到设备。

## 参数与返回值

| 参数 | 类型 | 形状 | 数据类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `src` | Tensor | [H,L,1,...] | 源与目标一致 | 必填 | 锁页主机源状态，按槽位、层、页内位置排列。 |
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

在当前 NPU 流提交异步二维拷贝。

## torch_native 后端

使用 PyTorch 在输入所在设备操作。

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
