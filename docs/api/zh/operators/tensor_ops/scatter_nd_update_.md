# scatter_nd_update_

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |

## 函数原型

`auto` 在对应代际选择已注册的 `torch_npu`。

```python
from sgl_kernel_npu.operators.tensor_ops import scatter_nd_update_


def scatter_nd_update_(input, indices, updates, *, impl="auto"): ...
```

## 功能说明

按 $\mathrm{input}[\mathrm{indices}[i],\ldots]=\mathrm{updates}[i,\ldots]$ 原地更新。每个坐标索引输入的前若干维，因此一次更新可以替换一个切片。

## 参数与返回值

除表中注明的主机侧元数据外，Tensor 参数应位于同一 NPU。示例使用一种 SDK 支持的配置。

| 参数 | 类型 | 形状 / dtype | 默认值 | 说明与约束 |
| --- | --- | --- | --- | --- |
| `input` | Tensor | ND, dtype supported by TorchNPU | 必传 | NPU 上待修改的目标 Tensor。 |
| `indices` | Tensor | [..., R], INT32/INT64 | 必传 | 末轴包含 R 个坐标。要获得确定性结果，应使用互不重复的有效坐标。 |
| `updates` | Tensor | indices.shape[:-1] + input.shape[R:], 输入 dtype | 必传 | 同卡上写入相应坐标的值。 |
| `impl` | str | - | `"auto"` | `auto`, `torch_npu`. |

| 返回值 | 类型 | 形状 / dtype | 说明 |
| --- | --- | --- | --- |
| `out` | Tensor | 与输入一致 | 与更新后的目标 Tensor 共享存储。 |

## torch_npu 后端

调用 `torch_npu.npu_scatter_nd_update_`。形状、dtype 和布局组合以已安装的 SDK 为准。 [TorchNPU 参考](https://github.com/Ascend/op-plugin/blob/255eb455f9e569df3a617eb39685d91400400299/docs/zh/custom_APIs/torch_npu/torch_npu-npu_scatter_nd_update_.md)

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.tensor_ops import scatter_nd_update_

dst = torch.zeros((7, 32), dtype=torch.float16, device="npu")
slots = torch.tensor([[6], [0], [3]], dtype=torch.int32, device=dst.device)
values = torch.ones((3, 32), dtype=dst.dtype, device=dst.device)
scatter_nd_update_(dst, slots, values, impl="torch_npu")
```
