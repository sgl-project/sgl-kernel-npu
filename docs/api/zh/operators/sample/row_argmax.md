# row_argmax

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

默认 `impl="auto"` 与 `triton` 注册到同一入口。注册成功不代表所有芯片均已完成验证。

## 函数原型

```python
from sgl_kernel_npu.operators.sample import row_argmax

def row_argmax(x, *, impl="auto"):
    ...
```

## 功能说明

返回每行最大值的索引，与 `torch.argmax(x, dim=-1)` 一致。最大值相同时取最小列索引；行中存在 NaN 时，返回第一个 NaN 的索引。

## 参数与返回值

| 参数 | 类型 | Shape / dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, V]`，`torch.float32` | 必填 | NPU 输入，`V > 0`，列步长为 1；`B` 可为 0，行可有额外步长。 |
| `impl` | `str` | — | `"auto"` | 可选 `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape / dtype | 描述 |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`，`torch.int64` | 在输入设备上新建的索引张量。 |

## triton 后端

使用分块 Triton kernel 归约每一行，要求输入为二维 FP32 张量且列步长为 1。

## torch_native 后端

调用 `torch.argmax(x, dim=-1)`。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import row_argmax

x = torch.tensor([[1.0, 3.0, 3.0], [4.0, 2.0, 1.0]], device="npu")
indices = row_argmax(x)
# indices 为 [1, 0]
```
