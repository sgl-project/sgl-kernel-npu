# add_example

## 芯片支持

| 实现 | arch22（2201，A2/A3） | arch35（3510，A5） |
| --- | --- | --- |
| `aicpu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

`auto` 是 `aicpu` 的别名，不算独立实现。请安装与设备架构匹配、且通过 `--ops`
选择了 `add_example` 的 wheel。构建方式见[构建安装说明](../../../../../python/sgl_kernel_npu/README.md#building-and-installing-aicpu-operators)。

## 函数原型

```python
from sgl_kernel_npu.operators.examples import add_example

def add_example(x1, x2, *, impl="auto"):
    ...
```

不传 `impl` 或传入 `"auto"` 均选择 `aicpu`。AICPU 实现不可用时不会自动回退；
需要独立参考实现时，显式传入 `impl="torch_native"`。

## 功能说明

将两个输入逐元素相加，返回新张量：

$$
y_i = (x_1)_i + (x_2)_i.
$$

两个输入的 shape 和 dtype 必须相同，不支持广播。输入保持不变；输出使用独立
存储，与输入的 shape、dtype 和 NPU 设备相同，存储格式为连续 ND。

本算子用于演示 AICPU 接入，不作为性能基准。示例采用不发生整数溢出的值，
尚未验证溢出行为。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x1` | `torch.Tensor` | `S` | `torch.float32` 或 `torch.int32` | 必填 | 至少包含一个元素的 NPU 张量，要求连续 ND 存储。 |
| `x2` | `torch.Tensor` | 与 `x1` 相同的 `S` | 与 `x1` 相同 | 必填 | 与 `x1` 位于同一 NPU，要求连续 ND 存储，不支持广播。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"aicpu"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `y` | `torch.Tensor` | 与输入相同的 `S` | 与输入相同 | 同一 NPU 上新分配的连续 ND 张量，不与任一输入共享存储。 |

公开接口在分派前检查 shape、dtype、设备、非空输入和存储布局。不支持的输入
会报错，不会被静默转换。

## aicpu 后端

通过 `torch.ops.npu.sgl_add_example` 调用包内的 AICPU 实现，支持符合上述约束且 dtype
相同的 FP32 或 INT32 输入。wheel 必须包含该 AICPU 实现；缺少绑定时会报错。

## torch_native 后端

使用 `torch.add` 提供独立参考。公开接口执行相同的输入检查，包括同一 NPU 的要求。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.examples import add_example

x1 = torch.tensor([2, 5], dtype=torch.float32, device="npu")
x2 = torch.tensor([3, 3], dtype=torch.float32, device=x1.device)
result = add_example(x1, x2)  # auto -> aicpu，结果为 [5, 8]
reference = add_example(x1, x2, impl="torch_native")
torch.testing.assert_close(result, reference, atol=0, rtol=0)
```
