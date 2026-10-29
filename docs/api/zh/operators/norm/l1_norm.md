# l1_norm

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.norm import l1_norm

def l1_norm(input, *, impl="auto"):
    ...
```

`auto` 和 `triton` 当前指向同一入口，不会在执行失败后自动切换实现。

## 功能说明

将二维输入的每一行除以该行元素的带符号和：

$$
Y_{i,j}=\frac{X_{i,j}}{\sum_{k=0}^{H-1}X_{i,k}}
$$

分母不取绝对值，也不添加 epsilon。该语义可用于对选中的专家权重重新归一化。行和为零时可能返回 NaN 或 Inf。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, H]` | `torch.float32`、`torch.float16`、`torch.bfloat16` | 必填 | `H > 0`；Triton 要求连续的 NPU tensor，且 `H` 为 2 的幂。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字参数；可选 `"auto"`、`"triton"`、`"torch_native"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | `torch.float32` | 新建输出，与输入位于同一设备，不修改输入。 |

公共入口不增加输入校验或连续化。Triton 按 `row * H + column` 寻址，不读取 tensor 的行 stride；输入切片应由调用方保证满足连续布局要求。

## triton 后端

- 使用 `l1_norm_kernel`，按行求和并归一化，写入 FP32 输出。
- 使用 `tl.arange(0, H)`，需要 2 的幂宽度。

## torch_native 后端

- 使用 CPU 上的 FP32 NumPy 数组计算：`values = input.to(torch.float32).cpu().numpy()`，随后执行 `values / values.sum(axis=-1, keepdims=True)`。
- 将 NumPy 结果转换为 FP32 Tensor，并放回输入设备，保持公共返回值约定。
- 需要 NumPy；NPU 输入会发生设备到 CPU 及结果回传的数据拷贝，用于精度参考，不适合放在 NPU graph 捕获中执行。

## 调用示例

示例需要已安装包含本接口的 wheel 及 NPU 运行依赖。

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import l1_norm

torch.npu.set_device(0)
weights = 0.5 + torch.rand(16, 8, device="npu", dtype=torch.bfloat16)

y_default = l1_norm(weights)
y_triton = l1_norm(weights, impl="triton")
y_native = l1_norm(weights, impl="torch_native")

print(y_triton.shape, y_triton.dtype)  # torch.Size([16, 8]), torch.float32
```
