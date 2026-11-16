# grouped_gemma_rmsnorm

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    ...
```

## 说明

分组 Gemma RMSNorm：连续 BF16 x[R,10240]、weight[10240]，group_size=2560，eps 为有限正数。每组独立计算均方，乘以 FP32 的 (1+weight)，输出 BF16。Triton 实现复用已有归一化 kernel。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

每组计算 `x * rsqrt(mean(x*x) + eps) * (1 + weight)`，统计量和缩放为 FP32，最后舍入到 BF16。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `x` | Tensor | [R,10240], BF16 | required | 连续 NPU 输入。 |
| `weight` | Tensor | [10240], BF16 | required | x 同设备连续权重；在 FP32 中加 1。 |
| `group_size` | int | — | required | 2560；四个独立分组。 |
| `eps` | float | — | required | 有限正数。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | x.shape | BF16 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU FP64 参考实现，结果返回输入设备。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=x.dtype)
y = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="auto")
```
