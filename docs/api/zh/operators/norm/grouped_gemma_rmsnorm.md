# grouped_gemma_rmsnorm

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Yes | No |
| `torch_native` | Yes | No |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(x, weight, group_size, eps, *, impl="auto"):
    ...
```

## 说明

分组 Gemma RMSNorm：连续 BF16 x[R,10240]、weight[10240]，group_size=2560，eps 为有限正数。每组独立计算均方，乘以 FP32 的 (1+weight)，输出 BF16。复用公共 layer_norm_fwd 的 RMSNorm 分支。

所有张量位于同一 NPU，输入不修改；零行返回空输出。`auto` 与 `triton` 指向同一入口，保留实现中的原生 PyTorch 子步骤，不捕获异常切换实现。`torch_native` 为独立 reference：短卷积使用原生 NPU 卷积，其余使用 CPU 计算（浮点 oracle 使用 FP64）后返回输入设备。reference 会同步，仅用于图外精度验证。

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

默认推理实现。启动前检查不支持的元数据；输入内容由调用方保证符合上述约束。支持 graph capture 和原地更新输入、权重后的 replay。实现中已有的原生子步骤保留在此分支。

## torch_native

针对上述输入契约的独立精度 reference。允许同步主机传输，应在图外运行；不会在默认实现失败后自动切换到此分支。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=x.dtype)
y = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="auto")
```

独立 CPU FP64 回归对 BF16 输出使用 atol=0.005、rtol=0.01。包括 R=4096 和 R=8193 在内的所有形状使用相同容差；graph 与 eager 一致不能替代精度验证。
