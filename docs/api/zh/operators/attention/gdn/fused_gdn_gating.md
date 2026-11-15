# fused_gdn_gating

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数原型

```python
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"):
    ...
```

`auto` 默认选择 Triton；`torch_native` 是独立的 PyTorch 对照实现。

## 功能说明

本算子用于 Gated DeltaNet（GDN）的 gate 预处理。对每行 `r`、每个 head `h`，先将时间步偏置
加到 `a`，再计算带阈值的 softplus 和对数衰减缩放，并从 `b` 计算 sigmoid gate：

$$
x_{r,h}=a_{r,h}+dt\_bias_h
$$

$$
\mathrm{softplus}_{\beta,t}(x)=
\begin{cases}
\frac{1}{\beta}\log(1+e^{\beta x}), & \beta x\le t\\
x, & \beta x>t
\end{cases}
$$

$$
g_{r,h}=-\exp(A\_log_h)\mathrm{softplus}_{\beta,t}(x_{r,h}),
\qquad beta\_output_{r,h}=\mathrm{sigmoid}(b_{r,h})
$$

其中 t、β 分别是参数 `threshold`、`beta`。

阈值语义与 `torch.nn.functional.softplus` 一致：`beta * x <= threshold` 时使用对数分支，
否则直接返回 `x`。`g` 的计算使用 FP32 中间值，输出为 FP32；`beta_output` 的 dtype 与 `b` 相同。

该算子用于 SGLang NPU `AscendGDNAttnBackend.forward_extend` 的非 `target_verify` 路径，输出会
进入 GDN chunk 流程。decode 路径则在 `fused_sigmoid_gating_delta_rule_update` 中计算 gating。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `A_log` | `torch.Tensor` | `[H]` | 浮点；生产中为 FP32 | 必填 | 每个 head 的对数衰减参数；Triton 要求连续。 |
| `a` | `torch.Tensor` | `[R, H]` | FP16/BF16/FP32 | 必填 | softplus 输入；Triton 要求连续。 |
| `b` | `torch.Tensor` | `[R, H]` | FP16/BF16/FP32 | 必填 | sigmoid 输入；shape 须与 `a` 相同，Triton 要求连续。 |
| `dt_bias` | `torch.Tensor` | `[H]` | 浮点；生产中为 FP32 | 必填 | 每个 head 的时间步偏置；Triton 要求连续。 |
| `beta` | `float` | — | — | `1.0` | 大于零的 softplus 缩放系数。 |
| `threshold` | `float` | — | — | `20.0` | softplus 阈值，与 `beta * x` 比较。 |
| `impl` | `str` | — | — | `"auto"` | 仅限关键字：`"auto"`、`"triton"` 或 `"torch_native"`。 |

输入预期位于同一 NPU。`a`、`b` 必须是 shape 相同的二维张量；`A_log`、`dt_bias` 各包含 `H`
个元素。`R` 必须大于零，且 `beta` 必须大于零。Triton 会检查 `a`、`b` 的 shape 是否匹配和
是否连续，并检查 `A_log`、`dt_bias` 的元素数。它不会检查这两个向量的 stride，但为保证直接
指针寻址正确，Triton 要求它们连续。`torch_native` 遵循 PyTorch 的广播和报错行为，因此无论
选择哪个实现，调用方都应满足相同的 shape 约定。

| 返回值 | 类型 | Shape | Dtype | 描述 |
| --- | --- | --- | --- | --- |
| `g` | `torch.Tensor` | `[R, H]` | FP32 | 对数衰减 gate，位于 `a` 的设备上。 |
| `beta_output` | `torch.Tensor` | `[R, H]` | 与 `b` 相同 | sigmoid gate，位于 `b` 的设备上。 |

输入不会被修改；两个返回值均为新分配的张量。

## triton 后端

在 Triton kernel 中计算带阈值的 softplus 和 sigmoid。kernel 通过不带 stride 的指针直接寻址，
因此四个输入张量都必须连续。对数衰减 gate 以 FP32 存储；sigmoid 结果转换为 `b.dtype`。

## torch_native 后端

使用独立的 PyTorch 逐元素运算计算公式。对数衰减部分使用 FP32 中间值；`torch.sigmoid(b)`
返回与 `b` 相同 dtype 的 sigmoid gate。

## 调用示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

rows, heads = 4, 32
A_log = torch.randn(heads, dtype=torch.float32, device="npu")
dt_bias = torch.randn(heads, dtype=torch.float32, device="npu")
a = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")
b = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")

g, beta_output = fused_gdn_gating(A_log, a, b, dt_bias)
g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias, impl="torch_native")
```
