# fused_kda_gate

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention.kda import fused_kda_gate


def fused_kda_gate(
    gate,
    A_log,
    head_dim,
    *,
    gate_bias=None,
    beta=1.0,
    threshold=20.0,
    lower_bound=None,
    impl="auto"
): ...
```

## 功能描述

以 FP32 计算 KDA 通道门控，默认使用 -exp(A_log)*softplus(gate+bias)；提供 lower_bound 时改为 lower_bound*sigmoid(exp(A_log)*(gate+bias))。输出把最后一个打包维展开为 [heads,head_dim]。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `gate` | Tensor | `Required` | 浮点打包门控 [...,heads*head_dim]。 |
| `A_log` | Tensor | `Required` | 浮点衰减权重 [heads] 或等价单维扩展形状。 |
| `head_dim` | int | `Required` | 每个门控头的通道数。 |
| `gate_bias` | Tensor or None | `None` | 可选浮点门控偏置 [heads*head_dim]。 |
| `beta` | float | `1.0` | 对应算子的 softplus 乘数或 SiTU 饱和参数。 |
| `threshold` | float | `20.0` | softplus 切换线性分支的阈值。 |
| `lower_bound` | float or None | `None` | 提供时选择有界 sigmoid 门控，None 选择 softplus。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | FP32 门控激活 [...,heads,head_dim]。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.kda import fused_kda_gate

gate = torch.randn(4, 256, device="npu", dtype=torch.bfloat16)
a = torch.zeros(2, device="npu")
y = fused_kda_gate(gate, a, 128, impl="triton")
```
