# fused_gdn_gating_without_sigmoid

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating_without_sigmoid


def fused_gdn_gating_without_sigmoid(
    A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"
): ...
```

## 功能描述

以 FP32 计算 `g = -exp(A_log) * softplus(a + dt_bias, beta, threshold)`。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | 浮点衰减权重 [heads] 或等价单维扩展形状。 |
| `a` | Tensor | `Required` | 形状为 [tokens,heads] 的浮点门控输入。 |
| `b` | Tensor | `Required` | 与 `a` 搭配传入的门控张量。 |
| `dt_bias` | Tensor | `Required` | 形状为 [heads] 的浮点门控偏置。 |
| `beta` | float | `1.0` | 非零的 softplus 系数。 |
| `threshold` | float | `20.0` | softplus 切换线性分支的阈值。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | FP32 的 `g`，形状为 [1,tokens,heads]，以及传入的 `b` 原张量。 |

## triton 后端

- 对连续的门控输入 `a` 融合偏置加法、带阈值的 softplus 和按头衰减缩放。

## torch_native 后端

- 用 `exp`、`log1p` 和 `where` 计算带阈值的 softplus，再应用衰减缩放。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating_without_sigmoid

a = torch.randn(16, 8, device="npu", dtype=torch.bfloat16)
b = torch.randn_like(a)
w = torch.zeros(8, device="npu")
g, raw_beta = fused_gdn_gating_without_sigmoid(w, a, b, w, impl="triton")
```
