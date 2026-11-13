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

计算 FP32 的 g=-exp(A_log)*softplus(a+dt_bias,beta,threshold)，为 g 添加前导单维，并原样返回 b。第二个返回值就是输入 b，不执行 sigmoid。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `A_log` | Tensor | `Required` | 浮点衰减权重 [heads] 或等价单维扩展形状。 |
| `a` | Tensor | `Required` | GDN 为 [tokens,heads]，KDA 为 [tokens,Hk,K]，可带前导单维。 |
| `b` | Tensor | `Required` | 浮点 [tokens,Hv]；KDA 可为 [1,tokens,Hv]。 |
| `dt_bias` | Tensor | `Required` | GDN 为 [heads]，KDA 为 [Hk,K] 门控偏置。 |
| `beta` | float | `1.0` | 对应算子的 softplus 乘数或 SiTU 饱和参数。 |
| `threshold` | float | `20.0` | softplus 切换线性分支的阈值。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | FP32 的 g [1,tokens,heads] 及未修改的 b。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

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
