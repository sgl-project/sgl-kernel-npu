# gemma_rmsnorm

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `torch_npu` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `torch_npu`。

```python
from sgl_kernel_npu.operators.norm import gemma_rmsnorm


def gemma_rmsnorm(input, weight, eps=1e-06, *, impl="auto"): ...
```

## 功能描述

执行 RMS 归一化后乘以 Gemma 缩放 (1+weight)，返回输出与 RMS 倒数。ARCH22 使用原生 Gemma 算子；ARCH35 在 weight 的存储类型中计算 1+weight，再执行 RMSNorm，保留这一舍入边界。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | Tensor | `Required` | 浮点 NPU 输入 [...,H]。 |
| `weight` | Tensor | `Required` | 卷积为 [channels,width]；Gemma 为 [H]，与输入同类型。 |
| `eps` | float | `1e-06` | RMSNorm 稳定常数。 |
| `impl` | str | `'auto'` | auto 选择 torch_npu；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | 输出与 input 同形状、同类型；FP32 RMS 倒数形状为 [...,1]。 |

## torch_npu

使用对应芯片架构的原生 NPU RMSNorm 接口。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import gemma_rmsnorm

x = torch.randn(4, 2304, device="npu", dtype=torch.bfloat16)
w = torch.zeros(2304, device="npu", dtype=x.dtype)
y, rstd = gemma_rmsnorm(x, w, impl="torch_npu")
```
