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
| `weight` | Tensor | `Required` | 形状为 [H] 的 RMSNorm 权重，H 等于输入最后一维，数据类型与输入相同。 |
| `eps` | float | `1e-06` | RMSNorm 稳定常数。 |
| `impl` | str | `'auto'` | auto 选择 torch_npu；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor] | 输出与 input 同形状、同类型；FP32 RMS 倒数形状为 [...,1]。 |

## torch_npu 后端

- ARCH22 直接调用 torch_npu.npu_gemma_rms_norm；ARCH35 先在 weight dtype 中计算 1+weight，再调用 torch_npu.npu_rms_norm。
- 返回归一化输出和 RMS 倒数，沿用对应 torch_npu 接口的执行与类型要求。

## torch_native 后端

- 使用 FP32 平方均值和 rsqrt 计算 RMS 倒数，再乘以输入及 Gemma 缩放，输出转换回 input dtype。
- ARCH22 在 FP32 中计算 1+weight；ARCH35 保留 weight dtype 加1的舍入。RMS 倒数保留 FP32，不调用融合 RMSNorm 接口。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import gemma_rmsnorm

x = torch.randn(4, 2304, device="npu", dtype=torch.bfloat16)
w = torch.zeros(2304, device="npu", dtype=x.dtype)
y, rstd = gemma_rmsnorm(x, w, impl="torch_npu")
```
