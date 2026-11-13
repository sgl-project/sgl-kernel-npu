# swiglu_oai

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.activation import swiglu_oai


def swiglu_oai(hidden_states, dim, gemm1_alpha, gemm1_clamp_limit, *, impl="auto"): ...
```

## 功能描述

计算 GPT-OSS 使用的交错布局 SwiGLU：偶数通道为 gate，奇数通道为 up。gate 只截断上界，up 截断到 [-limit,limit]，输出 gate*sigmoid(alpha*gate)*(up+1)，形状为 [rows,dim/2]。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `hidden_states` | Tensor | `Required` | 连续浮点输入 [rows,2*D]。 |
| `dim` | int | `Required` | 打包 gate/up 宽度，与输入最后一维相等。 |
| `gemm1_alpha` | float | `Required` | sigmoid 输入乘数。 |
| `gemm1_clamp_limit` | float | `Required` | gate 上界和 up 双边界。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | [rows,dim/2] 输出，与输入同类型、同设备。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.activation import swiglu_oai

x = torch.randn(4, 5760, device="npu", dtype=torch.bfloat16)
y = swiglu_oai(x, 5760, 1.702, 7.0, impl="triton")
```
