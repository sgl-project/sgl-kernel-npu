# top_k_renorm_prob

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.sample import top_k_renorm_prob


def top_k_renorm_prob(probs, top_ks, *, impl="auto"): ...
```

## 功能描述

对 NPU FP32 概率行执行 top-k 筛选，并除以 max(sum,1e-20) 重新归一化。top-k 阈值限制到 [1,vocab]，top-p 限制到 [0,1]。核采样比较包含当前项的累积概率减去当前项与阈值的大小。输入保持不变。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `probs` | Tensor | `Required` | FP32 NPU 概率 [batch,vocab]，1<=vocab<=2**23。 |
| `top_ks` | int or Tensor | `Required` | 保留数量，整数标量或 [batch] 张量。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | FP32 归一化概率 [batch,vocab]，与 probs 同设备。 |

## triton

使用 Triton-Ascend kernel。

## torch_native

使用输入设备上的 PyTorch 张量运算。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sample import top_k_renorm_prob

p = torch.softmax(torch.randn(2, 97, device="npu"), -1)
y = top_k_renorm_prob(p, 16, impl="triton")
```
