# top_k_top_p_renorm_prob

## 硬件支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 未验证 |
| `torch_native` | 支持 | 未验证 |

## 函数签名

`auto` 选择 `triton`。

```python
from sgl_kernel_npu.operators.sample import top_k_top_p_renorm_prob


def top_k_top_p_renorm_prob(probs, top_ks, top_ps, max_keep=None, *, impl="auto"): ...
```

## 功能描述

先在原始分布上执行 top-p，再执行 top-k，并对联合支撑集做一次归一化，分母下限为 1e-20。两个过滤都保留降序行的前缀，因此存活列是两者前缀掩码的交集；top-p 使用包含当前项的累积概率减去当前项与阈值比较。

`max_keep` 是每行 top-k 的可选上界；当它覆盖所有行时，只选择前 `max_keep` 列是精确的。如果某行 top-k 大于 `max_keep`，该行会被截断到 `max_keep` 列，因此应传入批次的最大 top-k 或安全上界。传 `None` 时过滤完整词表，对任意输入都精确，但属于慢路径；热路径应传上界。

选择边界上的等概率项可能与全排序落到不同的 token 下标，但保留的数值和总质量不变。

## 参数与返回值

所有张量须位于同一 NPU 设备。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `probs` | Tensor | `Required` | FP32 NPU 概率 [batch,vocab]，1<=vocab<=2**23。 |
| `top_ks` | int or Tensor | `Required` | 保留数量，整数标量或 [batch] 张量。 |
| `top_ps` | float or Tensor | `Required` | 核采样阈值，浮点标量或 [batch] 张量。 |
| `max_keep` | int | `None` | 每行 top-k 的可选正整数上界；`None` 使用完整词表路径。 |
| `impl` | str | `'auto'` | auto 选择 triton；可显式选择后端。 |

| 返回值 | 类型 | 说明 |
| --- | --- | --- |
| result | Tensor | FP32 归一化概率 [batch,vocab]，与 probs 同设备。 |

## triton 后端

- 用 PyTorch top-k 选出前 `max_keep` 列，并计算其累积和。
- 由 Triton kernel 应用 top-p/top-k 联合掩码并对选中的前缀归一化，分母下限为 1e-20。
- 通过只覆盖选中列的窄 scatter 恢复词表顺序，避免全词表写回。
- 宽度超过 2048 时退回 PyTorch 过滤路径。

## torch_native 后端

- 独立的两遍参考实现：先在完整分布上做 top-p，再做 top-k；忽略 `max_keep`。
- 使用 PyTorch sort、mask、sum 和 scatter 完成筛选、归一化与词表顺序恢复。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sample import top_k_top_p_renorm_prob

p = torch.softmax(torch.randn(2, 97, device="npu"), -1)
y = top_k_top_p_renorm_prob(p, 16, 0.95, max_keep=16, impl="triton")
```