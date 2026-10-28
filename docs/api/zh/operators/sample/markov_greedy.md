# markov_greedy

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `triton` | 支持 | 待验证 |
| `torch_native` | 支持 | 待验证 |
| `tilelang` | 未实现 | 未实现 |
| `ascendc` | 未实现 | 未实现 |

默认 `impl="auto"` 与 `triton` 注册到同一入口。注册成功不代表所有芯片均已完成验证。

## 函数原型

```python
from sgl_kernel_npu.operators.sample import markov_greedy

def markov_greedy(*, base_logits, prev_embeds, w2_weight, impl="auto"):
    ...
```

## 功能说明

将 `prev_embeds` 经过 `w2_weight` 线性投影后加到 `base_logits`，返回每行最大值的词表索引。计算与 `(base_logits + torch.nn.functional.linear(prev_embeds, w2_weight)).argmax(-1)` 对齐，包括投影和加法的数据类型舍入。最大值相同时取最小索引，NaN 优先于有限值并取第一个 NaN。

## 参数与返回值

张量须位于同一 NPU。线性投影的输入和权重须使用兼容的浮点类型；实现覆盖 FP32、FP16 和 BF16。

| 参数 | 类型 | Shape / dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- |
| `base_logits` | `torch.Tensor` | `[B, V]`，浮点 | 必填 | 基础 logits，`V > 0`，列步长为 1。 |
| `prev_embeds` | `torch.Tensor` | `[B, H]`，浮点 | 必填 | 上一 token 的嵌入。 |
| `w2_weight` | `torch.Tensor` | `[V, H]`，浮点 | 必填 | 线性投影权重，列步长为 1。 |
| `impl` | `str` | — | `"auto"` | 可选 `"auto"`、`"triton"` 或 `"torch_native"`。 |

| 返回值 | 类型 | Shape / dtype | 描述 |
| --- | --- | --- | --- |
| `indices` | `torch.Tensor` | `[B]`，`torch.int64` | 在输入设备上新建的词表索引；`B` 可为 0。 |

## triton 后端

使用 Torch 线性投影和 Triton 加法归约。当前 batch 为 16 至 31 时，投影后使用 Torch 加法和 argmax。

## torch_native 后端

使用 Torch 线性投影、加法和 argmax。

## 调用示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.sample import markov_greedy

base = torch.randn(2, 1024, device="npu", dtype=torch.float16)
embeds = torch.randn(2, 128, device="npu", dtype=torch.float16)
weight = torch.randn(1024, 128, device="npu", dtype=torch.float16)
tokens = markov_greedy(base_logits=base, prev_embeds=embeds, w2_weight=weight)
```
