# fused_linear_online_max_sum

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import (
    fused_linear_online_max_sum_batch_invariant,
)

def fused_linear_online_max_sum_batch_invariant(
    input,
    weight,
    target,
    vocab_start_index,
    vocab_end_index,
    return_logits=False,
    *,
    impl="auto",
):
    ...
```

该算子转发到
`torch.ops.batch_invariant_ops.npu_fused_linear_online_max_sum_batch_invariant`。
使用前需先安装外部组件，见[安装说明](installation.md)。

## 功能说明

计算词表并行线性层在本 rank 上的 logits，以及交叉熵所需的 online max/sum 统计量。
设 $X \in \mathbb{R}^{M\times K}$、本地权重 $W \in \mathbb{R}^{N\times K}$：

$$
L = XW^{\mathsf{T}}, \qquad
m_i = \max_j L_{ij}, \qquad
s_i = \sum_j e^{L_{ij}}
$$

归约顺序与 $M$ 无关，因此每个 token 的统计量在单独计算与批量计算下逐位相同。kernel
使用自身的 online 形式，这些统计量应成对消费，不要求与朴素的 `max`/`exp` 参考实现
逐位一致。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[M, K]` | `torch.float16`、`torch.bfloat16` | 必填 | NPU 上的隐状态。 |
| `weight` | `torch.Tensor` | `[N, K]` | 与 `input` 相同 | 必填 | 本地词表权重切片，保持 `[N, K]` 布局。 |
| `target` | `torch.Tensor` | `[M]` | 整型 | 必填 | 全局词表下的目标 token id。 |
| `vocab_start_index` | `int` | — | — | 必填 | 本地 logit 首列对应的全局下标。 |
| `vocab_end_index` | `int` | — | — | 必填 | 本地 logit 末列后一位的全局下标。 |
| `return_logits` | `bool` | — | — | `False` | 是否同时返回本地 logits。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `logits_max` | `torch.Tensor` | `[M]` | `torch.float32` | 每行 logits 最大值。 |
| `sum_exp_logits` | `torch.Tensor` | `[M]` | `torch.float32` | 每行指数之和。 |
| `predicted_logits` | `torch.Tensor` | `[M]` | `torch.float32` | 目标位置的 logits。 |
| `target_mask` | `torch.Tensor` | `[(M + 7) // 8]` | `torch.uint8` | 位打包的逐 token 掩码，每字节 8 个 token。 |
| `masked_target` | `torch.Tensor` | `[M]` | 与 `target` 相同 | 损失计算使用的本地目标下标。 |
| `vocab_parallel_logits` | `torch.Tensor` 或未定义 tensor | `[M, N]` | 与 `input` 相同 | `return_logits=True` 时为本地 logits，否则为未定义 tensor。 |

输入不会被修改。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import (
    fused_linear_online_max_sum_batch_invariant,
)

hidden = torch.randn(257, 7168, device="npu", dtype=torch.bfloat16)
weight = torch.randn(8192, 7168, device="npu", dtype=torch.bfloat16)
target = torch.randint(0, 32768, (257,), device="npu", dtype=torch.int64)

full = fused_linear_online_max_sum_batch_invariant(
    hidden, weight, target, 4096, 12288, True
)
single = fused_linear_online_max_sum_batch_invariant(
    hidden[:1], weight, target[:1], 4096, 12288, True
)
for full_out, single_out in zip(full, single):
    torch.testing.assert_close(full_out[:1], single_out, atol=0, rtol=0)
```
