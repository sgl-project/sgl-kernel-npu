# qsa_topk

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.indexer import qsa_topk

def qsa_topk(score, lengths, topk, row_starts=None, *, impl="auto"):
    ...
```

## 说明

选择序列内相对块索引。输入 FP32 score[R,M]，连续 int32 lengths[R]，可选 row_starts[R]。topk 只支持 512/2048。输出连续 int32[R,topk]。短行按 0..length-1 排列并以 -1 补齐；长行选择最大的 topk 个值，并列值允许不同合法索引。有效区间内分数必须有限。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

对第 r 行，只考虑 `score[r,start:start+length]`。length 不超过 topk 时，按输入顺序返回索引 `0..length-1`；否则返回该区间中分数最高的 topk 个索引。索引相对于 start，未使用的输出位置填 -1。同分时不要求固定的索引顺序。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `score` | Tensor | [R,M], FP32 | required | NPU；列连续、允许行间距；M <= 2**24；有效分数有限。 |
| `lengths` | Tensor | [R], int32 | required | score 同设备连续张量；有效区间不得越界。 |
| `topk` | int | — | required | 512 或 2048。 |
| `row_starts` | Tensor or None | [R], int32 | None | score 同设备连续张量；None 表示全零。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,topk] | int32 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU 参考实现。分数相同时，可以选出不同但同样有效的索引。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.indexer import qsa_topk

score = torch.randn(1, 1024, device="npu", dtype=torch.float32)
lengths = torch.full((1,), 1024, device="npu", dtype=torch.int32)
y = qsa_topk(score, lengths, 512, impl="auto")
```
