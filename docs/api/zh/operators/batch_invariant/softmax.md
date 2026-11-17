# softmax

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import softmax_batch_invariant

def softmax_batch_invariant(input, dim, *, impl="auto"):
    ...
```

该算子转发到 `torch.ops.batch_invariant_ops.npu_softmax_batch_invariant`。使用前需先
安装外部组件，见[安装说明](installation.md)。

## 功能说明

沿单个维度做 softmax，常见场景是 logits 的词表维。归约顺序与行数无关，因此同一行的
概率分布无论是单独计算还是在大 batch 中计算，结果逐位相同。

kernel 只支持**最后一维**，传入其他 `dim` 会触发 aclnn 参数错误。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[..., V]` | `torch.float16`、`torch.bfloat16`、`torch.float32` | 必填 | NPU 上的输入。 |
| `dim` | `int` | — | — | 必填 | 必须是最后一维（`-1` 或 `input.dim() - 1`）。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 与 input 相同 | 与 input 相同 | 新分配张量；不修改输入。 |

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import softmax_batch_invariant

logits = torch.randn(257, 16384, device="npu", dtype=torch.bfloat16)

full = softmax_batch_invariant(logits, -1)
single = softmax_batch_invariant(logits[:1], -1)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
