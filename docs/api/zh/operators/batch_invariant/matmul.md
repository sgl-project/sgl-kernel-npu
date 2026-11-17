# matmul

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

def matmul_batch_invariant(input, other, *, impl="auto"):
    ...
```

该算子转发到 `torch.ops.batch_invariant_ops.npu_matmul_batch_invariant`。使用前需先
安装外部组件，见[安装说明](installation.md)。

## 功能说明

支持广播的批量矩阵乘。设 $X \in \mathbb{R}^{\ldots \times N\times K}$、
$W \in \mathbb{R}^{\ldots \times K\times M}$，前置 batch 维按常规广播规则对齐。
$K$ 维的归约顺序与 batch 大小无关，因此每个矩阵的结果在单独计算与批量计算下逐位
相同。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[..., N, K]` | `torch.float16`、`torch.bfloat16`、`torch.float32` | 必填 | NPU 上的左操作数。 |
| `other` | `torch.Tensor` | `[..., K, M]` | 与 input 相同 | 必填 | 右操作数，前置维可与 input 广播；允许转置后的非连续视图。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[..., N, M]` | 与 input 相同 | 新分配张量；不修改输入。 |

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import matmul_batch_invariant

a = torch.randn(8, 1, 4096, device="npu", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)

full = matmul_batch_invariant(a, b)
single = matmul_batch_invariant(a[:1], b)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
