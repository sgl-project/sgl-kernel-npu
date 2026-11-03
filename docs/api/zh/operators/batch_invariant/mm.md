# mm

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

def mm_batch_invariant(input, mat2, *, impl="auto"):
    ...
```

该算子转发到 `torch.ops.batch_invariant_ops.npu_mm_batch_invariant`。使用前需先安装
CANN 算子包与 torch extension，见[安装说明](installation.md)。

## 功能说明

计算 $Y = XW$，其中 $X \in \mathbb{R}^{N\times K}$、$W \in \mathbb{R}^{K\times M}$。
$K$ 维的归约顺序与 $N$ 无关，因此每一行输出在“单独计算”和“在大 batch 中计算”两种
情况下逐位相同，这就是批次不变性；数值上仍是普通矩阵乘。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[N, K]` | `torch.float16`、`torch.bfloat16`、`torch.float32` | 必填 | NPU 上的左操作数；`N > 0`，`K > 0`。 |
| `mat2` | `torch.Tensor` | `[K, M]` | 与 input 相同 | 必填 | 同设备右操作数；允许传入转置后的非连续视图。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[N, M]` | 与 input 相同 | 新分配张量；不修改输入。 |

公开接口直接透传参数，不做输入检查。外部包缺失时首次调用抛出导入错误，不会回退到
其他实现。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import mm_batch_invariant

a = torch.randn(257, 7168, device="npu", dtype=torch.bfloat16)
b = torch.randn(16384, 7168, device="npu", dtype=torch.bfloat16).transpose(0, 1)

full = mm_batch_invariant(a, b)
single = mm_batch_invariant(a[:1], b)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
