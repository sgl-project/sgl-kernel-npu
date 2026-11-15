# reduce_sum

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import reduce_sum_batch_invariant

def reduce_sum_batch_invariant(input, dim, keepdim=False, *, impl="auto"):
    ...
```

该算子转发到 `torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant`。使用前需
先安装外部组件，见[安装说明](installation.md)。

## 功能说明

沿单个维度求和。归约顺序与行数无关，因此沿特征维归约时，单行计算与在大 batch 中
计算的结果逐位相同。沿 batch 维归约会把多行混合，结果应随 batch 变化。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | 任意 | `torch.float16`、`torch.bfloat16`、`torch.float32` | 必填 | NPU 上的输入。 |
| `dim` | `int` | — | — | 必填 | 单个归约维度，负数从末尾计数；不支持维度列表。 |
| `keepdim` | `bool` | — | — | `False` | 是否保留被归约维（长度 1）。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | 去掉 `dim` 后的形状；`keepdim=True` 时该维为 1 | 与 input 相同 | 新分配张量；不修改输入。 |

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import reduce_sum_batch_invariant

x = torch.randn(257, 16384, device="npu", dtype=torch.bfloat16)

full = reduce_sum_batch_invariant(x, -1)
single = reduce_sum_batch_invariant(x[:1], -1)
torch.testing.assert_close(full[:1], single, atol=0, rtol=0)
```
