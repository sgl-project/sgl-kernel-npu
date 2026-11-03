# add_rms_norm

## 芯片支持

| 实现 | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | 支持 | 支持 |

`auto` 与 `batch_invariant_ops` 注册到同一个入口函数。

## 函数原型

```python
from sgl_kernel_npu.operators.batch_invariant import add_rms_norm_batch_invariant

def add_rms_norm_batch_invariant(x, residual, weight, eps, *, impl="auto"):
    ...
```

该算子转发到 `torch.ops.batch_invariant_ops.npu_add_rms_norm_batch_invariant`。使用前
需先安装外部组件，见[安装说明](installation.md)。

## 功能说明

先将残差加到输入上，再沿最后一维做带权重的 RMS 归一化。设
$X \in \mathbb{R}^{B\times H}$、残差为 $R$、权重为 $w$，记 $S = X + R$：

$$
v_i = \frac{1}{H}\sum_{j=0}^{H-1}S_{ij}^{2}, \qquad
Y_{ij} = \frac{S_{ij}}{\sqrt{v_i+\varepsilon}}w_j
$$

归约顺序与 $B$ 无关，因此每一行的归一化结果、倒数标准差与残差和，在单独计算与批量
计算下逐位相同。

## 参数与返回值

| 参数 | 类型 | Shape | Dtype | 默认值 | 描述与约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, H]` | `torch.float16`、`torch.bfloat16` | 必填 | NPU 上的输入。 |
| `residual` | `torch.Tensor` | `[B, H]` | 与 `x` 相同 | 必填 | 归一化前相加的残差，与 `x` 同设备。 |
| `weight` | `torch.Tensor` | `[H]` | 与 `x` 相同 | 必填 | 每个通道的乘法权重。 |
| `eps` | `float` | — | — | 必填 | 均方项上的稳定项，如 `1e-6`。 |
| `impl` | `str` | — | — | `"auto"` | 仅关键字；可选 `"auto"`、`"batch_invariant_ops"`。 |

| 返回值 | 类型 | Shape | Dtype | 描述与约束 |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | 与 `x` 相同 | 归一化结果。 |
| `rstd` | `torch.Tensor` | `[B, 1]` | 浮点 | 每一行的倒数标准差。 |
| `residual_out` | `torch.Tensor` | `[B, H]` | 与 `x` 相同 | 残差和 `X + R`。 |

输入不会被修改，三个返回值均为新分配张量。

## 示例

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.batch_invariant import add_rms_norm_batch_invariant

x = torch.randn(16, 7168, device="npu", dtype=torch.bfloat16)
residual = torch.randn(16, 7168, device="npu", dtype=torch.bfloat16)
weight = torch.randn(7168, device="npu", dtype=torch.bfloat16)

full = add_rms_norm_batch_invariant(x, residual, weight, 1e-6)
single = add_rms_norm_batch_invariant(x[:1], residual[:1], weight, 1e-6)
for full_out, single_out in zip(full, single):
    torch.testing.assert_close(full_out[:1], single_out, atol=0, rtol=0)
```
