# hc_combine

## 硬件支持

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | 支持 | 支持 |
| `torch_native` | 支持 | 支持 |

## 签名

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    ...
```

## 说明

将同一行 block 加到四组 residual 中，每组使用自己的门控。固定 `hc=4`、`hs=2560`：

```text
gate = 2 * sigmoid(normed @ weight.T / 4)  # [R,4]
output[r, h*2560+d] = residual[r, h*2560+d] + gate[r, h] * block[r, d]
```

其中 `h=0..3`，`d=0..2559`。输出为连续 BF16 `[R,10240]`。`auto` 选择 Triton/native 混合实现。

所有张量位于同一 NPU，输入不修改；零行返回空输出。

## 参数与返回值

| 参数 | 类型 | 形状 / 类型 | 默认值 | 约束 |
| --- | --- | --- | --- | --- |
| `block` | Tensor | [R,2560], BF16 | required | 连续 NPU 输入。 |
| `residual / normed` | Tensor | [R,10240], BF16 | required | block 同设备连续输入。 |
| `weight` | Tensor | [4,10240], BF16 | required | block 同设备连续权重。 |
| `hc / hs` | int | — | required | 4 / 2560。 |
| `impl` | str | — | "auto" | 仅关键字参数：auto、triton、torch_native。 |

| 返回值 | 类型 | 形状 | 类型 | 设备 / 布局 |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,10240] | BF16 | 输入设备；新分配的连续张量。 |

## triton

默认实现。启动前检查输入元数据，支持 graph capture，以及原地更新输入后的 replay。

## torch_native

独立 CPU FP64 参考实现，结果返回输入设备。仅在图外使用，通过 `impl="torch_native"` 显式选择。

## 示例

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(1, 2560, device="npu", dtype=residual.dtype)
weight = torch.zeros(4, 10240, device="npu", dtype=residual.dtype)
y = hc_combine(block, residual, residual, weight, 4, 2560, impl="auto")
```
