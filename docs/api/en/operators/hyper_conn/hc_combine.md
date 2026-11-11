# hc_combine

## Hardware support

| impl | arch22 | arch35 |
| --- | --- | --- |
| `triton` / `auto` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` → `triton`.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(block, residual, normed, weight, hc, hs, *, impl="auto"):
    ...
```

## Description

Add the same block row to four residual groups, each with its own gate. With `hc=4` and `hs=2560`:

```text
gate = 2 * sigmoid(normed @ weight.T / 4)  # [R,4]
output[r, h*2560+d] = residual[r, h*2560+d] + gate[r, h] * block[r, d]
```

Here `h=0..3` and `d=0..2559`. The output is contiguous BF16 `[R,10240]`. `auto` selects the Triton/native hybrid.

All tensors are on the same NPU. Inputs are not modified; zero rows return empty output.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `block` | Tensor | [R,2560], BF16 | required | Contiguous NPU input. |
| `residual / normed` | Tensor | [R,10240], BF16 | required | Contiguous on block.device. |
| `weight` | Tensor | [4,10240], BF16 | required | Contiguous on block.device. |
| `hc / hs` | int | — | required | 4 / 2560. |
| `impl` | str | — | "auto" | Keyword-only: auto, triton, torch_native. |

| Return | Type | Shape | Dtype | Device / layout |
| --- | --- | --- | --- | --- |
| `output` | Tensor | [R,10240] | BF16 | Input device; contiguous, newly allocated. |

## triton

Default implementation. Checks input metadata before launch. Supports graph capture and replay after in-place input updates.

## torch_native

Independent CPU FP64 reference; returns results to the input device. Use outside graph capture. Select explicitly with `impl="torch_native"`.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(1, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(1, 2560, device="npu", dtype=residual.dtype)
weight = torch.zeros(4, 10240, device="npu", dtype=residual.dtype)
y = hc_combine(block, residual, residual, weight, 4, 2560, impl="auto")
```
