# top_p_renorm_prob

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.sample import top_p_renorm_prob


def top_p_renorm_prob(probs, top_ps, *, impl="auto"): ...
```

## Description

Filters FP32 NPU probability rows using top-p and renormalizes by max(sum,1e-20). Top-k thresholds are clamped to [1,vocab]; top-p thresholds to [0,1]. Nucleus selection compares inclusive cumulative probability minus the current value against the threshold. Inputs remain unchanged.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `probs` | Tensor | `Required` | FP32 NPU [batch,vocab], 1<=vocab<=2**23. |
| `top_ps` | float or Tensor | `Required` | Scalar or floating [batch] nucleus thresholds. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | FP32 [batch,vocab] normalized probabilities, same device as probs. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sample import top_p_renorm_prob

p = torch.softmax(torch.randn(2, 97, device="npu"), -1)
y = top_p_renorm_prob(p, 0.95, impl="triton")
```
