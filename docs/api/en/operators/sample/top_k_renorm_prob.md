# top_k_renorm_prob

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.sample import top_k_renorm_prob


def top_k_renorm_prob(probs, top_ks, *, impl="auto"): ...
```

## Description

Filters FP32 NPU probability rows using top-k and renormalizes by max(sum,1e-20). Top-k thresholds are clamped to [1,vocab]; top-p thresholds to [0,1]. Nucleus selection compares inclusive cumulative probability minus the current value against the threshold. Inputs remain unchanged.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `probs` | Tensor | `Required` | FP32 NPU [batch,vocab], 1<=vocab<=2**23. |
| `top_ks` | int or Tensor | `Required` | Scalar or integer [batch] keep counts. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | FP32 [batch,vocab] normalized probabilities, same device as probs. |

## triton backend

- Sorts FP32 probabilities with PyTorch, then uses Triton to filter the first k elements in each row.
- Encodes token indices to restore vocabulary order and renormalizes with a denominator floor of 1e-20, without modifying input probabilities.

## torch_native backend

- Uses PyTorch sort, masking, sum, and scatter to filter, normalize, and restore vocabulary order.
- Clamps each row’s k to [1,vocab_size], masks ranks beyond the first k, and normalizes with a row-sum floor of 1e-20.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sample import top_k_renorm_prob

p = torch.softmax(torch.randn(2, 97, device="npu"), -1)
y = top_k_renorm_prob(p, 16, impl="triton")
```
