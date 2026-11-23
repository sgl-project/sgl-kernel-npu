# top_k_top_p_renorm_prob

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Unverified |
| `torch_native` | Supported | Unverified |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.sample import top_k_top_p_renorm_prob


def top_k_top_p_renorm_prob(probs, top_ks, top_ps, max_keep=None, *, impl="auto"): ...
```

## Description

Applies top-p to the original distribution, then top-k, and renormalizes the joint support once by max(sum,1e-20). Both filters keep a prefix of the descending row, so the surviving columns are the intersection of their prefix masks; the inclusive cumulative probability minus the current value is compared with the top-p threshold.

`max_keep` is an optional upper bound on every row top-k. When it bounds every row, selecting only that many leading columns is exact. If a row top-k exceeds `max_keep`, that row is truncated to `max_keep` columns, so pass the exact batch maximum or a safe upper bound. `None` filters the full vocabulary and is exact for any input, but it is the slow path; prefer a bound in hot paths.

Equal probabilities at the selection boundary may resolve to different token indices than a full sort; the kept values and their total mass are unchanged.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `probs` | Tensor | `Required` | FP32 NPU [batch,vocab], 1<=vocab<=2**23. |
| `top_ks` | int or Tensor | `Required` | Scalar or integer [batch] keep counts. |
| `top_ps` | float or Tensor | `Required` | Scalar or floating [batch] nucleus thresholds. |
| `max_keep` | int | `None` | Optional positive upper bound on every row top-k; `None` uses the full vocabulary path. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tensor | FP32 [batch,vocab] normalized probabilities, same device as probs. |

## triton backend

- Selects the leading `max_keep` columns with PyTorch top-k and computes their cumulative sum.
- Uses a Triton kernel to apply the joint top-p/top-k mask and renormalize the selected prefix with a denominator floor of 1e-20.
- Restores vocabulary order with a narrow scatter over the selected columns, so the full-vocabulary write-back is avoided.
- Widths above 2048 fall back to the PyTorch filter path.

## torch_native backend

- Independent two-pass reference: top-p over the full distribution, then top-k. It ignores `max_keep`.
- Uses PyTorch sort, masking, sum, and scatter for filtering, normalization, and vocabulary-order restoration.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.sample import top_k_top_p_renorm_prob

p = torch.softmax(torch.randn(2, 97, device="npu"), -1)
y = top_k_top_p_renorm_prob(p, 16, 0.95, max_keep=16, impl="triton")
```