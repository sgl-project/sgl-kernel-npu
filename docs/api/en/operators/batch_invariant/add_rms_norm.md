# add_rms_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `batch_invariant_ops` | Supported | Supported |

`auto` is an alias of `batch_invariant_ops`.

## Signature

```python
from sgl_kernel_npu.operators.batch_invariant import add_rms_norm_batch_invariant

def add_rms_norm_batch_invariant(x, residual, weight, eps, *, impl="auto"):
    ...
```

The operator forwards to `torch.ops.batch_invariant_ops.npu_add_rms_norm_batch_invariant`.
Install the external packages first; see [installation](installation.md).

## Description

Adds the residual to the input, then applies RMS normalization over the last
dimension with a multiplicative weight. For
$X \in \mathbb{R}^{B\times H}$, residual $R$, weight $w$ and
$S = X + R$:

$$
v_i = \frac{1}{H}\sum_{j=0}^{H-1}S_{ij}^{2}, \qquad
Y_{ij} = \frac{S_{ij}}{\sqrt{v_i+\varepsilon}}w_j
$$

The reduction order does not depend on $B$, so a row's normalized output,
reciprocal standard deviation and residual sum are bitwise identical whether
the row is processed alone or inside a larger batch.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, H]` | `torch.float16`, `torch.bfloat16` | Required | Input on NPU. |
| `residual` | `torch.Tensor` | `[B, H]` | Same as `x` | Required | Residual added before normalization, on the same device. |
| `weight` | `torch.Tensor` | `[H]` | Same as `x` | Required | Per-channel multiplicative weight. |
| `eps` | `float` | — | — | Required | Stability term added to the mean square, such as `1e-6`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"` or `"batch_invariant_ops"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, H]` | Same as `x` | Normalized result. |
| `rstd` | `torch.Tensor` | `[B, 1]` | Floating point | Reciprocal standard deviation of each row. |
| `residual_out` | `torch.Tensor` | `[B, H]` | Same as `x` | Residual sum `X + R`. |

Inputs are not modified; every output is newly allocated.

## Example

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
