# fused_gdn_gating

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, *, impl="auto"):
    ...
```

`auto` selects Triton. `torch_native` is an independent PyTorch implementation for comparison.

## Description

Preprocesses the gates used by Gated DeltaNet (GDN). For each row `r` and head `h`, it adds
the time-step bias to `a`, applies thresholded softplus, scales by the log-space decay, and
computes a sigmoid gate from `b`:

$$
x_{r,h}=a_{r,h}+dt\_bias_h
$$

$$
\mathrm{softplus}_{\beta,t}(x)=
\begin{cases}
\frac{1}{\beta}\log(1+e^{\beta x}), & \beta x\le t\\
x, & \beta x>t
\end{cases}
$$

$$
g_{r,h}=-\exp(A\_log_h)\mathrm{softplus}_{\beta,t}(x_{r,h}),
\qquad beta\_output_{r,h}=\mathrm{sigmoid}(b_{r,h})
$$

Here t and β denote the `threshold` and `beta` parameters.

The threshold condition matches `torch.nn.functional.softplus`: the logarithmic branch is used
when `beta * x <= threshold`; otherwise `x` passes through. The calculation of `g` uses FP32
intermediates and returns FP32. `beta_output` has the dtype of `b`.

This is used by the non-`target_verify` extend path in SGLang NPU's
`AscendGDNAttnBackend.forward_extend`; its outputs feed the GDN chunk pipeline. The decode
path computes gating in `fused_sigmoid_gating_delta_rule_update` instead.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `A_log` | `torch.Tensor` | `[H]` | Floating point; FP32 in production | Required | Per-head log-space decay. Must be contiguous for Triton. |
| `a` | `torch.Tensor` | `[R, H]` | FP16/BF16/FP32 | Required | Input to softplus. Must be contiguous for Triton. |
| `b` | `torch.Tensor` | `[R, H]` | FP16/BF16/FP32 | Required | Input to sigmoid; must match `a` in shape and be contiguous for Triton. |
| `dt_bias` | `torch.Tensor` | `[H]` | Floating point; FP32 in production | Required | Per-head time-step bias. Must be contiguous for Triton. |
| `beta` | `float` | — | — | `1.0` | Positive softplus scaling factor. |
| `threshold` | `float` | — | — | `20.0` | Softplus cutoff, compared with `beta * x`. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only: `"auto"`, `"triton"`, or `"torch_native"`. |

Inputs are expected on the same NPU. `a` and `b` must be two-dimensional with the same shape;
`A_log` and `dt_bias` must each contain `H` elements. `R` must be greater than zero and
`beta` must be positive. Triton checks the shape match, contiguity of `a` and `b`, and the
element counts of `A_log` and `dt_bias`. It does not check the strides of those vectors, but
requires them to be contiguous for correct pointer access. `torch_native` follows PyTorch
broadcasting and error behavior, so callers should satisfy the same shape contract with either
implementation.

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `g` | `torch.Tensor` | `[R, H]` | FP32 | Decay gate, allocated on `a`'s device. |
| `beta_output` | `torch.Tensor` | `[R, H]` | Same as `b` | Sigmoid gate, allocated on `b`'s device. |

The inputs are not modified; both outputs are newly allocated.

## triton backend

Computes the thresholded softplus and sigmoid in the Triton kernel. All four tensor inputs must
be contiguous because the kernel uses direct pointer access without strides. The decay gate is
stored in FP32; the sigmoid result is converted to `b.dtype`.

## torch_native backend

Independently computes the formula with PyTorch elementwise operations. The decay calculation
uses FP32 intermediates, while `torch.sigmoid(b)` returns the sigmoid gate in `b.dtype`.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.attention.gdn import fused_gdn_gating

rows, heads = 4, 32
A_log = torch.randn(heads, dtype=torch.float32, device="npu")
dt_bias = torch.randn(heads, dtype=torch.float32, device="npu")
a = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")
b = torch.randn(rows, heads, dtype=torch.bfloat16, device="npu")

g, beta_output = fused_gdn_gating(A_log, a, b, dt_bias)
g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias, impl="torch_native")
```
