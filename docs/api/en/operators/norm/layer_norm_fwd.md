# layer_norm_fwd

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.norm import layer_norm_fwd

def layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation=None,
    *,
    impl="auto",
):
    ...
```

`auto` and `triton` currently refer to the same entry point. Execution failures do not trigger an automatic fallback.

## Description

Applies LayerNorm or RMSNorm to each group of each input row, with affine transformation and optional SiLU gating. For input `[M, N]`, the group width is `H = group_size` and the number of groups is `G = N // H`. An omitted group width uses `H = N`.

If `z` is provided and `norm_before_gate=False`, first set `u = x * SiLU(z)`; otherwise set `u = x`. Each group then uses the following calculation:

| Mode | Statistics and normalization |
| --- | --- |
| LayerNorm | `mean = mean(u)`, `rstd = 1 / sqrt(mean((u - mean) ** 2) + eps)`, `normalized = (u - mean) * rstd` |
| RMSNorm | `rstd = 1 / sqrt(mean(u ** 2) + eps)`, `normalized = u * rstd`; no mean tensor is returned |

The normalized values are multiplied by `weight` and added to the optional `bias`. If `z` is provided and `norm_before_gate=True`, multiply the affine output by `SiLU(z)`. SiLU is `z * sigmoid(z)`.

`activation` is a compatibility argument that currently does not select an implementation branch. Passing a different string does not change the gate function.

## Parameters and return values

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[M, N]` | `torch.float32`, `torch.float16`, `torch.bfloat16` | Required | `N > 0` and last-dimension stride 1. Triton input must be on NPU. Extra spacing between rows is allowed. |
| `weight` | `torch.Tensor` | `[N]` | Floating point | Required | Per-channel multiplier on x's device, with last-dimension stride 1. |
| `bias` | `torch.Tensor` or `None` | `[N]` or — | Floating point or — | Required; may be `None` | Optional per-channel bias on x's device, with last-dimension stride 1. |
| `eps` | `float` | — | — | Required | Stability term added to each group's variance or mean square. A finite positive value is recommended. |
| `z` | `torch.Tensor` or `None` | `[M, N]` or — | Floating point or — | `None` | Optional gate input on x's device, with last-dimension stride 1. This entry point requires a two-dimensional z matching x's shape. |
| `out` | `torch.Tensor` or `None` | `[M, N]` or — | Floating point or — | `None` | Reusable output on x's device, with last-dimension stride 1; preferably x.dtype. When provided, it is written and returned by identity. Otherwise a new output is allocated. |
| `group_size` | `int` or `None` | — | — | `None` | Positive divisor of N. None selects N. Must not exceed `65536 // x.element_size()`. |
| `norm_before_gate` | `bool` | — | — | `True` | True gates after normalization and affine transformation; False normalizes the gated input. |
| `is_rms_norm` | `bool` | — | — | `False` | False selects LayerNorm; True selects RMSNorm. |
| `activation` | `str` or `None` | — | — | `None` | Compatibility argument, currently ignored. Gating always uses SiLU. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"triton"` or `"torch_native"`. |

Returns `(out, mean, rstd)`:

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `out` | `torch.Tensor` | `[M, N]` | `x.dtype` by default; a supplied buffer uses `out.dtype` | Normalized and optionally gated output on x's device. |
| `mean` | `torch.Tensor` or `None` | `[G * M]` or — | `torch.float32` or — | Per-group means for LayerNorm; None for RMSNorm. |
| `rstd` | `torch.Tensor` | `[G * M]` | `torch.float32` | Per-group reciprocal standard deviations or reciprocal root mean squares. |

Statistics use flattened group-major order: group g, row m is stored at `g * M + m`. Use `stats.view(G, M)` to inspect this layout. The public function performs shape, stride, grouping and group-width checks. It does not make contiguous copies or add dtype or device checks. Callers must provide suitable devices and dtypes. When out is omitted, x, weight, bias and z are read-only.

## triton backend

- Uses `_layer_norm_fwd_1pass_kernel_npu`, with `BLOCK_M=48`.
- Addresses input, output and gate using their row strides; the last dimension must remain contiguous.
- Computes input, weight, bias and gate values in FP32. Results are stored in the output dtype; statistics remain FP32.

## torch_native backend

- Prepares `[M, G, H]` group views and statistic views within this API, then uses PyTorch for normalization, affine transformation and gating.
- Supports output-buffer reuse and returns the same statistic layout as Triton.

## Example

Install a wheel containing this API and the NPU runtime dependencies before running the example. Use separate outputs for backend comparisons so a shared buffer does not overwrite results.

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import layer_norm_fwd

torch.npu.set_device(0)
x = torch.randn(4, 128, device="npu", dtype=torch.float32)
weight = torch.ones(128, device="npu", dtype=x.dtype)
z = torch.randn_like(x)
buffer = torch.empty_like(x)

# Grouped RMSNorm with post-affine SiLU gating and output-buffer reuse.
y, mean, rstd = layer_norm_fwd(
    x, weight, None, 1e-5,
    z=z, out=buffer, group_size=32,
    norm_before_gate=True, is_rms_norm=True,
    impl="triton",
)
assert y is buffer
assert mean is None
print(rstd.shape)  # torch.Size([16]): 4 groups * 4 rows

reference, _, _ = layer_norm_fwd(
    x, weight, None, 1e-5,
    z=z, group_size=32, is_rms_norm=True,
    impl="torch_native",
)

# Default LayerNorm returns mean and rstd; auto currently selects Triton.
ln_output, ln_mean, ln_rstd = layer_norm_fwd(x, weight, None, 1e-5)
```
