# fused_scale_shift

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.norm import fused_scale_shift

def fused_scale_shift(x, scale, shift, scale_constant=1.0, *, impl="auto"):
    ...
```

`auto` and `triton` share the same registered entry point and lazy import cache.
The default selects Triton without automatic fallback. Both implementations are
declared for all known generations; the table records hardware validation, which
has not been performed as part of this migration.

## Description

Applies scalar or channel scaling and scalar, channel or elementwise shift to
an input $X \in \mathbb{R}^{B\times L\times H}$. Modulation tensors are flattened
before use: element counts determine broadcasting, regardless of their original
shape. Let $N=BLH$, $n_s=\mathrm{scale.numel()}$ and
$n_t=\mathrm{shift.numel()}$:

$$
S_h=\begin{cases}s_0,&n_s=1,\\s_h,&n_s=H,\end{cases}
\qquad
T_{blh}=\begin{cases}
\mathrm{reshape}(t,[B,L,H])_{blh},&n_t=N,\\
t_0,&n_t=1\ne N,\\
t_h,&n_t=H\ne N.
\end{cases}
$$

$$
c=\begin{cases}\mathrm{scale\_constant},&n_t=N,\\1,&n_t\ne N,\end{cases}
\qquad Y_{blh}=X_{blh}(c+S_h)+T_{blh}.
$$

This preserves the original two kernels' behavior: `scale_constant` is used
only for a full-input shift. Broadcast shift always uses `1 + scale`, including
when `scale_constant=0`. The full-input branch takes precedence when counts
overlap. In particular, for `B * L == 1`, a channel shift is a full-input shift.
Triton requires `H` scale values for that branch; a scalar is valid only if `H=1`.

The formula describes the mathematical operation. Floating intermediates follow
the dtype rules below, and the output is cast to `x.dtype`. Inputs are not modified.

## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[B, L, H]` | float32, float16 or bfloat16 | Required | Nonempty three-dimensional input; all dimensions must be positive. Triton requires contiguous NPU storage. |
| `scale` | `torch.Tensor` | Any shape containing `1` or `H` values | float32, float16 or bfloat16 | Required | Scalar or channel scale; same device as x. Triton requires contiguous storage and exactly `H` values when shift.numel() equals x.numel(). |
| `shift` | `torch.Tensor` | Any shape containing `1`, `H` or `B*L*H` values | float32, float16 or bfloat16 | Required | Scalar, channel or full-input shift; same device as x. Triton requires contiguous storage. Full-input values are reshaped to x.shape in flattened order. |
| `scale_constant` | Python `int` or `float` | — | — | `1.0` | Finite scalar, excluding bool. Used only for the full-input-shift branch. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; `"auto"`, `"triton"` or `"torch_native"`. |

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[B, L, H]` | x.dtype | Newly allocated contiguous tensor on x.device; does not alias an input. |

Tensor dtypes may differ. Use finite inputs for accuracy comparisons. The public
entry handles dispatch; the Triton wrapper checks types, dimensions, element
counts, layouts, devices and contiguity before querying launch properties.
Unsupported inputs raise `AssertionError` before a kernel is launched. Per-batch
or per-frame modulation with more than `H` scale values is not supported.

## triton backend

- Located in `implementation/triton/norm/scale_shift.py`. The original kernel
  bodies and autotune configurations are retained.
- `fused_scale_shift_kernel` handles broadcast shift with row tiles 128/112 and
  a 128-channel tile. Scale and shift independently support scalar or channel
  values. `fused_scale_shift_kernel_2` handles full-input shift with row tiles
  96/80/64 and a 128-channel tile, reading one scale value per channel.
- Both use persistent vector-core grids and masked row/channel tails. No padding
  or layout conversion is added to the wrapper.
- The constant addition uses scale's dtype. A channel shift is converted to
  float32 in the broadcast kernel, while a scalar shift keeps its dtype. The
  full-input kernel always converts shift to float32. Multiplication follows
  Triton type promotion; mixing float16 and bfloat16 prefers float16.
- The legacy `sgl_kernel_npu.norm.scale_shift` script remains unchanged. Access
  the migrated implementation and its input validation through the public
  `sgl_kernel_npu.operators.norm.fused_scale_shift` interface.

## torch_native backend

- Uses independent PyTorch arithmetic and does not import Triton. Flattens
  modulation tensors and preserves the same element-count branch precedence.
- Keeps the constant addition in scale.dtype, matches the float32 shift
  conversions and explicitly handles Triton's float16/bfloat16 promotion.
- Assumes valid inputs without explicit metadata validation. It also supports
  non-contiguous inputs and scalar scale with full-input shift, although those
  combinations are outside the Triton contract.
- Separate arithmetic operations and kernel multiply-add fusion can cause
  rounding differences; bitwise equality is not guaranteed.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_scale_shift

x = torch.randn(2, 3, 129, device="npu", dtype=torch.float32) * 0.25
scale = torch.randn(1, 1, 129, device="npu", dtype=torch.float32) * 0.25
shift = torch.randn_like(x) * 0.25

# Full-input shift: scale_constant=0 gives x * scale + shift.
output = fused_scale_shift(x, scale, shift, scale_constant=0)
reference = fused_scale_shift(x, scale, shift, scale_constant=0, impl="torch_native")
torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)

# Broadcast shift: uses x * (1 + scale) + channel_shift.
channel_shift = torch.zeros_like(scale)
broadcast_output = fused_scale_shift(x, scale, channel_shift)
```
