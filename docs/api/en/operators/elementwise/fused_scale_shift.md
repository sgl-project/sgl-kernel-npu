# fused_scale_shift

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.elementwise import fused_scale_shift

def fused_scale_shift(x, scale, shift, scale_constant=1.0, *, impl="auto"):
    ...
```

`impl="auto"` selects the Triton backend by default.

## Description

Applies scalar or channel scaling and scalar, channel or elementwise shift to
an input $X \in \mathbb{R}^{B\times L\times H}$. Modulation tensors are flattened
before use: element counts determine broadcasting. Let $N=BLH$,
$n_s=\mathrm{scale.numel()}$ and
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

`scale_constant` is used only for a full-input shift. Broadcast shift always
uses `1 + scale`, including when `scale_constant=0`. The full-input branch takes
precedence when counts overlap. In particular, for `B * L == 1`, a channel shift
is a full-input shift.
Triton requires `H` scale values for that branch; a scalar is valid only if `H=1`.

The output has dtype `x.dtype`. Inputs are not modified.

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

Tensor dtypes may differ. Inputs must satisfy the constraints above; invalid
inputs to the Triton backend raise `AssertionError`. Per-batch or per-frame
modulation with more than `H` scale values is not supported.

## triton backend

Select with `impl="triton"` or `impl="auto"` for NPU execution. All inputs must
be contiguous NPU tensors and satisfy the shape and dtype constraints above.

## torch_native backend

Select with `impl="torch_native"` for accuracy comparison or debugging. Follow
the input constraints above. Use finite inputs for comparisons; floating-point
rounding differences between backends can occur, so bitwise equality is not
guaranteed.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.elementwise import fused_scale_shift

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

## Tests

The public-interface suite contains 46 default cases and 266 optional
`extra_checks` cases. Run accuracy tests with:

```bash
pytest tests/python/operators/elementwise/test_scale_shift.py --enable-perf=false
pytest tests/python/operators/elementwise/test_scale_shift.py -m extra_checks --enable-perf=false
```
