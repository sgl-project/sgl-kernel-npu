# grouped_gemma_rmsnorm

## Hardware support

arch22 is split because the two members of that family are validated
differently. A3 means Ascend910_93 series, A2 means Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `ascendc` | Supported | Unverified | Not implemented |
| `torch_native` | Supported | Unverified | Unverified |
| `triton` | Not implemented | Not implemented | Not implemented |
| `tilelang` | Not implemented | Not implemented | Not implemented |

The `ascendc` implementation is registered for arch22, which covers both A2 and
A3; `SGL_KERNEL_ENABLE_A3_ONLY_OPS` in `cmake/legacy_kernels.cmake` builds its
kernel for every SoC outside the Ascend950 series with runtime vector core
discovery (`GetCoreNumAiv()`). A2 is marked unverified because hardware
measurements exist on A3 only; an Ascend950 build omits the native helper and
raises `AttributeError` on `torch.ops.npu.hyper_conn_grouped_norm`. Hardware
validation used Ascend910_9382 with CANN 9.1.0, torch 2.10.0 and
torch_npu 2.10.0. arch35 is not implemented.

## Signature

The default option is `auto`, registered to the same entry point as `ascendc`.
Omitting `impl` or passing `"auto"` selects the Ascend C kernel.

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: Optional[int] = None,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## Description

Applies grouped Gemma-style RMS normalization to fixed-width groups and scales
by `1 + weight`. Groups are the consecutive `group_size`-long spans of `x` in
row-major order, so a hyper-connection residual of shape
`[rows, branches * width]` (or `[rows, branches, width]`) with
`group_size = width` normalizes every branch of every row separately.

With `W = group_size` (or `x.shape[-1]` when `group_size` is `None`),
`L = weight.numel()` and group index `g` counted over the whole tensor in
row-major order:

$$
\mathrm{inv}_g=\left(\frac{1}{W}\sum_{j=0}^{W-1}x_{g,j}^{2}+\varepsilon\right)^{-1/2},
\qquad
y_{g,j}=\mathrm{bf16}\left(x_{g,j}\,\mathrm{inv}_g\left(1+w_{(gW \bmod L)+j}\right)\right)
$$

The weight offset cycles with a period of `L / W` groups, and the operator
requires the group count to be a whole number of those periods, so no group is
scaled by the wrong slice. A weight holding one slice per branch therefore gives
each branch its own slice, and a weight of exactly one group is shared by all
groups. Both the sum of squares and the scaling accumulate in float32; the
result is rounded to BF16 once.

This operator applies no bias, no quantization and no residual add. Unlike
standard RMSNorm, the weight enters with a unit offset (`1 + weight`), and
normalization runs per fixed-width group rather than only across the entire row.

## Parameters and return value

The public operator validates the contract below before selecting an
implementation, so both backends reject the same inputs. Invalid inputs raise
`ValueError` before any kernel launch.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., H]` | `torch.bfloat16` | Required | Input with at least two dimensions; contiguous, on an NPU device. `H` must be a multiple of `group_size`. Zero rows return an empty result. |
| `weight` | `torch.Tensor` | `[L]` | `torch.bfloat16` | Required | Contiguous 1-D weight on x's device. `L` must be a positive multiple of the effective group width, and `x.numel() / width` must be a multiple of `L / width`. |
| `group_size` | `int` or `None` | — | — | `None` | Group width; a multiple of 8 in `[8, 4096]`. `None` uses `H`, making one group per last-dimension span. `bool` is rejected. |
| `eps` | `int` or `float` | — | — | `1e-6` | Positive finite term added to the mean square. `bool` is rejected. |
| `out` | `torch.Tensor` or `None` | x's shape or — | `torch.bfloat16` or — | `None` | Optional pre-allocated contiguous output buffer on x's device with x's shape, `requires_grad=False`, and no memory overlap with `x` or `weight`. When `None`, a new tensor is allocated. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"` or `"torch_native"`. `auto` selects Ascend C. |

Autograd is not supported: any input or pre-allocated `out` with
`requires_grad=True` is rejected. When `out` is provided, it must match `x`'s
shape, dtype (`torch.bfloat16`), and device, be contiguous, and not alias or
overlap `x` or `weight`.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | x's shape | `torch.bfloat16` | Normalized output on x's device. Returns `out` (written in place) when supplied, or a newly allocated contiguous tensor otherwise. Inputs are not modified. |

## ascendc backend

- Dispatches to the internal native helper
  `torch.ops.npu.hyper_conn_grouped_norm`; the sources are in
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md).
- One block processes whole groups in a strided loop. The weight slice is
  loaded once outside the loop when it is shared by all groups, and reloaded
  per group otherwise, which makes a per-branch weight measurably slower than
  a shared one at the same shape.
- Arbitrary shapes `[*leading, cols]` (`ndim >= 2`) are supported functionally,
  and the block count comes from a measured table keyed on the group width and
  `x.size(0)` (the leading dimension of `x`), clamped to the group count and to
  the device's runtime vector core count (`GetCoreNumAiv()`). Because the table
  is tuned assuming the leading dimension is the token count, passing a leading
  batch dimension such as `[1, S, D]` is functionally exact but selects fewer
  blocks than optimal; flatten to `[S, D]` or `[B * S, D]` for full throughput.
  Row counts between 65 and 1022 interpolate from the 128 entry and were not
  measured individually.
- Intermediate arithmetic runs in `float32` throughout and rounds to `bfloat16`
  once on output. The sum of squares accumulates through `ReduceSum`, multiplies
  by a precomputed `1.0f / width` reciprocal, and computes the inverse RMS via
  `Sqrt` + `Div` rather than `torch.rsqrt`, so the normalization factor can
  differ from the reference by ~1 `float32` ulp before the output rounding.

## torch_native backend

- Independent PyTorch implementation; it calls no custom native operator.
- Reshapes to `[-1, L / width, width]` so each group meets its own weight slice
  without a gather, then evaluates the formula above in float32.
- It allocates one float32 copy of `x`, so a prefill-sized input costs twice
  the input in temporary memory. It is a reference, not a performance path.
- Reduction order and reciprocal RMS operations differ from the kernel, so
  bitwise equality is not expected; accuracy tests enforce a tightened
  `BF16_TIGHT` tolerance (`rtol = 8e-3`, 1–2 BF16 ulp). Measured across the test
  matrix, the two backends agree exactly on at least 99.999% of elements, and
  the largest disagreement is one BF16 ulp of the output.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

rows, branches, width = 32, 4, 2048
x = torch.randn(rows, branches * width, device="npu", dtype=torch.bfloat16)
# One weight slice per branch: the slice offset cycles across the row.
weight = torch.randn(branches * width, device="npu", dtype=torch.bfloat16) * 0.02

print(list_impls(grouped_gemma_rmsnorm))  # ['ascendc', 'auto', 'torch_native']
out = grouped_gemma_rmsnorm(x, weight, width, 1e-6)  # Default auto selects Ascend C
out_native = grouped_gemma_rmsnorm(x, weight, width, 1e-6, impl="torch_native")
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
