# hc_mix

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
kernels for every SoC outside the Ascend950 series with runtime vector core
discovery (`GetCoreNumAiv()`). A2 is marked unverified because hardware
measurements exist on A3 only; an Ascend950 build omits the native helpers and
raises `AttributeError` on `torch.ops.npu.hyper_conn_scaled_silu` or
`torch.ops.npu.hyper_conn_branch_mix`. Hardware validation used Ascend910_9382
with CANN 9.1.0, torch 2.10.0 and torch_npu 2.10.0. arch35 is not implemented.

## Signature

The default option is `auto`, registered to the same entry point as `ascendc`.
Omitting `impl` or passing `"auto"` selects the Ascend C pipeline.

```python
from sgl_kernel_npu.operators.hyper_conn import hc_mix

def hc_mix(
    normed,
    down_weight,
    up_weight,
    hc_count,
    hidden_size,
    *,
    impl="auto",
):
    ...
```

## Description

Computes the low-rank gated branch average of a hyper-connection residual. Given
`normed` of shape `[rows, hc_count * hidden_size]`, `down_weight` of shape
`[low_rank, hc_count * hidden_size]` and `up_weight` of shape
`[hc_count * hidden_size, low_rank]`, with $C = \mathrm{hc\_count}$,
$H = \mathrm{hidden\_size}$ and $\sigma(z)=1/(1+e^{-z})$:

$$
t = \frac{x\,W_{\text{down}}^{\top}}{C},
\qquad
a = \mathrm{bf16}\!\left(t\,\sigma(t)\right),
\qquad
g = a\,W_{\text{up}}^{\top}
$$

$$
\mathrm{out}_{m,j} = \mathrm{bf16}\!\left(\frac{1}{C}\sum_{c=0}^{C-1} \sigma(g_{m,c,j})\,x_{m,c,j}\right)
$$

where $x = \mathrm{normed}$, $x_{m,c,j} = \mathrm{normed}_{m,\,cH+j}$,
$g_{m,c,j} = g_{m,\,cH+j}$, and $\mathrm{out}_{m,j}$ is written to
`output[m, j]`. Intermediate arithmetic within the scaled-SiLU and branch-mix
stages runs in `float32`, rounding to `bfloat16` only at global-memory output
boundaries (`a` before the up-projection and `out`), matching the upstream
SGLang kernel contract.

Both sigmoid tails saturate cleanly rather than producing NaN: `exp(-z)`
overflowing to infinity gives a gate of 0, and underflowing to 0 gives a gate
of 1.

## Parameters and return value

The public operator validates the contract below before selecting an
implementation, so both backends reject the same inputs. Invalid inputs raise
`ValueError` before any kernel launch.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `normed` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | Required | Normalized residual; contiguous 2-D tensor on an NPU device. `rows == 0` returns an empty `[0, hidden_size]` result. |
| `down_weight` | `torch.Tensor` | `[low_rank, hc_count * hidden_size]` | `torch.bfloat16` | Required | Down-projection weight; contiguous 2-D tensor on normed's device. `low_rank` must be a multiple of 16 in `[16, 4096]`. |
| `up_weight` | `torch.Tensor` | `[hc_count * hidden_size, low_rank]` | `torch.bfloat16` | Required | Up-projection weight; contiguous 2-D tensor on normed's device. Accepts both standard ND storage and pre-converted `FRACTAL_NZ` (`torch_npu.npu_format_cast(w, 29)`). |
| `hc_count` | `int` | — | — | Required | Hyper-connection branch count; a power of two in `[1, 8]`. `bool` is rejected. |
| `hidden_size` | `int` | — | — | Required | Per-branch hidden size; a multiple of 16 in `[16, 4096]`. `bool` is rejected. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"` or `"torch_native"`. `auto` selects Ascend C. |

Autograd is not supported: an input with `requires_grad` is rejected.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, hidden_size]` | `torch.bfloat16` | Newly allocated contiguous output on normed's device. Inputs are not modified. |

## ascendc backend

- Runs the down and up projections via `torch.nn.functional.linear` (`aclnn`)
  and the two vector stages via the internal Ascend C helpers
  `torch.ops.npu.hyper_conn_scaled_silu` and
  `torch.ops.npu.hyper_conn_branch_mix` (sources in
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md)); both vector
  kernels keep intermediate arithmetic in `float32` and cast to `bfloat16` once
  when writing their outputs (`a` before the up-projection and `out`) to global
  memory. Pass only tensors validated through the public operator rather than
  calling the helper bindings directly.
- Pre-converting `up_weight` once at weight-loading time with
  `torch_npu.npu_format_cast(up_weight, 29)` (`FRACTAL_NZ`, with
  `torch.npu.config.allow_internal_format = True` enabled during conversion) is
  supported with zero steady-state extra memory and agrees with `ND` within
  standard BF16 GEMM rounding (bitwise equal at decode batch buckets on CANN
  9.1.0), substantially reducing the up-projection time at decode batch sizes.
- In `hyper_conn_branch_mix`, one task covers one `(row, column tile)` across
  all branches in a single strided copy, with block counts and column tiles
  selected from a measured table keyed on `hidden_size` and `rows` (clamped to
  the task count and runtime vector core count via `GetCoreNumAiv()`). Row
  counts between 65 and 1022 interpolate from the 128-row entry.

## torch_native backend

- Independent PyTorch implementation; it calls no custom native operator.
- Uses `torch.nn.functional.linear` for the two projections and evaluates
  intermediate arithmetic within the scaled-SiLU and branch-mix stages in
  `float32` using the `1 / (1 + exp(-x))` sigmoid form, rounding to `bfloat16`
  only at global-memory output boundaries (`a` before the up-projection and
  `out`).
- Allocates float32 temporaries of the full `[rows, hc_count, hidden_size]`
  shape, so prefill-sized calls are memory heavy. It is a reference, not a
  performance path.
- Transcendental implementations differ across backends, so bitwise equality is
  not guaranteed; accuracy tests enforce a tightened `BF16_TIGHT` tolerance
  (`rtol = 8e-3`, 1–2 BF16 ulp).

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.hyper_conn import hc_mix

rows, hc_count, hidden_size, low_rank = 32, 4, 2048, 320
total = hc_count * hidden_size
normed = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
down_weight = (
    torch.randn(low_rank, total, device="npu", dtype=torch.bfloat16) * 0.05
)
up_weight = (
    torch.randn(total, low_rank, device="npu", dtype=torch.bfloat16) * 0.2
)
# Optional one-time weight format conversion for faster decode GEMM:
torch.npu.config.allow_internal_format = True
up_weight_nz = torch_npu.npu_format_cast(up_weight.contiguous(), 29)

print(list_impls(hc_mix))  # ['ascendc', 'auto', 'torch_native']
out = hc_mix(normed, down_weight, up_weight_nz, hc_count, hidden_size)
out_native = hc_mix(
    normed, down_weight, up_weight, hc_count, hidden_size, impl="torch_native"
)
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
