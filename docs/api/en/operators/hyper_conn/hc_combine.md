# hc_combine

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
raises `AttributeError` on `torch.ops.npu.hyper_conn_branch_combine`. Hardware
validation used Ascend910_9382 with CANN 9.1.0, torch 2.10.0 and
torch_npu 2.10.0. arch35 is not implemented.

## Signature

The default option is `auto`, registered to the same entry point as `ascendc`.
Omitting `impl` or passing `"auto"` selects the Ascend C pipeline.

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## Description

Projects per-branch injection gates from `normed_residual` and adds the gated
sub-block output `block_output` back into every branch of `residual`. With
$C = \mathrm{hc\_count}$, $H = \mathrm{hidden\_size}$,
$\hat{r} = \mathrm{normed\_residual}$, and $\sigma(z)=1/(1+e^{-z})$:

$$
z = \hat{r}\,W_{\text{inject}}^{\top},
\qquad
g_{m,c} = 2\,\sigma\!\left(\frac{z_{m,c}}{C}\right)
$$

$$
\mathrm{out}_{m,c,j} = \mathrm{bf16}\!\left(r_{m,c,j} + b_{m,j}\,g_{m,c}\right)
$$

where $r_{m,c,j} = \mathrm{residual}_{m,\,cH+j}$,
$b_{m,j} = \mathrm{block\_output}_{m,j}$, and $\mathrm{out}_{m,c,j}$ is written
to `output[m, c * H + j]`. Intermediate gate scaling, sigmoid evaluation,
broadcast multiply, and residual addition all run in `float32`, rounding to
`bfloat16` once on output, matching the upstream SGLang CUDA kernel contract.
Both sigmoid tails saturate cleanly to 0 and 2 without producing NaN.

Unlike `hc_post` and `mhc_post`, which take precomputed `post` and $4\times 4$
`comb` stream-mixing matrices for fixed hidden sizes (`4096`/`7168`),
`hc_combine` projects raw injection logits internally, applies the `2 * sigmoid`
gate ladder with unit residual stream identity, and supports per-branch hidden
sizes in `[16, 4096]`.

## Parameters and return value

The public operator validates the contract below before selecting an
implementation, so both backends reject the same inputs. Invalid inputs raise
`ValueError` before any kernel launch.

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `block_output` | `torch.Tensor` | `[rows, hidden_size]` | `torch.bfloat16` | Required | Sub-block output; contiguous 2-D tensor on an NPU device. |
| `residual` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | Required | Multi-branch residual; contiguous 2-D tensor on block_output's device. `rows == 0` returns an empty result. |
| `normed_residual` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | Required | Normalized residual input to the injection projection; contiguous 2-D tensor on block_output's device. |
| `inject_weight` | `torch.Tensor` | `[hc_count, hc_count * hidden_size]` | `torch.bfloat16` | Required | Injection projection weight; contiguous 2-D tensor on block_output's device. |
| `hc_count` | `int` | — | — | Required | Hyper-connection branch count; a power of two in `[1, 8]`. `bool` is rejected. |
| `hidden_size` | `int` | — | — | Required | Per-branch hidden size; a multiple of 16 in `[16, 4096]`. `bool` is rejected. |
| `out` | `torch.Tensor` or `None` | `[rows, hc_count * hidden_size]` or — | `torch.bfloat16` or — | `None` | Optional pre-allocated contiguous output buffer on residual's device with residual's shape, `requires_grad=False`, and no memory overlap with any input tensor. When `None`, a new tensor is allocated. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only; accepts `"auto"`, `"ascendc"` or `"torch_native"`. `auto` selects Ascend C. |

Autograd is not supported: any input or pre-allocated `out` with
`requires_grad=True` is rejected. When `out` is provided, it must match
`residual`'s shape (`[rows, hc_count * hidden_size]`), dtype (`torch.bfloat16`),
and device, be contiguous, and not alias or overlap `block_output`, `residual`,
`normed_residual`, or `inject_weight`.

| Return value | Type | Shape | Dtype | Description and constraints |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[rows, hc_count * hidden_size]` | `torch.bfloat16` | Combined output on residual's device. Returns `out` (written in place) when supplied, or a newly allocated contiguous tensor otherwise. Inputs are not modified (`residual` is not updated in place). |

## ascendc backend

- Projects `[rows, hc_count]` injection logits via `torch.nn.functional.linear`
  (`aclnn`) and fuses the gate scaling, sigmoid, broadcast multiply and
  residual add in the internal Ascend C helper
  `torch.ops.npu.hyper_conn_branch_combine` (sources in
  [`csrc/hyper_conn`](../../../../../csrc/hyper_conn/README.md)), keeping
  intermediate arithmetic in `float32` throughout and rounding to `bfloat16`
  once on output. Pass only tensors validated through the public operator
  rather than calling the helper binding directly.
- One kernel task covers one `(row, column tile)`. Each row's `hc_count` gates
  are evaluated on an 8-lane float32 Unified Buffer block (and reused across
  consecutive tiles of the same row on a core), while all branches of the
  residual tile are transferred in single strided MTE2/MTE3 bursts.
- Block counts and column tiles come from a measured table keyed on
  `hidden_size` and `rows` (clamped to the task count and runtime vector core
  count via `GetCoreNumAiv()`). Row counts between 65 and 1022 interpolate from
  the 128-row entry.

## torch_native backend

- Independent PyTorch implementation; it calls no custom native operator.
- Runs `torch.nn.functional.linear` for the injection projection and evaluates
  the intermediate gate scaling, `1 / (1 + exp(-x))` sigmoid, broadcast
  multiply, and residual addition in `float32`, rounding to `bfloat16` once on
  output.
- Allocates float32 temporaries of the full residual size, so prefill-sized
  calls are memory heavy. It is a reference, not a performance path.
- Transcendental implementations differ across backends, so bitwise equality is
  not guaranteed; accuracy tests enforce a tightened `BF16_TIGHT` tolerance
  (`rtol = 8e-3`, 1–2 BF16 ulp).

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators import list_impls
from sgl_kernel_npu.operators.hyper_conn import hc_combine

rows, hc_count, hidden_size = 32, 4, 2048
total = hc_count * hidden_size
block_output = torch.randn(
    rows, hidden_size, device="npu", dtype=torch.bfloat16
)
residual = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
normed_residual = torch.randn(rows, total, device="npu", dtype=torch.bfloat16)
inject_weight = (
    torch.randn(hc_count, total, device="npu", dtype=torch.bfloat16) * 0.1
)

print(list_impls(hc_combine))  # ['ascendc', 'auto', 'torch_native']
out = hc_combine(
    block_output, residual, normed_residual, inject_weight, hc_count, hidden_size
)
out_native = hc_combine(
    block_output,
    residual,
    normed_residual,
    inject_weight,
    hc_count,
    hidden_size,
    impl="torch_native",
)
torch.testing.assert_close(out, out_native, rtol=5e-2, atol=5e-2)
```
