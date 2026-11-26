# grouped_gemma_rmsnorm

## Hardware support

arch22 is split because the two members of that family are validated
differently. A3 means Ascend910_93 series, A2 means Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton` (`auto`) | Supported | Unverified | Supported |
| `ascendc` | Supported | Unverified | Unsupported |
| `torch_native` | Supported | Unverified | Supported |

`auto` selects `triton` on both `arch22` and `arch35` (`x[R, 10240]`,
`weight[10240]`, `group_size=2560`). The explicit `ascendc` backend uses the
CANN/ACLNN `hyper_conn_grouped_norm` operator (`SUPPORTED_ARCHS 2201`), which
compiles for both `ascend910b` (A2) and `ascend910_93` (A3) and rejects arch35
via `UnsupportedGenerationError` in the operator registry. A2 is marked
`Unverified` because hardware CI covers A3 only.

## Signature

Omitting `impl` or passing `"auto"` selects `triton`. Select the CANN/ACLNN
Ascend C kernel explicitly with `impl="ascendc"`.

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

RMS-normalize fixed-width groups along the last dimension of `x` and scale by
`1 + weight` in float32, rounding once to BF16 when writing the output:

```text
inv    = rsqrt(mean(x_g^2) + eps)
out[g] = bf16((x_g * inv) * (1 + weight[(g * group_size) % weight.numel()]))
```

`triton` (`auto`) targets the Qwen3.8-Flash-Next shape `x[R, 10240]`,
`weight[10240]`, `group_size=2560`. `ascendc` supports general group widths
(multiples of 8 up to 4096), shared (`[group_size]`) or per-branch cycling
weights, 2-D/3-D contiguous inputs, and zero-copy pre-allocated `out` buffers.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `x` | `torch.Tensor` | `[..., N]`, BF16 | Required | Contiguous NPU tensor (`ndim >= 2`); `N` divisible by `group_size`. |
| `weight` | `torch.Tensor` | `[W]`, BF16 | Required | Contiguous 1-D tensor on `x.device`; `W` is a positive multiple of `group_size` whose period divides the group count. |
| `group_size` | `int` or `None` | — | `None` | Multiple of 8 in `[8, 4096]` (`None` uses `x.shape[-1]`). `triton` requires `2560`. |
| `eps` | `float` | — | `1e-6` | Positive finite float32-representable stability term. |
| `out` | `torch.Tensor` or `None` | `x.shape`, BF16 | `None` | Optional contiguous NPU buffer not aliasing or overlapping `x` or `weight`. |
| `impl` | `str` | — | `"auto"` | Keyword-only: `"auto"`, `"triton"`, `"ascendc"`, or `"torch_native"`. |

| Return | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `x.shape` | `torch.bfloat16` | Normalized output on `x.device`; returns `out` when supplied. |

## Implementations

- **`triton` (`auto`)**: Default on arch22 and arch35 for `x[R, 10240]`,
  `weight[10240]`, `group_size=2560`. Graph-capturable.
- **`ascendc`**: Explicit arch22 CANN/ACLNN AIV kernel
  (`torch.ops.npu.hyper_conn_grouped_norm`) with Ascend910_9382 tuned block
  tables, general widths up to 4096, and zero-copy `out` support. Graph-capturable.
- **`torch_native`**: Independent CPU FP64 accuracy reference.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import grouped_gemma_rmsnorm

x = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(10240, device="npu", dtype=torch.bfloat16)
y_triton = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6)
y_ascendc = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, impl="ascendc")
```
