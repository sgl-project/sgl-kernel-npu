# hc_combine

## Hardware support

arch22 is split because the two members of that family are validated
differently. A3 means Ascend910_93 series, A2 means Ascend910B series.

| Implementation | arch22 A3 | arch22 A2 | arch35 |
| --- | --- | --- | --- |
| `triton` (`auto`) | Supported | Unverified | Supported |
| `ascendc` | Supported | Unverified | Unsupported |
| `torch_native` | Supported | Unverified | Supported |

`auto` selects `triton` on both `arch22` and `arch35` (`hc=4, hs=2560`).
Explicit `impl="ascendc"` uses BF16 `F.linear` projection plus the arch22
CANN/ACLNN AIV operator `hyper_conn_branch_combine` (`SUPPORTED_ARCHS 2201`);
arch35 rejects `impl="ascendc"` via `UnsupportedGenerationError` in the registry.

## Signature

Omitting `impl` or passing `"auto"` selects `triton`. Select the Ascend C path
explicitly with `impl="ascendc"`.

```python
from typing import Optional
import torch
from sgl_kernel_npu.operators.hyper_conn import hc_combine

def hc_combine(
    block: torch.Tensor,
    residual: torch.Tensor,
    normed: torch.Tensor,
    weight: torch.Tensor,
    hc: int,
    hs: int,
    out: Optional[torch.Tensor] = None,
    *,
    impl: str = "auto",
) -> torch.Tensor:
    ...
```

## Description

Project per-branch injection gates from `normed` and add the gated `block`
output onto every branch of `residual`:

```text
gate = 2 * sigmoid((normed @ weight.T) / hc)
out[r, c * hs + d] = residual[r, c * hs + d] + gate[r, c] * block[r, d]
```

- **`triton` (`auto`)**: keeps intermediate gate projection in FP32 and rounds
  only the output to BF16 (`hc=4, hs=2560`).
- **`ascendc`**: projects `gate_logits = F.linear(normed, weight)` in BF16 and
  runs `torch.ops.npu.hyper_conn_branch_combine` in FP32 internally, rounding
  once to BF16 when writing `out`. Supports power-of-two `hc` in `[1, 8]`,
  multiples of 16 up to 4096 for `hs`, and zero-copy pre-allocated `out`.

## Parameters and return value

| Parameter | Type | Shape / dtype | Default | Constraints |
| --- | --- | --- | --- | --- |
| `block` | `torch.Tensor` | `[R, hs]`, BF16 | Required | Contiguous NPU input. |
| `residual` | `torch.Tensor` | `[R, hc * hs]`, BF16 | Required | Contiguous on `block.device`. |
| `normed` | `torch.Tensor` | `[R, hc * hs]`, BF16 | Required | Contiguous on `block.device`. |
| `weight` | `torch.Tensor` | `[hc, hc * hs]`, BF16 | Required | Contiguous on `block.device`. |
| `hc` | `int` | — | Required | Power of two in `[1, 8]` (`triton` requires `4`). |
| `hs` | `int` | — | Required | Multiple of 16 in `[16, 4096]` (`triton` requires `2560`). |
| `out` | `torch.Tensor` or `None` | `[R, hc * hs]`, BF16 | `None` | Optional contiguous NPU buffer not aliasing or overlapping any input. |
| `impl` | `str` | — | `"auto"` | Keyword-only: `"auto"`, `"triton"`, `"ascendc"`, or `"torch_native"`. |

| Return | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `output` | `torch.Tensor` | `[R, hc * hs]` | `torch.bfloat16` | Combined output on `residual.device`; returns `out` when supplied. |

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.hyper_conn import hc_combine

residual = torch.randn(32, 10240, device="npu", dtype=torch.bfloat16)
block = torch.randn(32, 2560, device="npu", dtype=torch.bfloat16)
weight = torch.zeros(4, 10240, device="npu", dtype=torch.bfloat16)
y_triton = hc_combine(block, residual, residual, weight, 4, 2560)
y_ascendc = hc_combine(block, residual, residual, weight, 4, 2560, impl="ascendc")
```
