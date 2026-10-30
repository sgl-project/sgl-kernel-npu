# chunk_delta_h

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import chunk_delta_h
def chunk_delta_h(k, w, u, g=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, cu_seqlens=None, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

Compute per-chunk gated delta-rule recurrent-state snapshots, residual values and optional final state.

## Parameters and return value

k: [B,T,Hg,K], w: [B,T,H,K], u: [B,T,H,V], g: [B,T,H]; g is required by the current wrapper. Returns (h, v_new or None, final_state or None).

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import chunk_delta_h

k = torch.randn(1, 64, 2, 64, device="npu", dtype=torch.float16)
w = torch.randn(1, 64, 4, 64, device="npu", dtype=torch.float16)
u = torch.randn_like(w)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
h, v_new, final = chunk_delta_h(k, w, u, g, output_final_state=True, impl="triton")
```
