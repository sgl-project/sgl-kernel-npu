# chunk_scaled_dot_kkt

## Hardware support

| Backend | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |

## Signature

```python
from sgl_kernel_npu.operators.attention import chunk_scaled_dot_kkt
def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32, *, impl="auto"): ...
```

`auto` selects `triton`.

## Description

Compute strictly lower triangular beta_i * exp(g_i - g_j) * dot(k_i, k_j) within each sequence chunk; positive gate differences are masked to zero.

## Parameters and return value

k: [B,T,Hg,K]; beta and g_cumsum: [B,T,H]; H must be divisible by Hg. The current wrapper requires g_cumsum even though the signature has a None default. Returns [B,T,H,chunk_size].

`impl` is `auto`, `triton`, or `torch_native`. The independent torch reference may synchronize or transfer sequence metadata to the host and is intended for correctness comparison, not performance.

## Triton

Migrated from the legacy implementation; supported shapes, dtypes, layouts and numerical behavior need hardware validation.

## torch_native

Independent PyTorch reference for numerical comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention import chunk_scaled_dot_kkt

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float16)
b = torch.rand(1, 64, 4, device="npu", dtype=torch.float16)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
y = chunk_scaled_dot_kkt(k, b, g, impl="triton")
```
