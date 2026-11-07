# chunk_scaled_dot_kkt

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_scaled_dot_kkt

def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

Within each chunk, compute $A_{ij}=\beta_i(k_i^\mathsf{T}k_j)\exp(g_i-g_j)$ for $i>j$ and $g_i-g_j\le0$; all other entries are zero.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `k` | Tensor | `[B,T,Hg,K]; float16/bfloat16` | Required | Grouped key heads; H must be divisible by Hg. |
| `beta` | Tensor | `[B,T,H]; k.dtype` | Required | Per-token update weights. |
| `g_cumsum` | Tensor | `[B,T,H]; float32` | `None` | Chunk-local cumulative log gates; required by this backend. |
| `cu_seqlens` | Tensor or None | `[N+1]; int32/int64` | `None` | Cumulative sequence offsets; flattened input requires B=1. |
| `chunk_size` | int | `—` | `64` | Positive power-of-two chunk length; GDN uses 64. |
| `output_dtype` | torch.dtype or None | `—` | `torch.float32` | Output dtype; see backend restrictions below. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `A` | Tensor or None | `[B,T,H,chunk_size]; output_dtype` | Strict-lower chunk products, with zero padded columns. |

## triton backend

Uses FP32 accumulation and casts to output_dtype. g_cumsum is required; contiguous inputs and monotone nonpositive chunk gates are expected.

## torch_native backend

Independent PyTorch operations provide the reference result. Host metadata transfers may synchronize the device; this backend is intended for accuracy comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_scaled_dot_kkt

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float16)
b = torch.rand(1, 64, 4, device="npu", dtype=torch.float16)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
y = chunk_scaled_dot_kkt(k, b, g, impl="triton")
```
