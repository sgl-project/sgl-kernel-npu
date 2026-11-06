# chunk_local_cumsum

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

def chunk_local_cumsum(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False, output_dtype=torch.float32, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

Compute independently within each sequence chunk: $o_t=s\sum_{j\le t}g_j$; reverse mode sums $j\ge t$. Scaling is omitted when scale=None.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `g` | Tensor | `[B,T,H] or [B,H,T]; float16/bfloat16/float32` | Required | Log-space scalar gates; cumsum also accepts FP16/BF16. |
| `chunk_size` | int | `—` | Required | Positive power-of-two chunk length; GDN uses 64. |
| `reverse` | bool | `—` | `False` | Accumulate in reverse time order within each chunk. |
| `scale` | float or None | `—` | `None` | Optional multiplier after cumulative sum. |
| `cu_seqlens` | Tensor or None | `[N+1]; int32/int64` | `None` | Cumulative sequence offsets; flattened input requires B=1. |
| `head_first` | bool | `—` | `False` | Interpret g as [B,H,T] instead of [B,T,H]. |
| `output_dtype` | torch.dtype or None | `—` | `torch.float32` | Output dtype; see backend restrictions below. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output` | Tensor or None | `g.shape; output_dtype or g.dtype` | New chunk-local sums. |

## triton backend

FP32 accumulation, then input-dtype rounding before store. Output dtype must equal g.dtype: low-precision inputs require None or g.dtype; default FP32 only matches FP32 inputs.

The public Triton adapter rejects head_first=True with cu_seqlens. Let BLOCK_T=next_power_of_2(floor(2**18/(H*chunk_size))); the quotient must be positive and BLOCK_T >= chunk_size. Head-first input additionally requires T divisible by BLOCK_T because partial time tiles are not supported. These Triton restrictions do not apply to torch_native; all dimensions must be nonempty.

## torch_native backend

Independent PyTorch operations provide the reference result. Host metadata transfers may synchronize the device; this backend is intended for accuracy comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_local_cumsum

g = torch.randn(1, 128, 8, device="npu", dtype=torch.float16)
y = chunk_local_cumsum(g, 64, output_dtype=None, impl="triton")
```
