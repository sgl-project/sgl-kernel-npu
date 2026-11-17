# chunk_delta_h

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_delta_h

def chunk_delta_h(k, w, u, g=None, initial_state=None, output_final_state=False, chunk_size=64, save_new_value=True, cu_seqlens=None, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

Save the pre-chunk state S, compute $D=U-WS$, then update $S\leftarrow S\exp(g_L)+K^\mathsf{T}(D\odot\exp(g_L-g))$. Positive gate differences are masked to zero.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `k` | Tensor | `[B,T,Hg,K]; float16/bfloat16` | Required | Grouped key heads; H must be divisible by Hg. |
| `w` | Tensor | `[B,T,H,K]; k.dtype` | Required | Transformed keys for the state projection. |
| `u` | Tensor | `[B,T,H,V]; k.dtype` | Required | Transformed values; does not modify inputs. |
| `g` | Tensor | `[B,T,H]; float32` | `None` | Log-space scalar gates; required, cannot be None. |
| `initial_state` | Tensor or None | `[N,H,K,V]; float32` | `None` | Initial state; None selects zero state. |
| `output_final_state` | bool | `—` | `False` | Return final FP32 state when true. |
| `chunk_size` | int | `—` | `64` | Positive power-of-two chunk length; GDN uses 64. |
| `save_new_value` | bool | `—` | `True` | Return residual values when true. |
| `cu_seqlens` | Tensor or None | `[N+1]; int32/int64` | `None` | Cumulative sequence offsets; flattened input requires B=1. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `h` | Tensor or None | `[B,NT,H,K,V]; k.dtype` | Pre-chunk snapshots; NT is total chunks for flattened variable-length input. |
| `v_new` | Tensor or None | `u.shape; u.dtype or None` | Residual D when save_new_value is true. |
| `final_state` | Tensor or None | `[N,H,K,V]; float32 or None` | Final states when output_final_state is true. |

## triton backend

Uses FP32 state accumulation with input-dtype state/projection rounding. Model tests use K=V=128; V=64 is an extra check. This fixed-width kernel does not support K>128 or V>128.

The public Triton adapter rejects K/V outside [1,128] and accepts only the validated chunk_size=64. Other positive power-of-two chunk sizes remain available to torch_native.

## torch_native backend

CPU FP64 reference preserves both input-dtype rounding points. Outputs retain API dtypes and return to the input device; host transfers are intended for accuracy checks.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import chunk_delta_h

k = torch.randn(1, 64, 2, 128, device="npu", dtype=torch.float16)
w = torch.randn(1, 64, 4, 128, device="npu", dtype=torch.float16)
u = torch.randn_like(w)
g = -torch.rand(1, 64, 4, device="npu").cumsum(1)
h, v_new, final = chunk_delta_h(k, w, u, g, output_final_state=True, impl="triton")
```
