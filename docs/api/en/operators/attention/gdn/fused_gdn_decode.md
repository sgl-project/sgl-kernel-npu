# fused_gdn_decode

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

def fused_gdn_decode(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source, initial_state_indices, scale=None, use_qk_l2norm_in_kernel=False, cu_seqlens=None, *, impl='auto'):
    ...
```

`auto` aliases `triton`; it does not automatically fall back to the reference.

## Description

With optional q/k normalization, compute decay $d=\exp(-\exp(A_{log})\,softplus(a+dt_{bias}))$, $S_d=dS$, $r=\sigma(b)(v-S_dk)$, $S=S_d+rk^\mathsf{T}$ and $o=S(q\,scale)$.

## Parameters and return values

Triton tensor inputs should share one NPU and use contiguous layouts; outputs reside on the input device.

| Parameter | Type | Shape / dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- |
| `A_log` | Tensor | `[HV]; float32` | Required | Log decay coefficients. |
| `a` | Tensor | `[N,HV]; q.dtype` | Required | Softplus gate inputs. |
| `dt_bias` | Tensor | `[HV]; float32` | Required | Bias added before softplus. |
| `softplus_beta` | float | `—` | Required | Positive softplus slope. |
| `softplus_threshold` | float | `—` | Required | Use linear softplus above this threshold. |
| `q` | Tensor | `[B,T,H,K]; float16/bfloat16` | Required | Queries; one token per sequence. |
| `k` | Tensor | `[B,T,Hg,K]; float16/bfloat16` | Required | Grouped key heads; H must be divisible by Hg. |
| `v` | Tensor | `[B,T,HV,V]; q.dtype` | Required | Value heads; HV must be divisible by H. |
| `b` | Tensor | `[N,HV]; q.dtype` | Required | Sigmoid update gate inputs. |
| `initial_state_source` | Tensor | `[slots,HV,V,K]; float32` | Required | Contiguous state pool, updated in place; noncontiguous inputs are copied. |
| `initial_state_indices` | Tensor | `[N]; int32/int64` | Required | State slot per sequence; negative slots are zero-initialized and not written. |
| `scale` | float or None | `—` | `None` | Positive query scale; None selects K**-0.5. |
| `use_qk_l2norm_in_kernel` | bool | `—` | `False` | Normalize q and k using norm + 1e-6. |
| `cu_seqlens` | Tensor or None | `[N+1]; int32/int64` | `None` | Cumulative sequence offsets; flattened input requires B=1. |
| `impl` | str | `—` | `'auto'` | Keyword-only: auto, triton or torch_native. |

| Return value | Type | Shape / dtype | Description |
| --- | --- | --- | --- |
| `output` | Tensor or None | `v.shape; q.dtype` | Single-token output; selected contiguous state slots are updated in place. |

## triton backend

Uses FP32 recurrence. Fixed-length input requires T=1; variable-length input uses B=1 and exactly one token per sequence. State memory is interpreted in V,K order.

## torch_native backend

Independent PyTorch operations provide the reference result. Host metadata transfers may synchronize the device; this backend is intended for accuracy comparison.

## Example

```python
import torch
from sgl_kernel_npu.operators.attention.gdn import fused_gdn_decode

q = torch.randn(1, 1, 2, 64, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(1, 1, 4, 64, device="npu", dtype=torch.float16)
a = torch.randn(1, 4, device="npu", dtype=torch.float16)
b = torch.randn_like(a)
A = torch.randn(4, device="npu")
bias = torch.randn(4, device="npu")
state = torch.zeros(1, 4, 64, 64, device="npu")
idx = torch.tensor([0], device="npu", dtype=torch.int32)
y = fused_gdn_decode(A, a, bias, 1.0, 20.0, q, k, v, b, state, idx, impl="triton")
```
