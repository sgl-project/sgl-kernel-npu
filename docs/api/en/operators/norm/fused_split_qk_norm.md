# fused_split_qk_norm

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Unverified | Unverified |
| `torch_native` | Unverified | Unverified |
| `tilelang` | Not implemented | Not implemented |
| `ascendc` | Not implemented | Not implemented |

## Signature

```python
from sgl_kernel_npu.operators.norm import fused_split_qk_norm

def fused_split_qk_norm(
    fused_qkv_a_proj_out,
    q_a_layernorm,
    kv_a_layernorm,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_dim,
    eps=1e-06,
    *,
    impl="auto",
):
    ...
```

`auto` selects the same lazily imported entry as `triton`; there is no fallback.
The implementations declare all known generations, but hardware execution remains unverified.

## Description

Split the MLA latent projection into Q, non-rotary K and rotary K. Independently RMS-normalize Q and non-rotary K with the supplied objects' weights and optional biases. The objects are attribute containers; their forward methods are never called. The explicit eps argument is used for both normalizations.

$$
Z=[Z_q\mid Z_k\mid Z_r],\qquad Y_q=\frac{Z_q}{\sqrt{\mathrm{mean}(Z_q^2)+\epsilon}}w_q+b_q,\qquad Y_k=\frac{Z_k}{\sqrt{\mathrm{mean}(Z_k^2)+\epsilon}}w_k+b_k.
$$


## Parameters and return value

| Parameter | Type | Shape | Dtype | Default | Description and constraints |
| --- | --- | --- | --- | --- | --- |
| `fused_qkv_a_proj_out` | Tensor | [N,Rq+Rkv+Dr] | floating | required | Contiguous latent projection. |
| `q_a_layernorm` | object | weight/bias: at least Rq | floating | required | Exposes weight and optional bias, on the input device. |
| `kv_a_layernorm` | object | weight/bias: at least Rkv | floating | required | Exposes weight and optional independent bias, on the input device. |
| `q_lora_rank` | int | — | — | required | Positive Rq, excluding bool. |
| `kv_lora_rank` | int | — | — | required | Positive Rkv, excluding bool. |
| `qk_rope_dim` | int | — | — | required | Positive Dr, excluding bool; K rotary features are copied without normalization. |
| `eps` | int/float | — | — | 1e-6 | Finite positive Python scalar, excluding bool. |
| `impl` | str | — | — | `"auto"` | Keyword-only: auto, triton, torch_native. |

Floating means float16, bfloat16 or float32; read tensors must share the input device. Norm vectors are contiguous and may use a different floating dtype. N may be zero. Use finite participating values for accuracy comparisons. Positive integer options exclude bool.

| Return value | Type | Shape | Dtype | Description |
| --- | --- | --- | --- | --- |
| `q_lora` | Tensor | [N,Rq] | input.dtype | New contiguous output on the input device. |
| `k_nope` | Tensor | [N,1,Rkv] | input.dtype | New contiguous output on the input device. |
| `k_pe` | Tensor | [N,1,Dr] | input.dtype | New contiguous output on the input device. |

No input tensor, weight or bias is modified. Each backend validates metadata with its own norm/norm_utils.py; the implementations have no shared validation dependency. Triton additionally requires an NPU. Invalid metadata raises AssertionError before a launch; no device values are read back for validation.

## triton backend

Retains fused_split_qk_norm_kernel unchanged: one program per token, FP32 variance and affine arithmetic, then an output-dtype cast. Metadata is checked before launch. Empty token batches return empty outputs without launching a kernel.

The original script under `sgl_kernel_npu.norm` is unchanged. The checks documented here belong to the migrated implementation.

## torch_native backend

Uses torch split and independent FP32 RMSNorm arithmetic. K rotary features are cloned; all returned tensors have fresh storage. Input, norm weights and biases are not modified.

No Triton kernel is called. The reference is usable on CPU as well as NPU. Reduction order, rsqrt approximation and multiply-add fusion may cause numerical differences; dtype-specific tolerance checks do not prove bitwise equivalence to the framework unfused path.

CPU arithmetic support does not change package initialization: ordinary public
imports still require torch_npu, the native library and ACL hardware discovery.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import fused_split_qk_norm

from types import SimpleNamespace

x = torch.randn(4, 56, device='npu', dtype=torch.bfloat16)
qn = SimpleNamespace(weight=torch.ones(32, device='npu'), bias=None)
kn = SimpleNamespace(weight=torch.ones(16, device='npu'), bias=None)
args = (x, qn, kn, 32, 16, 8)
outputs = fused_split_qk_norm(*args)
reference = fused_split_qk_norm(*args, impl="torch_native")
for result, expected in zip(outputs, reference):
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
```
