# split_qkv_rmsnorm_mrope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `triton` | Supported | Supported |
| `torch_native` | Supported | Supported |

## Signature

`auto` selects `triton`.

```python
from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_mrope


def split_qkv_rmsnorm_mrope(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    num_q_heads,
    num_kv_heads,
    head_size,
    eps,
    mrope_section,
    is_interleaved,
    rope_dim=None,
    q_bias=None,
    k_bias=None,
    has_gate=False,
    *,
    impl="auto"
): ...
```

## Description

Splits packed Q/K/V and optional per-head gate channels, normalizes Q and K with RMSNorm weights and optional biases, then applies temporal/height/width rotary embeddings. Full and partial rotary dimensions and interleaved or contiguous MRoPE sections are supported. V and gate are copied without normalization.

## Parameters and return value

All tensors must be on the same NPU device.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `qkv` | Tensor | `Required` | BF16/FP16 contiguous [tokens,QH*H*(2 if has_gate else 1)+2*KVH*H]. |
| `q_weight` | Tensor | `Required` | Floating [head_size] query RMSNorm weight. |
| `k_weight` | Tensor | `Required` | Floating [head_size] key RMSNorm weight. |
| `cos_sin` | Tensor | `Required` | Floating [3,tokens,rope_dim]; cosine then sine halves per axis. |
| `num_q_heads` | int | `Required` | Local query head count after tensor parallel partitioning. |
| `num_kv_heads` | int | `Required` | Local KV head count. |
| `head_size` | int | `Required` | Channels per Q/K/V head. |
| `eps` | float | `Required` | RMSNorm stabilization constant. |
| `mrope_section` | list[int] | `Required` | Three temporal/height/width rotary sections. |
| `is_interleaved` | bool | `Required` | Interleave rotary axes instead of using contiguous sections. |
| `rope_dim` | int or None | `None` | Even rotated width <=head_size; None rotates the complete head. |
| `q_bias` | Tensor or None | `None` | Optional floating [head_size] query bias. |
| `k_bias` | Tensor or None | `None` | Key bias paired with q_bias, same shape. |
| `has_gate` | bool | `False` | Input contains per-query-head gate channels. |
| `impl` | str | `'auto'` | auto selects triton; explicit backend selection. |

| Return | Type | Description |
| --- | --- | --- |
| result | Tuple[Tensor,Tensor,Tensor,Tensor] | Q [T,QH*H], K/V [T,KVH*H], gate [T,QH*H] or [T,0]; all match qkv dtype. |

## triton

Uses Triton-Ascend kernels.

## torch_native

Uses PyTorch tensor operations on the input device.

## Example

```python
import torch
import torch_npu

from sgl_kernel_npu.operators.norm import split_qkv_rmsnorm_mrope

qkv = torch.randn(7, 1024, device="npu", dtype=torch.bfloat16)
w = torch.ones(256, device="npu", dtype=qkv.dtype)
cs = torch.randn(3, 7, 256, device="npu", dtype=qkv.dtype)
q, k, v, gate = split_qkv_rmsnorm_mrope(
    qkv, w, w, cs, 2, 1, 256, 1e-6, [32, 48, 48], True, impl="triton"
)
```
