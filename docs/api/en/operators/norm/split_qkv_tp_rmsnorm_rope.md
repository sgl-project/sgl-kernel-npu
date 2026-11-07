# split_qkv_tp_rmsnorm_rope

## Hardware support

| Implementation | arch22 | arch35 |
| --- | --- | --- |
| `torch_native` | Supported | Supported |
| `triton` | Supported | Supported |

## Signature

```python
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

def split_qkv_tp_rmsnorm_rope(
    input, cos, sin, q_hidden_size, kv_hidden_size, head_dim, eps, q_weight,
    k_weight, rotary_dim, tp_world, tp_group, *, impl="auto",
): ...
```

`auto` selects `triton`.

## Description

Prepare Q, K and V from a concatenated local projection for attention, including the MiniMax-M2 QK-normalization path. Each token uses separate Q and K mean-square statistics, reduced across the TP group when tp_world > 1, to apply weighted RMSNorm. Normalized Q/K values are rounded to the input dtype before NeoX RoPE rotates their leading rotary_dim channels; any remaining channels are left unrotated. V is copied without normalization or rotation. The function returns new Q, K and V tensors on the input device.

## Parameters and return value

Use tensors on the same NPU.

| Parameter | Type | Shape | Dtype | Default | Description |
| --- | --- | --- | --- | --- | --- |
| `input` | `torch.Tensor` | `[B, Q + 2*K]` | `float32` / `float16` / `bfloat16` | Required | Dense local projection in Q, K, V order. Q and K denote the local widths. |
| `cos` | `torch.Tensor` | `[B, R]` | `floating point` | Required | Dense cosine rows. The first R/2 entries of each row apply to every head. |
| `sin` | `torch.Tensor` | `[B, R]` | `floating point` | Required | Dense sine rows, paired with cos. |
| `q_hidden_size` | `int` | — | — | Required | Local Q width (Q), a multiple of the local K width. |
| `kv_hidden_size` | `int` | — | — | Required | Local width (K) of each of K and V. Local widths must contain whole heads. |
| `head_dim` | `int` | — | — | Required | Channels per head, a power of two. |
| `eps` | `float` | — | — | Required | Added to the mean square before normalization. |
| `q_weight` | `torch.Tensor` | `[Q]` | `floating point` | Required | Local Q channel weights. |
| `k_weight` | `torch.Tensor` | `[K]` | `floating point` | Required | Local K channel weights. |
| `rotary_dim` | `int` | — | — | Required | Number of leading channels rotated per head (R), even and at most head_dim. |
| `tp_world` | `int` | — | — | Required | Number of ranks in the TP group. Use 1 on a single device. |
| `tp_group` | `ProcessGroup / None` | — | — | Required | Use None on one device. With multiple ranks, all group members must call collectively with matching batch sizes and shard widths. |
| `impl` | `str` | — | — | `"auto"` | Keyword-only. Choices are `"auto"`, `"triton"`, `"torch_native"`. |

| Return value | Type | Description |
| --- | --- | --- |
| `(q, k, v)` | `tuple[Tensor, Tensor, Tensor]` | New tensors with input dtype and device, shaped [B, Q], [B, K] and [B, K]. |

## triton backend

Uses the supplied TP group to combine Q/K statistics when tp_world exceeds one.

## torch_native backend

Computes on the input device with the same TP group and output layout.

## Example

```python
import torch
import torch_npu
from sgl_kernel_npu.operators.norm import split_qkv_tp_rmsnorm_rope

device = "npu"
qkv = torch.randn((2, 256), device=device, dtype=torch.bfloat16)
cos = torch.ones((2, 64), device=device, dtype=torch.bfloat16)
sin = torch.zeros_like(cos)
qw = torch.ones(128, device=device, dtype=qkv.dtype)
kw = torch.ones(64, device=device, dtype=qkv.dtype)
q, k, v = split_qkv_tp_rmsnorm_rope(
    qkv, cos, sin, 128, 64, 64, 1e-6, qw, kw, 64, 1, None, impl="triton",
)
print(q.shape, k.shape, v.shape)  # [2, 128], [2, 64], [2, 64]
```
